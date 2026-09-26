"""gplugins-shaped entry points: layout and PDK in, labelled results out.

- :func:`simulate_stack` -- the planar stack at one point of a layout (from the
  PDK's LayerStack), swept in wavelength, angle and in-plane orientation;
- :func:`simulate_grating` -- a line grating in the layout, etched into that
  stack, swept in wavelength and angle;
- :func:`simulate_crossed_grating` -- a 2D-periodic pattern (pillars, holes,
  any polygons) etched into that stack;
- :func:`solve_stack` / :func:`solve_grating` / :func:`solve_crossed_grating`
  -- the same for a
  :class:`~hyperbolic_optics.integrations.gdsfactory.stack.VerticalStack` you
  wrote yourself.

Results carry their axes and the polarization analysis anisotropy calls for:
Jones matrices, Mueller matrices and generalized ellipsometry, with the
package's corrected p/s convention (``r[..., out, in]``, 0 = p, 1 = s).

Planar stacks run on the differentiable JAX solver by default, or on the NumPy
engine (``solver="numpy"``), the package's independent reference; the two agree
to rounding.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack
from numpy.typing import ArrayLike

from hyperbolic_optics import bcm, rcwa, rcwa2d
from hyperbolic_optics.integrations.gdsfactory.geometry import LineGrating, extract_line_grating
from hyperbolic_optics.integrations.gdsfactory.lattice import (
    UnitCell,
    crossed_grating_layers,
    crossed_grating_stack,
)
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec, material_tensor
from hyperbolic_optics.integrations.gdsfactory.stack import (
    VerticalStack,
    grating_layers,
    grating_stack,
    vertical_stack,
)
from hyperbolic_optics.mueller import Mueller

__all__ = [
    "CrossedGratingResponse",
    "GratingResponse",
    "PlanarResponse",
    "simulate_component",
    "simulate_crossed_grating",
    "simulate_grating",
    "simulate_stack",
    "solve_crossed_grating",
    "solve_grating",
    "solve_stack",
    "stack_sax_model",
]


def _mueller(jones: np.ndarray) -> np.ndarray:
    """Mueller matrices ``[..., 4, 4]`` from Jones matrices ``[..., out, in]``."""
    # the package's input-first names: r_ps (p in, s out) is jones[..., 1, 0]
    return Mueller._mueller_from_jones(
        jones[..., 0, 0], jones[..., 1, 0], jones[..., 0, 1], jones[..., 1, 1]
    )


def _ellipsometry(r: np.ndarray) -> dict[str, np.ndarray]:
    """Ψ/Δ from r_pp/r_ss and the anisotropy ratios r_ps/r_ss, r_sp/r_pp (degrees)."""
    r_pp, r_ps, r_sp, r_ss = r[..., 0, 0], r[..., 1, 0], r[..., 0, 1], r[..., 1, 1]
    out = {}
    for key, numerator, denominator in (("", r_pp, r_ss), ("_ps", r_ps, r_ss), ("_sp", r_sp, r_pp)):
        rho = numerator / denominator
        out[f"Psi{key}"] = np.degrees(np.arctan(np.abs(rho)))
        out[f"Delta{key}"] = np.degrees(np.angle(rho))
    return out


@dataclass(frozen=True)
class PlanarResponse:
    """A planar stack's response over ``[wavelength, angle, azimuth]``.

    Attributes:
        wavelength_um, angle_deg, azimuth_deg: The sweep axes. The azimuth turns
            the sample about the stack normal (every layer's tensor), as the
            payload API's ``rotationZ`` does; the plane of incidence stays x-z.
        r, t: Complex Jones matrices ``[F, A, B, out, in]``, 0 = p, 1 = s.
        R, T: The corresponding powers.
    """

    wavelength_um: np.ndarray
    angle_deg: np.ndarray
    azimuth_deg: np.ndarray
    r: np.ndarray
    t: np.ndarray
    R: np.ndarray
    T: np.ndarray

    def jones(self, transmission: bool = False) -> np.ndarray:
        return self.t if transmission else self.r

    def mueller(self, transmission: bool = False) -> np.ndarray:
        """Mueller matrices ``[F, A, B, 4, 4]`` of the reflected (or transmitted) light.

        The transmitted one describes amplitudes, not flux: its M₀₀ equals the
        transmittance only when the incident and exit media match.
        """
        return _mueller(self.jones(transmission))

    def ellipsometry(self) -> dict[str, np.ndarray]:
        """Generalized ellipsometry: ``Psi, Delta`` and ``Psi_ps, Delta_ps, Psi_sp, Delta_sp``."""
        return _ellipsometry(self.r)


@dataclass(frozen=True)
class GratingResponse:
    """A line grating's response over ``[wavelength, angle, order]``.

    Attributes:
        wavelength_um, angle_deg: The sweep axes; ``azimuth_deg`` the fixed
            plane of incidence relative to the grating vector.
        orders: The diffraction orders kept.
        r, t, R, T: ``[F, A, order, out, in]``, 0 = p, 1 = s; R and T are
            diffraction efficiencies.
    """

    wavelength_um: np.ndarray
    angle_deg: np.ndarray
    azimuth_deg: float
    orders: np.ndarray
    r: np.ndarray
    t: np.ndarray
    R: np.ndarray
    T: np.ndarray

    def order(self, m: int) -> int:
        return int(np.flatnonzero(self.orders == m)[0])

    def mueller(self, order: int = 0, transmission: bool = False) -> np.ndarray:
        """Mueller matrices ``[F, A, 4, 4]`` of one diffraction order."""
        amplitudes = self.t if transmission else self.r
        return _mueller(amplitudes[..., self.order(order), :, :])


# ---------------------------------------------------------------------------
# Planar
# ---------------------------------------------------------------------------


def _layer_tensors(stack: VerticalStack, wavelengths: np.ndarray, azimuths: np.ndarray):
    """Lab-frame tensors ``[F, B, L, 3, 3]``, the sample turned by each azimuth."""
    if stack.patterned:
        raise ValueError("This stack has patterned layers: use solve_grating.")
    turn = bcm.rotation_matrix(0.0, 0.0, jnp.radians(jnp.asarray(azimuths)))[None]  # [1, B, 3, 3]
    tensors = []
    for layer in stack.layers:
        eps = material_tensor(layer.material, wavelengths, layer.rotation_deg)[:, None]
        tensors.append(bcm.rotate_tensor(eps, turn))
    if not tensors:
        return jnp.zeros((len(wavelengths), len(azimuths), 0, 3, 3), dtype=complex)
    return jnp.stack(tensors, axis=-3)


def _power(r, t, eps_in, eps_out, sin_theta):
    """R and T from amplitudes; ε are ``[F, 1, 1]``, ``sin_theta`` is ``[1, A, 1]``."""
    kz_in = np.real(np.sqrt(eps_in - eps_in * sin_theta**2 + 0j))
    kz_out = np.real(np.sqrt(eps_out - eps_in * sin_theta**2 + 0j))
    flux = (kz_out / kz_in)[..., None, None]  # [F, A, 1, 1, 1]
    return np.abs(r) ** 2, np.abs(t) ** 2 * flux


def solve_stack(
    stack: VerticalStack,
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuths_deg: ArrayLike = (0.0,),
    solver: str = "jax",
) -> PlanarResponse:
    """Reflection and transmission of an unpatterned stack.

    Args:
        stack: The stack, incident side first.
        wavelengths_um: Vacuum wavelengths (µm).
        angles_deg: Angles of incidence in the incident medium (degrees).
        azimuths_deg: In-plane rotations of the sample (degrees).
        solver: ``"jax"`` (differentiable) or ``"numpy"`` (the eigendecomposition
            engine, the package's reference).
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    azimuths = np.atleast_1d(np.asarray(azimuths_deg, dtype=float))
    eps_in = stack.half_space_permittivity("incident", wavelengths)[:, None, None]  # [F, 1, 1]
    eps_out = stack.half_space_permittivity("exit", wavelengths)[:, None, None]
    tensors = _layer_tensors(stack, wavelengths, azimuths)  # [F, B, L, 3, 3]
    thickness = np.array([float(layer.thickness_um) for layer in stack.layers])
    theta = np.radians(angles)[None, :, None]

    if solver == "jax":
        result = bcm.solve_stack(
            tensors[:, None],  # [F, 1, B, L, 3, 3]
            jnp.asarray(thickness),
            jnp.asarray(wavelengths)[:, None, None],
            jnp.asarray(theta),
            jnp.asarray(eps_in),
            jnp.asarray(eps_out),
        )
        shape = (len(wavelengths), len(angles), len(azimuths), 2, 2)
        r = np.broadcast_to(np.asarray(result.r), shape)
        t = np.broadcast_to(np.asarray(result.t), shape)
    elif solver == "numpy":
        r, t = _numpy_engine(np.asarray(tensors), thickness, wavelengths, angles, eps_in, eps_out)
    else:
        raise ValueError(f"Unknown solver {solver!r}; use 'jax' or 'numpy'.")
    R, T = _power(r, t, eps_in, eps_out, np.sin(theta))
    return PlanarResponse(wavelengths, angles, azimuths, r, t, R, T)


def _numpy_engine(tensors, thickness, wavelengths, angles, eps_in, eps_out):
    """The same stack through Structure, one wavelength and azimuth at a time."""
    from hyperbolic_optics.fields import FieldProfile
    from hyperbolic_optics.structure import Structure

    def as_dict(tensor):
        keys = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
        flat = tensor.reshape(-1)
        return {
            f"eps_{k}": {"real": float(v.real), "imag": float(v.imag)} for k, v in zip(keys, flat)
        }

    F, B = tensors.shape[:2]
    r = np.zeros((F, len(angles), B, 2, 2), dtype=complex)
    t = np.zeros_like(r)
    for f in range(F):
        for b in range(B):
            layers = [
                {"type": "Crystal Layer", "material": as_dict(tensors[f, b, i]), "thickness": d}
                for i, d in enumerate(thickness)
            ]
            payload = {
                "ScenarioData": {
                    "type": "Incident",
                    "incidentAngle": angles.tolist(),
                    "frequency": [1e4 / wavelengths[f]],
                },
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": float(eps_in[f, 0, 0])},
                    *layers,
                    {
                        "type": "Semi Infinite Isotropic Layer",
                        "permittivity": float(eps_out[f, 0, 0]),
                    },
                ],
            }
            structure = Structure()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # arbitrary tensors carry no fitted range
                structure.execute(payload, backend="scattering")
            coefficients = FieldProfile(structure).transmission_coefficients()
            for source, target in (
                ({k: getattr(structure, f"r_{k}") for k in ("pp", "ps", "sp", "ss")}, r),
                ({k: coefficients[f"t_{k}"] for k in ("pp", "ps", "sp", "ss")}, t),
            ):
                # input-first names: ps is p in, s out -> [out s, in p]
                target[f, :, b, 0, 0] = np.ravel(source["pp"])
                target[f, :, b, 1, 0] = np.ravel(source["ps"])
                target[f, :, b, 0, 1] = np.ravel(source["sp"])
                target[f, :, b, 1, 1] = np.ravel(source["ss"])
    return r, t


def simulate_stack(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuths_deg: ArrayLike = (0.0,),
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    solver: str = "jax",
) -> tuple[VerticalStack, PlanarResponse]:
    """The PDK stack at one point of a layout, and its plane-wave response.

    Returns the extracted :class:`VerticalStack` alongside the response, so what
    was simulated can be inspected.
    """
    stack = vertical_stack(component, layer_stack, point_um, materials, ambient)
    return stack, solve_stack(stack, wavelengths_um, angles_deg, azimuths_deg, solver)


# ---------------------------------------------------------------------------
# Gratings
# ---------------------------------------------------------------------------


def solve_grating(
    grating: LineGrating,
    stack: VerticalStack,
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuth_deg: float = 0.0,
    max_order: int = 20,
    edge_width: float = 0.0,
) -> GratingResponse:
    """Diffraction by a line grating etched into a stack, over wavelength and angle."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    eps_in = stack.half_space_permittivity("incident", wavelengths)[:, None]  # [F, 1]
    eps_out = stack.half_space_permittivity("exit", wavelengths)[:, None]
    layers = grating_layers(grating, stack, jnp.asarray(wavelengths)[:, None], edge_width)
    result = rcwa.solve_grating(
        layers,
        grating.period_um,
        jnp.asarray(wavelengths)[:, None],
        jnp.radians(jnp.asarray(angles))[None, :],
        jnp.asarray(eps_in),
        jnp.asarray(eps_out),
        max_order,
        azimuth_rad=np.radians(azimuth_deg),
    )
    shape = (len(wavelengths), len(angles), 2 * max_order + 1, 2, 2)
    arrays = [np.broadcast_to(np.asarray(getattr(result, k)), shape) for k in ("r", "t", "R", "T")]
    return GratingResponse(wavelengths, angles, azimuth_deg, np.asarray(result.orders), *arrays)


def simulate_grating(
    component: gf.Component,
    grating_layer: gf.typings.LayerSpec,
    layer_stack: LayerStack,
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuth_deg: float = 0.0,
    max_order: int = 20,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    edge_width: float = 0.0,
) -> tuple[LineGrating, VerticalStack, GratingResponse]:
    """A line grating in a layout, etched into the PDK's stack: geometry, stack, response."""
    grating, stack = grating_stack(component, grating_layer, layer_stack, materials, ambient)
    response = solve_grating(
        grating, stack, wavelengths_um, angles_deg, azimuth_deg, max_order, edge_width
    )
    return grating, stack, response


def simulate_component(
    component: gf.Component,
    layer: gf.typings.LayerSpec,
    stack,
    wavelength_um: float,
    incidence_rad: float,
    eps_incident: float,
    eps_exit: float,
    max_order: int,
    azimuth_rad: float = 0.0,
) -> tuple[LineGrating, rcwa.GratingResult]:
    """Extract the grating from a layout and solve it with a hand-written stack."""
    grating = extract_line_grating(component, layer)
    result = rcwa.solve_grating(
        grating_layers(grating, stack, wavelength_um),
        grating.period_um,
        wavelength_um,
        incidence_rad,
        eps_incident,
        eps_exit,
        max_order,
        azimuth_rad=azimuth_rad,
    )
    return grating, result


# ---------------------------------------------------------------------------
# 2D-periodic patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossedGratingResponse:
    """A 2D-periodic pattern's response over ``[wavelength, angle, order]``.

    Attributes:
        wavelength_um, angle_deg: The sweep axes; ``azimuth_deg`` the fixed
            plane of incidence, from the lattice's x axis.
        orders: ``[N, 2]``, the diffraction orders ``(m, n)`` kept.
        r, t, R, T: ``[F, A, N, out, in]``, 0 = p and 1 = s, in each order's
            own p/s basis; R and T are diffraction efficiencies.
    """

    wavelength_um: np.ndarray
    angle_deg: np.ndarray
    azimuth_deg: float
    orders: np.ndarray
    r: np.ndarray
    t: np.ndarray
    R: np.ndarray
    T: np.ndarray

    def order(self, m: int, n: int) -> int:
        return int(np.flatnonzero((self.orders[:, 0] == m) & (self.orders[:, 1] == n))[0])

    def mueller(self, order: tuple[int, int] = (0, 0), transmission: bool = False) -> np.ndarray:
        """Mueller matrices ``[F, A, 4, 4]`` of one diffraction order."""
        amplitudes = self.t if transmission else self.r
        return _mueller(amplitudes[..., self.order(*order), :, :])


def solve_crossed_grating(
    cell: UnitCell,
    stack: VerticalStack,
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuth_deg: float = 0.0,
    max_order: tuple[int, int] = (5, 5),
    factorization: str = "normal",
) -> CrossedGratingResponse:
    """Diffraction by a 2D-periodic pattern etched into a stack, over wavelength and angle."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    eps_in = stack.half_space_permittivity("incident", wavelengths)[:, None]  # [F, 1]
    eps_out = stack.half_space_permittivity("exit", wavelengths)[:, None]
    layers = crossed_grating_layers(cell, stack, jnp.asarray(wavelengths)[:, None])
    result = rcwa2d.solve_crossed_grating(
        layers,
        cell.period_um,
        jnp.asarray(wavelengths)[:, None],
        jnp.radians(jnp.asarray(angles))[None, :],
        jnp.asarray(eps_in),
        jnp.asarray(eps_out),
        max_order,
        azimuth_rad=np.radians(azimuth_deg),
        factorization=factorization,
    )
    size = result.orders.shape[0]
    shape = (len(wavelengths), len(angles), size, 2, 2)
    arrays = [np.broadcast_to(np.asarray(getattr(result, k)), shape) for k in ("r", "t", "R", "T")]
    return CrossedGratingResponse(
        wavelengths, angles, azimuth_deg, np.asarray(result.orders), *arrays
    )


def simulate_crossed_grating(
    component: gf.Component,
    pattern_layer: gf.typings.LayerSpec,
    layer_stack: LayerStack,
    wavelengths_um: ArrayLike,
    angles_deg: ArrayLike,
    azimuth_deg: float = 0.0,
    max_order: tuple[int, int] = (5, 5),
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    period_um: tuple[float, float] | None = None,
    factorization: str = "normal",
) -> tuple[UnitCell, VerticalStack, CrossedGratingResponse]:
    """A 2D-periodic pattern in a layout, etched into the PDK's stack: cell, stack, response."""
    cell, stack = crossed_grating_stack(
        component, pattern_layer, layer_stack, materials, ambient, period_um
    )
    response = solve_crossed_grating(
        cell, stack, wavelengths_um, angles_deg, azimuth_deg, max_order, factorization
    )
    return cell, stack, response


# ---------------------------------------------------------------------------
# SAX
# ---------------------------------------------------------------------------


def stack_sax_model(stack: VerticalStack, doublings: int = bcm.DEFAULT_DOUBLINGS):
    """A SAX model of an unpatterned stack, e.g. from :func:`vertical_stack`.

    Returns ``model(wl, kx, azimuth) -> SDict`` over ``top@p``, ``top@s``,
    ``bottom@p`` and ``bottom@s`` (see :mod:`hyperbolic_optics.integrations.sax`),
    with every material evaluated with its dispersion at SAX's ``wl``.
    """
    from hyperbolic_optics.integrations import sax as hsax

    if stack.patterned:
        raise ValueError("stack_sax_model takes an unpatterned stack; this one has a grating")

    def evaluate(spec, rotation=(0.0, 0.0, 0.0)):
        return lambda wl: material_tensor(spec, wl, rotation)

    layers = [
        (evaluate(layer.material, layer.rotation_deg), layer.thickness_um) for layer in stack.layers
    ]
    return hsax.planar_model(
        layers, evaluate(stack.incident), evaluate(stack.exit), doublings=doublings
    )
