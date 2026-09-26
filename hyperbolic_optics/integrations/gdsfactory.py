"""gdsfactory bridge: line gratings between GDS layouts and the RCWA solver.

Three pieces, which together close the loop layout → simulation → optimisation →
layout:

- :func:`line_grating`, a gdsfactory PCell for a grating of identical parallel
  ridges, with its period and ridge width snapped to the layout grid so that the
  exported geometry is exactly periodic;
- :func:`extract_line_grating`, which reads that geometry back from any
  gdsfactory component (or imported GDS), and refuses anything that is not a
  clean line grating rather than guessing;
- :class:`StackLayer` and :func:`grating_layers`, which turn the geometry and a
  vertical stack -- thickness, material and crystal orientation per layer -- into
  :mod:`hyperbolic_optics.rcwa` layers at a given wavelength.

Materials may be named built-in phonon-polariton media (``"hBN"``, ``"MoO3"``,
``"Calcite"``, …), evaluated with their dispersion by
:mod:`hyperbolic_optics.dispersion` so that wavelength derivatives are complete;
a number (isotropic ε); a 3×3 tensor; or any callable ``wavelength_um → ε``.

The solver treats the grating as infinitely periodic under plane-wave
illumination, so the result describes the interior of a large array -- the
finite number of periods drawn in the layout does not enter.

Requires the ``gdsfactory`` extra, and an active gdsfactory PDK for
:func:`line_grating` (for example ``gf.gpdk.PDK.activate()``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gdsfactory as gf
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "hyperbolic_optics.integrations.gdsfactory needs gdsfactory: "
        'pip install "hyperbolic-optics[gdsfactory]"'
    ) from error

from hyperbolic_optics import bcm, dispersion, rcwa

__all__ = [
    "LineGrating",
    "StackLayer",
    "extract_line_grating",
    "grating_layers",
    "line_grating",
    "material_tensor",
    "simulate_component",
]

MaterialSpec = str | complex | float | np.ndarray | jax.Array | Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineGrating:
    """A grating of identical ridges along y, repeating along x.

    Attributes:
        period_um: Grating period in µm.
        fill: Ridge width as a fraction of the period.
        offset: Position of the first ridge's left edge as a fraction of the
            period (mod 1). It shifts only the phases of the diffracted orders.
        n_periods: Number of ridges drawn.
        length_um: Ridge length along y in µm.
    """

    period_um: float
    fill: float
    offset: float = 0.0
    n_periods: int = 1
    length_um: float = 0.0


@gf.cell
def line_grating(
    period_um: float = 2.0,
    fill: float = 0.5,
    n_periods: int = 20,
    length_um: float = 20.0,
    layer: gf.typings.LayerSpec = (1, 0),
) -> gf.Component:
    """PCell: ``n_periods`` ridges of width ``fill · period_um`` and length ``length_um``.

    The period and the ridge width are snapped to the layout grid (the active
    PDK's database unit), so every period drawn is identical; the snapped values
    are recorded in ``component.info`` and are what :func:`extract_line_grating`
    recovers.
    """
    if not 0.0 < fill < 1.0:
        raise ValueError(f"fill must lie strictly between 0 and 1, got {fill}")
    component = gf.Component()
    dbu = component.kcl.dbu
    period_dbu = round(period_um / dbu)
    width_dbu = round(fill * period_um / dbu)
    length_dbu = round(length_um / dbu)
    if not 0 < width_dbu < period_dbu:
        raise ValueError("ridge width rounds to zero or to the full period on this grid")
    for index in range(n_periods):
        x0, x1 = index * period_dbu * dbu, (index * period_dbu + width_dbu) * dbu
        top = length_dbu * dbu
        component.add_polygon([(x0, 0.0), (x1, 0.0), (x1, top), (x0, top)], layer=layer)
    component.info["period_um"] = period_dbu * dbu
    component.info["fill"] = width_dbu / period_dbu
    component.info["n_periods"] = n_periods
    component.info["length_um"] = length_dbu * dbu
    return component


def extract_line_grating(
    component: gf.Component, layer: gf.typings.LayerSpec, tolerance_um: float | None = None
) -> LineGrating:
    """Read a line grating back from the polygons of ``component`` on ``layer``.

    Every polygon must be an axis-aligned rectangle; all must span the same y
    range, have the same width, and start at equally spaced x. Anything else
    raises ``ValueError`` naming what failed, because a solver fed a guessed
    geometry returns a confident, wrong answer.

    Args:
        component: A gdsfactory component, e.g. from ``gf.import_gds``.
        layer: The layer the ridges are drawn on.
        tolerance_um: Allowed deviation; defaults to one database unit.

    Returns:
        The :class:`LineGrating`, with the period from a least-squares fit to the
        ridge positions.
    """
    tolerance = component.kcl.dbu * 1.01 if tolerance_um is None else tolerance_um
    layer_index = gf.get_layer(layer)
    polygons = component.get_polygons_points(layers=[layer])
    points = polygons.get(layer_index, [])
    if len(points) < 2:
        raise ValueError(f"need at least two ridges on layer {layer}, found {len(points)}")

    rectangles = []
    for polygon in points:
        polygon = np.asarray(polygon, dtype=float)
        xs, ys = np.unique(np.round(polygon[:, 0], 9)), np.unique(np.round(polygon[:, 1], 9))
        if len(polygon) != 4 or len(xs) != 2 or len(ys) != 2:
            raise ValueError(
                f"polygon on layer {layer} is not an axis-aligned rectangle: {polygon}"
            )
        rectangles.append((xs[0], xs[1], ys[0], ys[1]))
    rectangles.sort()
    x0, x1, y0, y1 = (np.array(column) for column in zip(*rectangles, strict=True))

    if np.ptp(y0) > tolerance or np.ptp(y1) > tolerance:
        raise ValueError("ridges do not all span the same y range; not a line grating")
    widths = x1 - x0
    if np.ptp(widths) > tolerance:
        raise ValueError(f"ridge widths differ by {np.ptp(widths):.4g} µm; not a line grating")
    index = np.arange(len(x0))
    period, start = np.polyfit(index, x0, 1)
    if np.max(np.abs(x0 - (start + period * index))) > tolerance:
        raise ValueError("ridges are not equally spaced; not a periodic line grating")
    if np.mean(widths) >= period:
        raise ValueError("ridges touch or overlap; no gap to form a grating")
    return LineGrating(
        period_um=float(period),
        fill=float(np.mean(widths) / period),
        offset=float((start / period) % 1.0),
        n_periods=len(x0),
        length_um=float(np.mean(y1 - y0)),
    )


# ---------------------------------------------------------------------------
# Vertical stack and materials
# ---------------------------------------------------------------------------


def material_tensor(
    spec: MaterialSpec, wavelength_um: jax.Array, rotation_deg: Sequence = (0.0, 0.0, 0.0)
) -> jax.Array:
    """Lab-frame permittivity tensor ``[..., 3, 3]`` for a material specification.

    Args:
        spec: A built-in material name (see :func:`hyperbolic_optics.dispersion.materials`),
            a number (isotropic ε), a 3×3 tensor, or a callable ``wavelength_um → ε``.
        wavelength_um: Vacuum wavelength in µm; differentiable.
        rotation_deg: Crystal orientation as payload-style ``(rotationX, rotationY,
            rotationZ)`` in degrees; differentiable.
    """
    if isinstance(spec, str):
        tensor = dispersion.at_wavelength(spec, wavelength_um)
    elif callable(spec):
        tensor = jnp.asarray(spec(wavelength_um))
    else:
        tensor = jnp.asarray(spec, dtype=complex)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):  # an isotropic ε
        tensor = tensor[..., None, None] * jnp.eye(3)
    tensor = jnp.broadcast_to(tensor, jnp.shape(wavelength_um) + (3, 3))
    tensor = tensor.astype(complex)
    angles = [jnp.radians(jnp.asarray(a, dtype=float)) for a in rotation_deg]
    if all(isinstance(a, (int, float)) and a == 0 for a in rotation_deg):
        return tensor
    return bcm.rotate_tensor(tensor, bcm.rotation_matrix(*angles))


@dataclass(frozen=True)
class StackLayer:
    """One layer of the vertical stack, from the incident side down.

    Attributes:
        thickness_um: Thickness in µm (may be a JAX value for optimisation).
        material: The layer's material -- the ridges, if ``background`` is set.
        background: Material between the ridges. ``None`` makes the layer
            unpatterned (a uniform film of ``material``).
        rotation_deg: Crystal orientation of ``material`` (payload-style, degrees).
    """

    thickness_um: Any
    material: MaterialSpec
    background: MaterialSpec | None = None
    rotation_deg: Sequence = (0.0, 0.0, 0.0)


def grating_layers(
    grating: LineGrating,
    stack: Sequence[StackLayer],
    wavelength_um: jax.Array,
    edge_width: float = 0.0,
) -> list[rcwa.GratingLayer]:
    """:mod:`~hyperbolic_optics.rcwa` layers for a line grating etched into a stack.

    Every patterned layer shares the grating's geometry (a single etch). Fields
    of ``grating`` may be JAX values, so fill and period can be optimised.
    """
    layers = []
    for spec in stack:
        eps = material_tensor(spec.material, wavelength_um, spec.rotation_deg)
        if spec.background is None:
            layers.append(rcwa.uniform_layer(eps, spec.thickness_um))
        else:
            groove = material_tensor(spec.background, wavelength_um)
            layers.append(
                rcwa.lamellar_layer(
                    eps,
                    groove,
                    grating.fill,
                    spec.thickness_um,
                    offset=grating.offset,
                    edge_width=edge_width,
                )
            )
    return layers


def simulate_component(
    component: gf.Component,
    layer: gf.typings.LayerSpec,
    stack: Sequence[StackLayer],
    wavelength_um: float,
    incidence_rad: float,
    eps_incident: float,
    eps_exit: float,
    max_order: int,
    azimuth_rad: float = 0.0,
) -> tuple[LineGrating, rcwa.GratingResult]:
    """Extract the grating from a layout and solve it: layout in, efficiencies out."""
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
