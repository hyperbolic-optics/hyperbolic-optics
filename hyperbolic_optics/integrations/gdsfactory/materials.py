"""Materials for gdsfactory stacks: specifications, crystal orientation, PDK names.

A material specification is any of:

- a built-in material name (``"LiNbO3"``, ``"hBN"``, ``"SiO2"``, … -- see
  :func:`hyperbolic_optics.dispersion.materials`), evaluated with its dispersion;
- a number (isotropic ε) or a 3×3 tensor;
- a callable ``wavelength_um → ε`` (scalar or tensor);
- :class:`Oriented`, which pairs any of the above with a crystal orientation.

Anisotropy is the reason this package exists, so orientation is explicit: a PDK
names a material (``"ln"``), but only the designer knows the cut. Map it with
:func:`x_cut` or :func:`z_cut`, e.g. ``{"ln": Oriented("LiNbO3", x_cut(90))}``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import bcm, dispersion

__all__ = [
    "DEFAULT_MATERIALS",
    "MaterialSpec",
    "Oriented",
    "material_tensor",
    "resolve_material",
    "x_cut",
    "z_cut",
]

MaterialSpec = str | complex | float | np.ndarray | jax.Array | Callable[[Any], Any]


@dataclass(frozen=True)
class Oriented:
    """A material and its crystal orientation.

    Attributes:
        material: Any material specification.
        rotation_deg: ``(rotationX, rotationY, rotationZ)`` in degrees, the
            payload API's convention: a uniaxial crystal's optic axis (crystal
            z) points at polar angle ``rotationY`` from the stack normal and
            azimuth ``rotationZ`` from the layout x axis.
    """

    material: MaterialSpec
    rotation_deg: Sequence = (0.0, 0.0, 0.0)


def z_cut() -> tuple[float, float, float]:
    """Optic axis along the wafer normal (a z-cut wafer of a uniaxial crystal)."""
    return (0.0, 0.0, 0.0)


def x_cut(axis_azimuth_deg: float = 0.0) -> tuple[float, float, float]:
    """Optic axis in the wafer plane at ``axis_azimuth_deg`` from the layout x axis.

    An x- or y-cut wafer of a uniaxial crystal such as LiNbO₃: which of the two
    only matters for biaxial crystals. For a waveguide along layout y, TE light
    (E along x) sees the extraordinary index when ``axis_azimuth_deg = 0``.
    """
    return (0.0, 90.0, float(axis_azimuth_deg))


#: How the material names used by gdsfactory's generic PDK map to this package's
#: dispersive models. Anisotropic PDK materials are deliberately absent: their
#: orientation depends on the wafer, so map them yourself with :class:`Oriented`.
DEFAULT_MATERIALS: dict[str, MaterialSpec] = {
    "si": "Si",
    "sio2": "SiO2",
    "sin": "Si3N4",
    "si3n4": "Si3N4",
    "air": 1.0,
}


def resolve_material(
    name: str, overrides: Mapping[str, MaterialSpec] | None = None
) -> MaterialSpec:
    """The specification for a PDK material name: ``overrides`` first, then defaults."""
    table = {**DEFAULT_MATERIALS, **(overrides or {})}
    if name in table:
        return table[name]
    if name.lower() in table:
        return table[name.lower()]
    raise KeyError(
        f"No material for PDK material {name!r}. Pass materials={{{name!r}: ...}}, e.g. a "
        f"built-in name from hyperbolic_optics.dispersion.materials(), a permittivity, or "
        f"Oriented('LiNbO3', x_cut(0)) for an anisotropic crystal."
    )


def _check_range(name: str, wavelength_um: Any) -> None:
    """Warn when a named material is used outside its fit (host values only)."""
    if isinstance(wavelength_um, jax.core.Tracer):
        return
    low, high = dispersion.frequency_range(name)
    wavenumber = 1e4 / np.asarray(wavelength_um, dtype=float)
    if wavenumber.min() < low * (1 - 1e-9) or wavenumber.max() > high * (1 + 1e-9):
        warnings.warn(
            f"{name} is evaluated at {1e4 / wavenumber.max():.3g}-{1e4 / wavenumber.min():.3g} µm, "
            f"outside the {1e4 / high:.3g}-{1e4 / low:.3g} µm its model was fitted over; "
            "the extrapolated permittivity may be unphysical.",
            UserWarning,
            stacklevel=4,
        )


def material_tensor(
    spec: MaterialSpec | Oriented,
    wavelength_um: jax.Array,
    rotation_deg: Sequence = (0.0, 0.0, 0.0),
) -> jax.Array:
    """Lab-frame permittivity tensor ``[..., 3, 3]`` for a material specification.

    Args:
        spec: A material specification or :class:`Oriented`. An oriented
            crystal is cut first, then turned by ``rotation_deg``: a layer's
            twist turns the crystal as cut.
        wavelength_um: Vacuum wavelength in µm; differentiable.
        rotation_deg: Payload-style ``(rotationX, rotationY, rotationZ)``, degrees.
    """
    if isinstance(spec, Oriented):
        base = material_tensor(spec.material, wavelength_um)
        rotated = _rotate(base, spec.rotation_deg)
        return _rotate(rotated, rotation_deg)
    if isinstance(spec, str):
        _check_range(spec, wavelength_um)
        tensor = dispersion.at_wavelength(spec, wavelength_um)
    elif callable(spec):
        tensor = jnp.asarray(spec(wavelength_um))
    else:
        tensor = jnp.asarray(spec, dtype=complex)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):  # an isotropic ε
        tensor = tensor[..., None, None] * jnp.eye(3)
    tensor = jnp.broadcast_to(tensor, jnp.shape(wavelength_um) + (3, 3)).astype(complex)
    return _rotate(tensor, rotation_deg)


def _rotate(tensor: jax.Array, rotation_deg: Sequence) -> jax.Array:
    if all(isinstance(a, (int, float)) and a == 0 for a in rotation_deg):
        return tensor
    angles = [jnp.radians(jnp.asarray(a, dtype=float)) for a in rotation_deg]
    return bcm.rotate_tensor(tensor, bcm.rotation_matrix(*angles))
