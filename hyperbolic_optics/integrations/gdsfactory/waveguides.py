"""Waveguide cross-sections from a layout and the PDK's LayerStack, and their modes.

At a point on a waveguide, the PDK's material column is read at positions
across it (:func:`~hyperbolic_optics.integrations.gdsfactory.stack.vertical_profile`),
giving ``ε(lateral, vertical)``, whose full-vector modes
:func:`hyperbolic_optics.modes2d.solve_modes` finds -- anisotropic cores (an
x-cut LiNbO₃ rib) included, with each material's tensor turned into the
waveguide's frame.
"""

from __future__ import annotations

from collections.abc import Mapping

import gdsfactory as gf
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack

from hyperbolic_optics import modes2d
from hyperbolic_optics.integrations.gdsfactory.materials import (
    MaterialSpec,
    material_tensor,
    resolve_material,
)
from hyperbolic_optics.integrations.gdsfactory.stack import vertical_profile

__all__ = ["waveguide_cross_section", "waveguide_modes"]


def waveguide_cross_section(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    wavelength_um: float,
    direction: str = "x",
    lateral_um: tuple[float, float] = (-2.0, 2.0),
    vertical_um: tuple[float, float] | None = None,
    step_um: float = 0.02,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    subsamples: int = 4,
) -> modes2d.CrossSection:
    """The cross-section of a waveguide at a point of a layout.

    Args:
        component: The layout.
        layer_stack: The PDK's ``LayerStack``.
        point_um: A point on the waveguide's axis.
        wavelength_um: For the materials' dispersion.
        direction: The waveguide's direction in the layout, ``"x"`` or ``"y"``.
        lateral_um: The window across the waveguide, relative to the point.
        vertical_um: The vertical window (layout z); by default the stack's
            span with 1 µm of the ambient above.
        step_um: The grid step.
        materials, ambient: As in :func:`~hyperbolic_optics.integrations.gdsfactory.vertical_stack`.
        subsamples: Samples per cell side for the permittivity average.

    Returns:
        A :class:`hyperbolic_optics.modes2d.CrossSection` in the waveguide's
        frame: x across it (the layout's y for a waveguide along x), y vertical.
    """
    x0, y0 = point_um
    lateral = np.arange(lateral_um[0], lateral_um[1] + step_um / 2, step_um)
    # the columns, read exactly where the permittivity is sampled: every subsample
    # of every Yee position lies on lateral[0] + (m + ½)·step/subsamples
    pitch = step_um / subsamples
    count = int(np.ceil((lateral[-1] - lateral[0] + 2 * step_um) / pitch)) + 1
    fine = lateral[0] - step_um + (np.arange(count) + 0.5) * pitch
    columns = []
    for offset in fine:
        point = (x0, y0 + offset) if direction == "x" else (x0 + offset, y0)
        columns.append(vertical_profile(component, layer_stack, point))
    bottom = min(interval.zmin for column in columns for interval in column)
    top = max(interval.zmax for column in columns for interval in column)
    if vertical_um is None:
        vertical_um = (bottom, top + 1.0)
    vertical = np.arange(vertical_um[0], vertical_um[1] + step_um / 2, step_um)

    # each material's diagonal in the waveguide's frame (across, vertical, along)
    order = [1, 2, 0] if direction == "x" else [0, 2, 1]
    cache: dict = {}

    def diagonal(name):
        if name not in cache:
            spec = ambient if name is None else resolve_material(name, materials)
            tensor = np.asarray(material_tensor(spec, wavelength_um))[np.ix_(order, order)]
            off = tensor - np.diag(np.diag(tensor))
            if np.abs(off).max() > 1e-9 * np.abs(tensor).max():
                raise ValueError(
                    f"{name!r} is not diagonal in the waveguide's frame (across, vertical, "
                    "along); modes2d needs the crystal axes along the waveguide or across it"
                )
            cache[name] = np.diag(tensor)
        return cache[name]

    below = columns[0][0].material  # the lowest level extends down

    def eps_at(px, py, component_index):
        px, py = np.broadcast_arrays(px, py)
        out = np.empty(px.shape, dtype=complex)
        index = np.clip(np.round((px - fine[0]) / pitch).astype(int), 0, len(fine) - 1)
        for i in np.unique(index):
            where = index == i
            z = py[where]
            values = np.full(z.shape, diagonal(None)[component_index])
            for interval in columns[i]:
                inside = (z >= interval.zmin) & (z < interval.zmax)
                values[inside] = diagonal(interval.material)[component_index]
            values[z < bottom] = diagonal(below)[component_index]
            out[where] = values
        return out

    return modes2d.sampled(lateral, vertical, eps_at, subsamples)


def waveguide_modes(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    wavelength_um: float,
    n_modes: int = 2,
    target_index: float | None = None,
    **kwargs,
) -> tuple[modes2d.CrossSection, list[modes2d.Mode]]:
    """The full-vector modes of a waveguide at a point of a layout.

    ``**kwargs`` go to :func:`waveguide_cross_section`. Returns the
    cross-section and the modes, highest effective index first.
    """
    section = waveguide_cross_section(component, layer_stack, point_um, wavelength_um, **kwargs)
    return section, modes2d.solve_modes(section, wavelength_um, n_modes, target_index)
