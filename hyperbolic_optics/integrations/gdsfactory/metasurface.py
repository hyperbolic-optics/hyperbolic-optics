"""Metasurface layouts: a library of unit cells placed site by site.

:func:`metasurface_component` draws a metasurface designed with
:mod:`hyperbolic_optics.metasurfaces`: each library entry is one cell (its
polygons, in cell coordinates centred on the origin), and each site an
instance of the chosen entry, translated and optionally turned (for
geometric-phase designs). A lens of ten thousand pillars is then as many
references to a few dozen cells, not ten thousand polygons.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import gdsfactory as gf
import kfactory as kf
import numpy as np

from hyperbolic_optics import rcwa2d

__all__ = ["disc_polygons", "metasurface_component"]


def disc_polygons(radii_um: Sequence[float], n_vertices: int = 48):
    """One disc per library entry, centred on the origin: ``[[vertices [V, 2]]]``."""
    return [[np.asarray(rcwa2d.ellipse((0.0, 0.0), (r, r), n_vertices))] for r in radii_um]


def metasurface_component(
    sites_um: np.ndarray,
    indices: np.ndarray,
    library_polygons: Sequence[Sequence[np.ndarray]],
    layer: gf.typings.LayerSpec = (1, 0),
    rotations_rad: np.ndarray | None = None,
    name: str | None = None,
) -> gf.Component:
    """A metasurface: the library cell ``indices[k]`` at each site ``sites_um[k]``.

    Args:
        sites_um: Cell centres ``[K, 2]`` (e.g.
            :func:`hyperbolic_optics.metasurfaces.square_sites`).
        indices: The library entry at each site ``[K]`` (e.g.
            :func:`hyperbolic_optics.metasurfaces.choose_cells`).
        library_polygons: For each entry, its polygons (vertices ``[V, 2]``,
            centred on the origin), e.g. :func:`disc_polygons`.
        layer: The layer to draw on.
        rotations_rad: A turn of each cell about its centre ``[K]``
            (geometric phase); none if omitted.
        name: The component's name; by default a hash of the design. A
            name already in the layout returns that cell, as ``@gf.cell`` does.

    Only the entries used are drawn; ``component.info`` records the number of
    sites and the entries used.
    """
    sites = np.asarray(sites_um, dtype=float).reshape(-1, 2)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if len(indices) != len(sites):
        raise ValueError(f"{len(sites)} sites but {len(indices)} indices")
    rotations = None if rotations_rad is None else np.asarray(rotations_rad, float).reshape(-1)

    if name is None:
        digest = hashlib.sha256()
        for array in (sites, indices, *(np.asarray(p) for c in library_polygons for p in c)):
            digest.update(np.ascontiguousarray(array).tobytes())
        if rotations is not None:
            digest.update(rotations.tobytes())
        digest.update(repr(layer).encode())
        name = f"metasurface_{digest.hexdigest()[:12]}"

    kcl = gf.kcl
    if kcl.layout.cell(name) is not None:  # built before: the layout's own copy
        return gf.Component(base=kcl[name].base)
    component = gf.Component(name)
    used = np.unique(indices)
    cells = {}
    for entry in used:
        cell = gf.Component(f"{name}_cell{entry}")
        for polygon in library_polygons[entry]:
            cell.add_polygon([tuple(map(float, v)) for v in np.asarray(polygon)], layer=layer)
        cells[entry] = cell
    for k, (x, y) in enumerate(sites):
        angle = 0.0 if rotations is None else float(np.degrees(rotations[k]))
        component.create_inst(cells[indices[k]], kf.kdb.DCplxTrans(1, angle, False, x, y))
    component.info["n_sites"] = len(sites)
    component.info["entries_used"] = tuple(int(e) for e in used)
    return component
