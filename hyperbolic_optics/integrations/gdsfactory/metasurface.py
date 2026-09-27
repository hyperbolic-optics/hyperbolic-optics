"""Metasurfaces whose unit cells differ site by site: layout, read-back and SAX.

A cell is a :class:`~.stack.VerticalStack` whose layers carry their own
polygons (:attr:`~.stack.StackLayer.polygons`, in cell coordinates about the
origin): any number of layers, each with its own material, crystal orientation
(``rotation_deg`` or :class:`~.materials.Oriented`), thickness and shapes.
Twisted stacks, crystals at any orientation and cells with different numbers
of layers are all the same thing.

- :func:`metasurface_component` draws each distinct cell once (level ``k`` on
  ``layers[k]``, an unpatterned level as the cell's footprint) with its levels
  recorded in the cell's ``info``, and places a reference to it at every site,
  turned if asked. A turned reference turns the whole cell, crystal included.
- :func:`read_metasurface` reads a layout back into per-site cells, geometry
  from the polygons and materials from the cells' ``info``.
- :func:`cell_layers` gives a cell's :mod:`~hyperbolic_optics.rcwa2d` layers,
  and :func:`cell_sax_model` its zeroth-order four-port SAX model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import gdsfactory as gf
import kfactory as kf
import numpy as np

from hyperbolic_optics import rcwa2d
from hyperbolic_optics.integrations.gdsfactory.lattice import crossed_grating_layers
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec, Oriented
from hyperbolic_optics.integrations.gdsfactory.stack import StackLayer, VerticalStack

__all__ = ["cell_layers", "cell_sax_model", "metasurface_component", "read_metasurface", "turned"]


def _spec_to_json(spec):
    if spec is None or isinstance(spec, str):
        return spec
    if callable(spec):
        raise TypeError(
            f"material {spec!r} cannot be stored in a GDS; give it a name and pass "
            "materials={name: spec} to read_metasurface"
        )
    if isinstance(spec, Oriented):
        return {"material": _spec_to_json(spec.material),
                "rotation_deg": [float(a) for a in spec.rotation_deg]}  # fmt: skip
    if np.ndim(spec) == 0 and np.isrealobj(spec):
        return float(spec)
    if np.ndim(spec) == 0:
        return [float(np.real(spec)), float(np.imag(spec))]  # complex: GDS metadata is real
    raise TypeError(f"material {spec!r} cannot be stored in a GDS: use a name or a number")


def _spec_from_json(value, materials: Mapping[str, MaterialSpec] | None):
    if isinstance(value, dict):
        return Oriented(_spec_from_json(value["material"], materials), tuple(value["rotation_deg"]))
    if isinstance(value, list):
        return complex(value[0], value[1])
    if isinstance(value, str) and materials and value in materials:
        return materials[value]
    return value


def _payload(cell: VerticalStack, layers, lattice_um) -> dict:
    """The cell as JSON-able data: what is drawn, and what ``info`` records."""
    if len(cell.layers) > len(layers):
        raise ValueError(f"a cell has {len(cell.layers)} levels but only {len(layers)} layers")
    levels = []
    for level, gds_layer in zip(cell.layers, layers, strict=False):
        if (level.background is None) != (level.polygons is None):
            raise ValueError(
                f"level {level.name!r}: a patterned level needs both polygons and a background"
            )
        levels.append({
            "thickness_um": float(level.thickness_um),
            "material": _spec_to_json(level.material),
            "background": _spec_to_json(level.background),
            "rotation_deg": [float(a) for a in level.rotation_deg],
            "name": level.name,
            "layer": [int(v) for v in gf.get_layer_tuple(gds_layer)],
            "polygons": None if level.polygons is None
            else [np.round(np.asarray(p, float), 6).tolist() for p in level.polygons],
        })  # fmt: skip
    return {
        "lattice_um": [[float(v) for v in a] for a in lattice_um],
        "incident": _spec_to_json(cell.incident),
        "exit": _spec_to_json(cell.exit),
        "levels": levels,
    }


@gf.cell(set_settings=False, set_name=False)
def _cell_component(payload: str) -> gf.Component:
    data = json.loads(payload)
    component = gf.Component(
        f"metasurface_cell_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"
    )
    a1, a2 = (np.asarray(a) for a in data["lattice_um"])
    footprint = [
        tuple(0.5 * (s1 * a1 + s2 * a2)) for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    for level in data["levels"]:
        for polygon in level["polygons"] or [footprint]:
            component.add_polygon([tuple(v) for v in polygon], layer=tuple(level["layer"]))
    component.info["lattice_um"] = data["lattice_um"]
    component.info["incident"] = data["incident"]
    component.info["exit"] = data["exit"]
    component.info["levels"] = [
        {k: v for k, v in level.items() if k != "polygons"} for level in data["levels"]
    ]
    return component


# references at any angle are valid GDS; foundries that want them flattened can flatten
# the result, at the cost of the cells' info that read_metasurface needs
@gf.cell(set_settings=False, set_name=False, check_instances="ignore")
def _placed(design: str) -> gf.Component:
    data = json.loads(design)
    component = gf.Component(f"metasurface_{hashlib.sha256(design.encode()).hexdigest()[:12]}")
    cells = [_cell_component(payload) for payload in data["cells"]]
    for (x, y), index, angle in zip(
        data["sites"], data["indices"], data["angles_deg"], strict=True
    ):
        component.create_inst(cells[index], kf.kdb.DCplxTrans(1, angle, False, x, y))
    component.info["n_sites"] = len(data["sites"])
    component.info["n_cells"] = len(cells)
    return component


def metasurface_component(
    sites_um,
    cells: Sequence[VerticalStack],
    layers: Sequence[gf.typings.LayerSpec],
    lattice_um: tuple[tuple[float, float], tuple[float, float]],
    rotations_rad=None,
) -> gf.Component:
    """A metasurface: the cell ``cells[k]`` at each site ``sites_um[k]``.

    Args:
        sites_um: Cell centres ``[K, 2]``.
        cells: One :class:`~.stack.VerticalStack` per site (the same object may
            repeat); levels from the top down, patterned ones with
            ``polygons`` about the cell's centre.
        layers: The GDS layer of each level: level ``k`` of every cell is drawn
            on ``layers[k]``.
        lattice_um: ``(a1, a2)``: the lattice the cells were designed on; an
            unpatterned level is drawn as the cell's footprint.
        rotations_rad: A turn of each site's cell about its centre ``[K]``.

    Returns:
        The component, a reference per site to one cell per distinct design.
        The same design gives the same component (cached like any ``@gf.cell``).
    """
    sites = np.asarray(sites_um, dtype=float).reshape(-1, 2)
    if len(cells) != len(sites):
        raise ValueError(f"{len(sites)} sites but {len(cells)} cells")
    angles = np.zeros(len(sites)) if rotations_rad is None else np.degrees(rotations_rad)
    payloads, indices, seen = [], [], {}
    for cell in cells:
        if id(cell) not in seen:
            payload = json.dumps(_payload(cell, layers, lattice_um), sort_keys=True)
            if payload not in payloads:
                payloads.append(payload)
            seen[id(cell)] = payloads.index(payload)
        indices.append(seen[id(cell)])
    design = json.dumps({
        "cells": payloads, "indices": indices,
        "sites": np.round(sites, 6).tolist(), "angles_deg": np.round(angles, 6).tolist(),
    })  # fmt: skip
    return _placed(design)


def _polygons(cell, layer) -> list[np.ndarray]:
    """A cell's polygons on ``layer``, in µm (KLayout's own region API)."""
    dbu = cell.kcl.dbu
    region = kf.kdb.Region(cell.kdb_cell.begin_shapes_rec(gf.get_layer(layer)))
    return [
        np.array([(p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()])
        for polygon in region.each()
    ]


def read_metasurface(
    component: gf.Component, materials: Mapping[str, MaterialSpec] | None = None
) -> tuple[np.ndarray, list[VerticalStack], tuple[tuple[float, float], tuple[float, float]]]:
    """A metasurface layout read back into per-site cells.

    Returns ``(sites_um [K, 2], cells, lattice_um)``. Each site's cell has its
    reference's turn applied: its polygons turned and ``rotation_deg`` about z
    increased by the same angle (exact: the orientation is ``Rz·Ry·Rx``).
    Geometry comes from the polygons, so what is re-simulated is what was
    drawn; materials, thicknesses and orientations from each cell's ``info``,
    with named materials looked up in ``materials``.
    """
    sites, cells, lattice = [], [], None
    for inst in component.insts:
        info = inst.cell.info
        if "levels" not in info:
            continue
        trans = inst.dcplx_trans
        levels = tuple(
            StackLayer(
                thickness_um=level["thickness_um"],
                material=_spec_from_json(level["material"], materials),
                background=_spec_from_json(level["background"], materials),
                rotation_deg=tuple(level["rotation_deg"]), name=level["name"],
                polygons=None if level["background"] is None
                else tuple(_polygons(inst.cell, tuple(level["layer"]))),
            )
            for level in info["levels"]
        )  # fmt: skip
        cell = VerticalStack(levels, _spec_from_json(info["incident"], materials),
                             _spec_from_json(info["exit"], materials))  # fmt: skip
        cells.append(turned(cell, np.radians(float(trans.angle))))
        sites.append((trans.disp.x, trans.disp.y))
        lattice = tuple(tuple(a) for a in info["lattice_um"])
    return np.asarray(sites), cells, lattice


def cell_layers(cell: VerticalStack, wavelength_um: Any) -> list[rcwa2d.PatternedLayer]:
    """A cell's :mod:`~hyperbolic_optics.rcwa2d` layers at a wavelength (traceable)."""
    return crossed_grating_layers(None, cell, wavelength_um)


def cell_sax_model(
    cell: VerticalStack,
    lattice_um,
    max_order: tuple[int, int] = (4, 4),
    **kwargs,
):
    """A SAX model of one cell in an infinite array of itself: its zeroth-order S-matrix.

    Returns ``model(wl, kx=0, azimuth=0) -> SDict`` over ``top@p``, ``top@s``,
    ``bottom@p``, ``bottom@s``, from
    :func:`hyperbolic_optics.rcwa2d.crossed_grating_smatrix`, with every
    material evaluated at SAX's ``wl``; ``kx`` is ``n·sin θ`` in the incident
    medium. Only the zeroth order is a port, so the model describes a
    subwavelength array (higher orders evanescent).
    """
    import jax.numpy as jnp

    from hyperbolic_optics.integrations import sax as hsax

    lattice = rcwa2d.Lattice(*(tuple(a) for a in lattice_um))

    def model(wl=1.55, kx=0.0, azimuth=0.0):
        eps_top = cell.half_space_permittivity("incident", wl)
        eps_bottom = cell.half_space_permittivity("exit", wl)
        angle = jnp.arcsin(jnp.asarray(kx) / jnp.sqrt(eps_top))
        smatrix = rcwa2d.crossed_grating_smatrix(
            cell_layers(cell, jnp.asarray(wl)), lattice, jnp.asarray(wl), angle,
            eps_top, eps_bottom, max_order, azimuth_rad=azimuth, **kwargs,
        )  # fmt: skip
        return hsax.sdict(smatrix)

    return model


def turned(cell: VerticalStack, angle_rad) -> VerticalStack:
    """``cell`` turned about its centre, polygons and crystals together (traceable)."""
    import jax.numpy as jnp

    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    turn = jnp.array([[c, -s], [s, c]])
    return replace(cell, layers=tuple(
        replace(level,
                polygons=None if level.polygons is None
                else tuple(jnp.asarray(p) @ turn.T for p in level.polygons),
                rotation_deg=(level.rotation_deg[0], level.rotation_deg[1],
                              level.rotation_deg[2] + jnp.degrees(angle_rad)))
        for level in cell.layers
    ))  # fmt: skip
