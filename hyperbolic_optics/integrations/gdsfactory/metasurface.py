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
  turned if asked. By default a reference turns the whole cell; use
  ``rotate_materials=False`` for masks etched into a wafer with fixed axes.
- :func:`read_metasurface` reads a layout back into per-site cells, geometry
  from the polygons and materials from the cells' ``info``.
- :func:`cell_layers` gives a cell's :mod:`~hyperbolic_optics.rcwa2d` layers,
  and :func:`cell_sax_model` its zeroth-order four-port SAX model.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import gdsfactory as gf
import kfactory as kf
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack, LogicalLayer

from hyperbolic_optics import rcwa2d
from hyperbolic_optics.integrations.gdsfactory.lattice import crossed_grating_layers
from hyperbolic_optics.integrations.gdsfactory.materials import (
    MaterialSpec,
    Oriented,
    resolve_material,
)
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    _mesh_order,
    _warn_process_geometry,
)

__all__ = [
    "cell_from_pdk",
    "cell_layers",
    "cell_sax_model",
    "metasurface_component",
    "read_metasurface",
    "turned",
]


def cell_from_pdk(
    component: gf.Component,
    layer_stack: LayerStack,
    lattice_um,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    *,
    sidewall_slices: int = 0,
) -> VerticalStack:
    """Read one cell's centred lattice footprint through a PDK LayerStack.

    Each z interval uses its own mask polygons, including derived layers.
    Lower mesh_order wins; WAFER covers the footprint. Uncovered regions are
    ambient. The lowest interval must be uniform and becomes the substrate.
    A StackLayer can represent two materials per interval; more are refused.
    Crystal cuts come from ``materials``, as in :func:`vertical_stack`.
    ``sidewall_slices=N`` approximates each nonvertical level by N midpoint
    slices, respecting ``width_to_z`` and scalar ``bias`` (µm per edge).
    Positive angles narrow towards increasing z. Polygon offsets snap to dbu;
    refine N and the layout grid separately. The default warns and uses vertical
    walls. Generic ``z_to_bias`` and directional biases are refused when slicing.
    """
    dbu = component.kcl.dbu
    if not isinstance(sidewall_slices, int) or sidewall_slices < 0:
        raise ValueError("sidewall_slices must be a nonnegative integer.")
    a1, a2 = np.asarray(lattice_um, float)
    # one rounded corner plus whole lattice steps, so the footprint tiles the plane exactly
    step1, step2 = np.rint(a1 / dbu).astype(int), np.rint(a2 / dbu).astype(int)
    start = np.floor(_corners(a1, a2)[0] / dbu + 0.5).astype(int)
    footprint = kf.kdb.Region(
        kf.kdb.Polygon(
            [kf.kdb.Point(*map(int, start + s)) for s in (0 * step1, step1, step1 + step2, step2)]
        )
    )
    if footprint.area() == 0:
        raise ValueError("The lattice footprint must have nonzero area on the layout grid.")
    present = []
    for name, level in layer_stack.layers.items():
        if not level.thickness:
            continue
        wafer = isinstance(level.layer, LogicalLayer) and gf.get_layer(
            level.layer.layer
        ) == gf.get_layer("WAFER")
        # shapes crossing the footprint's edge wrap around, as RCWA periodises them
        periodic = None if wafer else _tiled(level.layer.get_shapes(component), a1, a2, dbu)
        region = footprint if wafer else periodic & footprint
        if not region.is_empty():
            low, high = sorted((level.zmin, level.zmin + level.thickness))
            if wafer:
                present.append((low, high, name, level, region))
                continue
            if not sidewall_slices:
                _warn_process_geometry(name, level)
                present.append((low, high, name, level, region))
                continue
            if level.z_to_bias is not None or isinstance(level.bias, (tuple, list)):
                raise ValueError(
                    f"PDK level {name!r}: slicing supports angle and scalar bias only."
                )
            if not np.isfinite(level.sidewall_angle) or abs(level.sidewall_angle) >= 90:
                raise ValueError(f"Invalid sidewall angle for level {name!r}.")
            edges = (
                np.linspace(low, high, sidewall_slices + 1) if level.sidewall_angle else [low, high]
            )
            reference_z = level.zmin + level.width_to_z * level.thickness
            altitude = abs(np.linalg.det(np.array([a1, a2]))) / max(
                np.linalg.norm(a1), np.linalg.norm(a2)
            )
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                offset = (level.bias or 0.0) + (reference_z - (lo + hi) / 2) * np.tan(
                    np.radians(level.sidewall_angle)
                )
                if abs(offset) >= altitude / 2:
                    raise ValueError(f"PDK level {name!r}: offset exceeds half the cell altitude.")
                sliced = periodic.sized(round(offset / dbu)) & footprint
                # Keep even an empty slice: its z interval still contains background.
                present.append((float(lo), float(hi), name, level, sliced))
    if not present:
        raise ValueError("No PDK material is present in the cell footprint.")
    edges = sorted({z for low, high, *_ in present for z in (low, high)})

    def spec(name):
        return ambient if name is None else resolve_material(name, materials)

    layers, substrate = [], None
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        active = sorted(
            (p for p in present if p[0] <= low and p[1] >= high), key=lambda p: _mesh_order(p[3])
        )
        occupied, regions, ties = kf.kdb.Region(), {}, {}
        for _, _, name, level, region in active:
            priority = _mesh_order(level)
            for other_material, other in ties.get(priority, []):
                if other_material != level.material and not (region & other).is_empty():
                    raise ValueError(f"PDK materials tie on mesh_order near level {name!r}.")
            ties.setdefault(priority, []).append((level.material, region))
            visible = region - occupied
            if not visible.is_empty():
                regions[level.material] = regions.get(level.material, kf.kdb.Region()) | visible
            occupied = occupied | region
        uncovered = footprint - occupied
        if not uncovered.is_empty():
            regions[None] = uncovered
        if substrate is None:
            if len(regions) != 1:
                raise ValueError("The lowest PDK interval must be a uniform substrate.")
            substrate = spec(next(iter(regions)))
            continue
        if len(regions) > 2:
            raise ValueError(f"More than two materials in the cell at z={low:g}–{high:g} µm.")
        label = f"z={low:g}–{high:g}"
        if len(regions) == 1:
            layers.append(StackLayer(high - low, spec(next(iter(regions))), name=label))
            continue
        background = None if None in regions else max(regions, key=lambda k: regions[k].area())
        material = next(k for k in regions if k != background)
        region = regions[material].merged()
        # Trapezoids represent holes without discarding their interiors. All
        # pieces have the same material, so their common edges cancel in RCWA.
        if any(p.holes() for p in region.each()):
            region = region.decompose_trapezoids_to_region()
        polygons = tuple(_vertices_um(region, dbu))
        layers.append(
            StackLayer(
                high - low,
                spec(material),
                background=spec(background),
                polygons=polygons,
                name=label,
            )
        )
    return VerticalStack(tuple(reversed(layers)), incident=ambient, exit=substrate)


def _corners(a1, a2) -> np.ndarray:
    """The cell's footprint ``[4, 2]``, centred on the origin."""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    return np.array([0.5 * (i * a1 + j * a2) for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _tiled(region, a1, a2, dbu) -> kf.kdb.Region:
    """``region`` and its eight nearest lattice images, unmerged."""
    out = kf.kdb.Region()
    step1, step2 = (np.rint(np.asarray(a) / dbu).astype(int) for a in (a1, a2))
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            dx, dy = i * step1 + j * step2
            out += region.transformed(kf.kdb.Trans(int(dx), int(dy)))
    return out


def _vertices_um(region, dbu) -> list[np.ndarray]:
    return [np.array([(p.x * dbu, p.y * dbu) for p in q.each_point_hull()]) for q in region.each()]


def _check_overlaps(cell: gf.Component, angle_deg: float, lattice_um) -> None:
    """Refuse shapes that, turned, overlap each other or their lattice images (RCWA adds them)."""
    dbu = cell.kcl.dbu
    a1, a2 = (np.asarray(a, float) for a in lattice_um)
    turn = kf.kdb.ICplxTrans(1, angle_deg, False, 0, 0)
    for level in cell.info["levels"]:
        if level["background"] is None:
            continue
        region = _region(cell, tuple(level["layer"])).transformed(turn)
        if _tiled(region, a1, a2, dbu).area() < 9 * sum(p.area() for p in region.each()):
            raise ValueError(
                f"level {level['name']!r} turned by {angle_deg:g}° overlaps itself or its periodic "
                "images: shrink the shapes or enlarge the lattice."
            )


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
    if np.shape(spec) == (3, 3):
        return {"tensor": np.stack([np.real(spec), np.imag(spec)], -1).tolist()}
    raise TypeError(f"material {spec!r} cannot be stored in a GDS: use a name, number or tensor")


def _spec_from_json(value, materials: Mapping[str, MaterialSpec] | None):
    if isinstance(value, dict) and "tensor" in value:
        parts = np.asarray(value["tensor"], float)
        return parts[..., 0] + 1j * parts[..., 1]
    if isinstance(value, dict):
        return Oriented(_spec_from_json(value["material"], materials), tuple(value["rotation_deg"]))
    if isinstance(value, list):
        return complex(value[0], value[1])
    if isinstance(value, str) and materials and value in materials:
        return materials[value]
    return value


def _payload(cell: VerticalStack, layers, lattice_um, rotate_materials=True) -> dict:
    """The cell as JSON-able data: what is drawn, and what ``info`` records."""
    if len(cell.layers) > len(layers):
        raise ValueError(f"a cell has {len(cell.layers)} levels but only {len(layers)} layers")
    resolved = [tuple(gf.get_layer_tuple(layer)) for layer in layers[: len(cell.layers)]]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each vertical level needs a distinct GDS layer for read-back.")
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
        "rotate_materials": rotate_materials,
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
    footprint = [tuple(v) for v in _corners(*data["lattice_um"])]
    for level in data["levels"]:
        polygons = [footprint] if level["polygons"] is None else level["polygons"]
        for polygon in polygons:
            component.add_polygon([tuple(v) for v in polygon], layer=tuple(level["layer"]))
    component.info["lattice_um"] = data["lattice_um"]
    component.info["incident"] = data["incident"]
    component.info["exit"] = data["exit"]
    component.info["levels"] = [
        {k: v for k, v in level.items() if k != "polygons"} for level in data["levels"]
    ]
    component.info["rotate_materials"] = data["rotate_materials"]
    return component


# references at any angle are valid GDS; foundries that want them flattened can flatten
# the result, at the cost of the cells' info that read_metasurface needs
@gf.cell(set_settings=False, set_name=False, check_instances="ignore")
def _placed(design: str) -> gf.Component:
    data = json.loads(design)
    component = gf.Component(f"metasurface_{hashlib.sha256(design.encode()).hexdigest()[:12]}")
    cells = [
        gf.kcl[json.loads(p)["component"]] if p.startswith('{"component"') else _cell_component(p)
        for p in data["cells"]
    ]
    for (x, y), index, angle in zip(
        data["sites"], data["indices"], data["angles_deg"], strict=True
    ):
        component.create_inst(cells[index], kf.kdb.DCplxTrans(1, angle, False, x, y))
    component.info["lattice_um"] = data["lattice_um"]
    component.info["n_sites"] = len(data["sites"])
    component.info["n_cells"] = len(cells)
    return component


def metasurface_component(
    sites_um,
    cells: Sequence[VerticalStack],
    layers: Sequence[gf.typings.LayerSpec],
    lattice_um: tuple[tuple[float, float], tuple[float, float]],
    rotations_rad=None,
    *,
    rotate_materials: bool = True,
) -> gf.Component:
    """A metasurface: the cell ``cells[k]`` at each site ``sites_um[k]``.

    Args:
        sites_um: Cell centres ``[K, 2]``.
        cells: One per site (the same object may repeat): a
            :class:`~.stack.VerticalStack`, levels from the top down and
            patterned ones with ``polygons`` about the cell's centre; or a
            ``gf.Component`` drawn on PDK layers about its centre, read back
            with ``read_metasurface(..., layer_stack=...)``.
        layers: The GDS layer of each level: level ``k`` of every stack cell is
            drawn on ``layers[k]``.
        lattice_um: ``(a1, a2)``: the lattice the cells were designed on; an
            unpatterned level is drawn as the cell's footprint.
        rotations_rad: A turn of each site's cell about its centre ``[K]``.
        rotate_materials: Turn the crystals with each reference (the default,
            e.g. placed flakes). False turns only the mask polygons, preserving
            the wafer axes and each layer's explicit crystal orientation.
            Stored in each cell's metadata for GDS read-back.

    Returns:
        The component, a reference per site to one cell per distinct design.
        The same design gives the same component (cached like any ``@gf.cell``).
    """
    sites = np.asarray(sites_um, dtype=float).reshape(-1, 2)
    if len(cells) != len(sites):
        raise ValueError(f"{len(sites)} sites but {len(cells)} cells")
    angles = np.zeros(len(sites)) if rotations_rad is None else np.degrees(rotations_rad)
    angles = np.round(np.broadcast_to(angles, len(sites)), 6)
    payloads, indices, seen = [], [], {}
    for cell in cells:
        if id(cell) not in seen:
            drawn = isinstance(cell, gf.Component)
            payload = json.dumps(
                _drawn_key(cell) if drawn else _payload(cell, layers, lattice_um, rotate_materials),
                sort_keys=True,
            )
            if payload not in payloads:
                payloads.append(payload)
                if not drawn:
                    _warn_off_grid(cell, gf.kcl.dbu)
            seen[id(cell)] = payloads.index(payload)
        indices.append(seen[id(cell)])
    for index, angle in sorted(set(zip(indices, angles.tolist(), strict=True))):
        if not payloads[index].startswith('{"component"'):  # PDK-drawn cells merge on read-back
            _check_overlaps(_cell_component(payloads[index]), angle, lattice_um)
    design = json.dumps({
        "cells": payloads, "indices": indices,
        "lattice_um": [[float(v) for v in a] for a in lattice_um],
        "sites": np.round(sites, 6).tolist(), "angles_deg": angles.tolist(),
    })  # fmt: skip
    return _placed(design)


def _drawn_key(cell: gf.Component) -> dict:
    """A drawn cell by index (names need not be unique), with its shapes' hash for caching."""
    shapes = {str(k): sorted(map(str, v)) for k, v in cell.get_polygons().items()}
    digest = hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()
    return {"component": cell.cell_index(), "shapes": digest}


def _warn_off_grid(cell: VerticalStack, dbu: float) -> None:
    for level in cell.layers:
        for polygon in level.polygons or ():
            steps = np.asarray(polygon, float) / dbu
            if np.abs(steps - np.rint(steps)).max() > 1e-6:
                warnings.warn(
                    f"level {level.name!r} has vertices off the {dbu * 1e3:g} nm layout grid; "
                    "they are snapped when drawn. Re-solve read_metasurface's cells for what "
                    "is drawn.",
                    UserWarning,
                    stacklevel=3,
                )
                return


def _region(cell, layer) -> kf.kdb.Region:
    """A cell's shapes on ``layer``, unmerged (empty if the layer is absent)."""
    index = cell.kcl.layout.find_layer(*layer)
    if index is None:
        return kf.kdb.Region()
    region = kf.kdb.Region(cell.kdb_cell.begin_shapes_rec(index))
    if any(p.holes() for p in region.each()):
        raise ValueError("Polygons with holes must be decomposed into simple polygons.")
    return region


def read_metasurface(
    component: gf.Component,
    materials: Mapping[str, MaterialSpec] | None = None,
    *,
    layer_stack: LayerStack | None = None,
    lattice_um=None,
    ambient: MaterialSpec = 1.0,
    sidewall_slices: int = 0,
) -> tuple[np.ndarray, list[VerticalStack], tuple[tuple[float, float], tuple[float, float]]]:
    """A metasurface layout read back into per-site cells.

    Returns ``(sites_um [K, 2], cells, lattice_um)``. Geometry always comes from
    the polygons, so what is re-simulated is what was drawn.

    Without ``layer_stack``, materials, thicknesses and orientations come from
    each cell's ``info`` (named materials looked up in ``materials``) and the
    reference's turn is applied with :func:`turned`, turning crystals unless
    the cell was drawn with ``rotate_materials=False``.

    With ``layer_stack``, each reference's cell, turned as placed, is read
    through the PDK by :func:`cell_from_pdk` (``materials``, ``ambient`` and
    ``sidewall_slices`` as there): crystals keep the wafer's axes. Cells need
    no ``info``; ``lattice_um`` is required unless the layout records it.

    Pass the metasurface's top cell. Nested layouts, arrays, mirrors and
    magnification are refused rather than reconstructed incompletely.
    """
    lattice = lattice_um if lattice_um is not None else component.info.get("lattice_um")
    lattice = None if lattice is None else tuple(tuple(float(v) for v in a) for a in lattice)
    sites, cells, solved = [], [], {}
    # a snapshot: drawing while iterating defers KLayout's hierarchy update
    for inst in list(component.insts):
        trans = inst.dcplx_trans
        if trans.is_mirror() or not np.isclose(trans.mag, 1.0, rtol=0, atol=1e-12):
            raise ValueError("Mirrored or magnified metasurface references are unsupported.")
        if inst.na > 1 or inst.nb > 1:
            raise ValueError("Expand array references into individual metasurface sites.")
        info = inst.cell.info
        if layer_stack is None and "levels" not in info:
            raise ValueError(
                "Expected direct cell references with metasurface metadata; "
                "pass layer_stack= for cells drawn on PDK layers."
            )
        if "lattice_um" in info:
            cell_lattice = tuple(tuple(float(v) for v in a) for a in info["lattice_um"])
            if lattice is not None and lattice != cell_lattice:
                raise ValueError("All metasurface cells must share the same design lattice.")
            lattice = cell_lattice
        if lattice is None:
            raise ValueError("No design lattice recorded in the layout; pass lattice_um.")
        key = (inst.cell.name, trans.angle)
        if key not in solved:
            if layer_stack is None:
                solved[key] = _cell_from_info(inst.cell, trans.angle, materials)
            else:
                placed = gf.Component()
                placed.create_inst(inst.cell, kf.kdb.DCplxTrans(1, trans.angle, False, 0, 0))
                try:
                    solved[key] = cell_from_pdk(
                        placed, layer_stack, lattice, materials, ambient,
                        sidewall_slices=sidewall_slices,
                    )  # fmt: skip
                finally:
                    placed.delete()
        cells.append(solved[key])
        sites.append((trans.disp.x, trans.disp.y))
    if not cells:
        raise ValueError("No metasurface cell references found; pass an unflattened layout.")
    return np.asarray(sites), cells, lattice


def _cell_from_info(cell, angle_deg, materials) -> VerticalStack:
    """A cell drawn by :func:`metasurface_component`, from its ``info``, turned."""
    info = cell.info
    _check_overlaps(cell, angle_deg, info["lattice_um"])
    level_layers = [tuple(level["layer"]) for level in info["levels"]]
    if len(set(level_layers)) != len(level_layers):
        raise ValueError("Each vertical level needs a distinct GDS layer for read-back.")
    levels = tuple(
        StackLayer(
            thickness_um=level["thickness_um"],
            material=_spec_from_json(level["material"], materials),
            background=_spec_from_json(level["background"], materials),
            rotation_deg=tuple(level["rotation_deg"]), name=level["name"],
            polygons=None if level["background"] is None
            else tuple(_vertices_um(_region(cell, tuple(level["layer"])), cell.kcl.dbu)),
        )
        for level in info["levels"]
    )  # fmt: skip
    stack = VerticalStack(levels, _spec_from_json(info["incident"], materials),
                          _spec_from_json(info["exit"], materials))  # fmt: skip
    return turned(
        stack, np.radians(float(angle_deg)), rotate_materials=info.get("rotate_materials", True)
    )


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
    subwavelength array (higher orders evanescent). ``azimuth`` rotates the
    sample, as in ``stack_sax_model``; the ports retain a common p/s basis.
    """
    import jax.numpy as jnp

    from hyperbolic_optics.integrations import sax as hsax

    lattice = rcwa2d.Lattice(*(tuple(a) for a in lattice_um))

    def model(wl=1.55, kx=0.0, azimuth=0.0):
        eps_top = cell.half_space_permittivity("incident", wl)
        eps_bottom = cell.half_space_permittivity("exit", wl)
        smatrix = rcwa2d.crossed_grating_smatrix(
            cell_layers(cell, jnp.asarray(wl)), lattice, jnp.asarray(wl), 0.0,
            eps_top, eps_bottom, max_order, azimuth_rad=-jnp.asarray(azimuth), kx=kx, **kwargs,
        )  # fmt: skip
        return hsax.sdict(smatrix)

    return model


def turned(cell: VerticalStack, angle_rad, *, rotate_materials: bool = True) -> VerticalStack:
    """Turn a cell's polygons, optionally turning its crystals too (traceable).

    The default rotates the whole cell, suitable for a placed crystal flake.
    ``rotate_materials=False`` models a mask turned on a fixed wafer: material
    and background tensors retain their cuts and per-layer orientations.
    Half-spaces and the periodic lattice are unchanged in either case.
    """
    import jax.numpy as jnp

    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    turn = jnp.array([[c, -s], [s, c]])
    return replace(cell, layers=tuple(
        replace(level,
                polygons=None if level.polygons is None
                else tuple(jnp.asarray(p) @ turn.T for p in level.polygons),
                rotation_deg=(level.rotation_deg[0], level.rotation_deg[1],
                              level.rotation_deg[2] + jnp.degrees(angle_rad))
                if rotate_materials else level.rotation_deg)
        for level in cell.layers
    ))  # fmt: skip
