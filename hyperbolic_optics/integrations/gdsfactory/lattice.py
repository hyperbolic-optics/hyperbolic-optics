"""2D-periodic layouts between gdsfactory and the crossed-grating solver.

- :func:`periodic_array`, a PCell that repeats a unit cell of polygons on a
  rectangular lattice, with the period and every vertex snapped to the layout
  grid, so the exported geometry is exactly periodic;
- :func:`extract_unit_cell`, which reads a 2D-periodic pattern back from any
  gdsfactory component or imported GDS, and refuses anything that is not one;
- :func:`crossed_grating_stack`, the PDK stack that pattern's etch makes, by
  comparing the ``LayerStack`` column inside the shapes with the column around
  them, as :func:`~hyperbolic_optics.integrations.gdsfactory.stack.grating_stack`
  does for line gratings;
- :func:`crossed_grating_layers`, the :mod:`hyperbolic_optics.rcwa2d` layers
  for a unit cell etched into a stack. Its polygons may be JAX values, so a
  shape can be optimised and then drawn with :func:`periodic_array`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import kfactory as kf
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack

from hyperbolic_optics import rcwa2d
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec, material_tensor
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    patterned_stack,
    vertical_profile,
)

__all__ = [
    "UnitCell",
    "crossed_grating_layers",
    "crossed_grating_stack",
    "extract_unit_cell",
    "periodic_array",
]


@dataclass(frozen=True)
class UnitCell:
    """The repeating unit of a 2D-periodic pattern on a rectangular lattice.

    Attributes:
        period_um: ``(Λx, Λy)`` in µm.
        polygons: The shapes of one cell, each ``[V, 2]`` in µm relative to
            the cell origin. They may extend past the cell; the solver
            periodises them. May be JAX values, for optimisation.
        n_cells: ``(nx, ny)`` cells drawn in the layout.
        origin_um: Layout position of the first cell's origin, so the stack
            beneath it can be sampled.
    """

    period_um: tuple[Any, Any]
    polygons: tuple[Any, ...]
    n_cells: tuple[int, int] = (1, 1)
    origin_um: tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------------
# PCell
# ---------------------------------------------------------------------------


def _as_vertices(polygon) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in np.asarray(polygon, dtype=float))


@gf.cell
def periodic_array(
    period_um: tuple[float, float] = (1.0, 1.0),
    polygons: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
    ),
    n_cells: tuple[int, int] = (10, 10),
    layer: gf.typings.LayerSpec = (1, 0),
) -> gf.Component:
    """PCell: the ``polygons`` of one cell, repeated ``n_cells`` times at ``period_um``.

    The period and every vertex are snapped to the layout grid (the database
    unit), so every cell drawn is an exact translate of the first. The snapped
    values are in ``component.info`` and are what :func:`extract_unit_cell`
    recovers. Pass polygon vertices as nested tuples (a PCell's arguments must
    be hashable), e.g. ``tuple(map(tuple, rcwa2d.ellipse(...).tolist()))``.
    """
    component = gf.Component()
    dbu = component.kcl.dbu
    px, py = (round(p / dbu) for p in period_um)
    if px <= 0 or py <= 0:
        raise ValueError(f"period {period_um} rounds to zero on this grid")
    snapped = [np.round(np.asarray(polygon, dtype=float) / dbu).astype(int) for polygon in polygons]
    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            for vertices in snapped:
                shifted = (vertices + np.array([i * px, j * py])) * dbu
                component.add_polygon([tuple(v) for v in shifted], layer=layer)
    component.info["period_um"] = (px * dbu, py * dbu)
    component.info["n_cells"] = tuple(n_cells)
    component.info["polygons"] = tuple(_as_vertices(v * dbu) for v in snapped)
    return component


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _shape_key(points: np.ndarray) -> tuple:
    """A translation-invariant key: the vertices relative to the bbox corner, rotated
    to start at the smallest one (so a polygon's starting vertex doesn't matter)."""
    relative = points - points.min(axis=0)
    rows = [tuple(r) for r in relative]
    start = rows.index(min(rows))
    return tuple(rows[start:] + rows[:start])


def _infer_period(corners: np.ndarray, axis: int) -> int | None:
    other = 1 - axis
    steps = []
    for value in np.unique(corners[:, other]):
        line = np.unique(corners[corners[:, other] == value, axis])
        if len(line) > 1:
            steps.append(int(np.min(np.diff(line))))
    return min(steps) if steps else None


def extract_unit_cell(
    component: gf.Component,
    layer: gf.typings.LayerSpec,
    period_um: tuple[float, float] | None = None,
) -> UnitCell:
    """Read a 2D-periodic pattern back from the polygons of ``component`` on ``layer``.

    The polygons are grouped into translated copies of one another. Each group
    must fill the same complete ``nx × ny`` lattice, so every cell holds the
    same motif. Shapes must not overlap, even across cell edges, and must not
    have holes. Anything else raises ``ValueError``, because a solver fed a
    guessed geometry returns a confident, wrong answer.

    Args:
        component: A gdsfactory component, e.g. from ``gf.import_gds``.
        layer: The layer the pattern is drawn on.
        period_um: The lattice period. Inferred from the pattern when every
            cell holds one copy of each distinct shape; required when a cell
            repeats a shape (a dimer of identical discs, say).
    """
    dbu = component.kcl.dbu
    region = kf.kdb.Region(component.begin_shapes_rec(gf.get_layer(layer)))
    polygons = []
    for polygon in region.each():
        if polygon.holes():
            raise ValueError(
                f"a polygon on layer {layer} has a hole; draw the hole's material as its own "
                "shape on the background instead"
            )
        polygons.append(np.array([(p.x, p.y) for p in polygon.each_point_hull()], dtype=np.int64))
    if len(polygons) < 2:
        raise ValueError(
            f"need a repeated pattern on layer {layer}, found {len(polygons)} polygons"
        )

    groups: dict[tuple, list[np.ndarray]] = {}
    for points in polygons:
        groups.setdefault(_shape_key(points), []).append(points)
    corners = {key: np.array([p.min(axis=0) for p in members]) for key, members in groups.items()}

    if period_um is not None:
        period = tuple(round(p / dbu) for p in period_um)
    else:
        reference = max(corners.values(), key=len)
        period = (_infer_period(reference, 0), _infer_period(reference, 1))
        if None in period:
            raise ValueError(
                "cannot infer the period: the pattern does not repeat along both x and y. "
                "Pass period_um, or draw at least two cells in each direction."
            )
    px, py = period

    # every shape: a set of complete, identical lattices
    counts = None
    motif = []  # (points, first corner) per shape in one cell
    for key, members in groups.items():
        c = corners[key]
        residues = {(int(x) % px, int(y) % py) for x, y in c}
        for rx, ry in residues:
            sub = c[(c[:, 0] % px == rx) & (c[:, 1] % py == ry)]
            i, j = (sub[:, 0] - sub[:, 0].min()) // px, (sub[:, 1] - sub[:, 1].min()) // py
            shape = (int(i.max()) + 1, int(j.max()) + 1)
            if len({(a, b) for a, b in zip(i, j, strict=True)}) != len(sub) or (
                shape[0] * shape[1] != len(sub)
            ):
                raise ValueError(
                    f"a shape on layer {layer} does not fill a complete lattice at period "
                    f"{px * dbu:g} × {py * dbu:g} µm; the pattern is not a periodic array"
                )
            if counts is None:
                counts = shape
            elif counts != shape:
                raise ValueError(
                    f"shapes on layer {layer} repeat {counts} and {shape} times; every cell "
                    "must hold the same motif"
                )
            first = sub[np.lexsort((sub[:, 1], sub[:, 0]))[0]]
            member = next(p for p in members if np.array_equal(p.min(axis=0), first))
            motif.append((member, first))

    origin = np.min([first for _, first in motif], axis=0)
    cell_polygons = tuple((points - origin) for points, _ in motif)

    # the motif, repeated on the lattice, must be exactly what is drawn
    drawn = {(_shape_key(points), tuple(points.min(axis=0))) for points in polygons}
    rebuilt = {
        (_shape_key(points), tuple(first + np.array([i * px, j * py])))
        for points, first in motif
        for i in range(counts[0])
        for j in range(counts[1])
    }
    if drawn != rebuilt or len(polygons) != len(motif) * counts[0] * counts[1]:
        raise ValueError(
            f"the pattern on layer {layer} is not one motif repeated at period "
            f"{px * dbu:g} × {py * dbu:g} µm; if a cell repeats a shape, pass period_um"
        )

    # no overlaps, including between a cell's shapes and its neighbours'
    tiled = kf.kdb.Region()
    area = 0
    extent = np.ptp(np.concatenate(cell_polygons), axis=0)
    reach_x, reach_y = int(extent[0] // px) + 1, int(extent[1] // py) + 1
    for shift_x in range(-reach_x, reach_x + 1):
        for shift_y in range(-reach_y, reach_y + 1):
            for points in cell_polygons:
                shifted = points + np.array([shift_x * px, shift_y * py])
                polygon = kf.kdb.Polygon([kf.kdb.Point(int(x), int(y)) for x, y in shifted])
                tiled.insert(polygon)
                area += polygon.area()
    if tiled.merged().area() < area:
        raise ValueError(
            f"shapes on layer {layer} overlap (within a cell or across its edges) at period "
            f"{px * dbu:g} × {py * dbu:g} µm; the solver needs non-overlapping shapes. If a "
            "cell repeats a shape, pass period_um"
        )

    return UnitCell(
        period_um=(px * dbu, py * dbu),
        polygons=tuple(p.astype(float) * dbu for p in cell_polygons),
        n_cells=counts,
        origin_um=(float(origin[0] * dbu), float(origin[1] * dbu)),
    )


# ---------------------------------------------------------------------------
# Stack and solver layers
# ---------------------------------------------------------------------------


def _inside_periodic(point: np.ndarray, cell: UnitCell) -> bool:
    px, py = cell.period_um
    for polygon in cell.polygons:
        polygon = np.asarray(polygon, dtype=float)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if _point_in_polygon(point - np.array([sx * px, sy * py]), polygon):
                    return True
    return False


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _sample_points(cell: UnitCell, samples: int):
    """Cell-coordinate points inside each shape, and around them."""
    px, py = cell.period_um
    fractions = (np.arange(4 * samples) + 0.5) / (4 * samples)
    grid = np.array([(a * px, b * py) for a in fractions for b in fractions])
    inside = []
    for polygon in cell.polygons:
        polygon = np.asarray(polygon, dtype=float)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        span = (np.arange(samples + 2)[1:-1]) / (samples + 1)
        candidates = [low + (high - low) * np.array([a, b]) for a in span for b in span]
        points = [p for p in candidates if _point_in_polygon(p, polygon)]
        if not points:
            points = [polygon.mean(axis=0)]
        inside.extend(points[:samples])
    outside = [p for p in grid if not _inside_periodic(p, cell)][: 2 * samples]
    if not outside:
        raise ValueError("the shapes cover the whole cell; there is no background to sample")
    return inside, outside


def crossed_grating_stack(
    component: gf.Component,
    pattern_layer: gf.typings.LayerSpec,
    layer_stack: LayerStack,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    period_um: tuple[float, float] | None = None,
    samples: int = 3,
) -> tuple[UnitCell, VerticalStack]:
    """A 2D-periodic pattern and the patterned stack its etch makes in the PDK's LayerStack.

    The stack is sampled at points inside every shape and around them, in the
    first cell and in one in the middle of the array; every inside sample must
    agree, and every outside one, or another layer is patterned differently and
    a ``ValueError`` says so. Where the two columns differ, the layer is
    patterned (one etch shared by every patterned layer).
    """
    cell = extract_unit_cell(component, pattern_layer, period_um)
    px, py = cell.period_um
    x0, y0 = cell.origin_um
    sites = sorted({(0, 0), (cell.n_cells[0] // 2, cell.n_cells[1] // 2)})
    inside, outside = _sample_points(cell, samples)

    def column(points, where):
        profiles = {
            tuple(
                vertical_profile(component, layer_stack, (x0 + i * px + p[0], y0 + j * py + p[1]))
            )
            for i, j in sites
            for p in points
        }
        if len(profiles) != 1:
            raise ValueError(
                f"The stack varies {where} the shapes on {pattern_layer}: another layer is "
                "patterned differently there, so this is not a single 2D grating in this "
                "layer stack."
            )
        return list(profiles.pop())

    stack = patterned_stack(column(inside, "inside"), column(outside, "around"), materials, ambient)
    return cell, stack


def crossed_grating_layers(
    cell: UnitCell,
    stack: VerticalStack | Sequence[StackLayer],
    wavelength_um: Any,
) -> list[rcwa2d.PatternedLayer]:
    """:mod:`~hyperbolic_optics.rcwa2d` layers for a unit cell etched into a stack.

    Every patterned layer shares the cell's shapes (a single etch). Polygons,
    thicknesses and the wavelength may be JAX values.
    """
    layers = stack.layers if isinstance(stack, VerticalStack) else stack
    out = []
    for spec in layers:
        eps = material_tensor(spec.material, wavelength_um, spec.rotation_deg)
        if spec.background is None:
            out.append(rcwa2d.uniform(eps, spec.thickness_um))
        else:
            background = material_tensor(spec.background, wavelength_um)
            shapes = tuple(rcwa2d.Shape(jnp.asarray(p), eps) for p in cell.polygons)
            out.append(rcwa2d.PatternedLayer(spec.thickness_um, background, shapes))
    return out
