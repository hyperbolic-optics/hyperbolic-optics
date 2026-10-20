"""2D-periodic layouts: grid-snapped PCells, exact extraction, PDK stacks and rcwa2d layers."""

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
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec
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
    """The repeating unit of a 2D-periodic pattern.

    Attributes:
        period_um: ``(Λx, Λy)`` in µm for a rectangular lattice; ``None`` when
            ``lattice_um`` gives another.
        polygons: The shapes of one cell, each ``[V, 2]`` in µm relative to
            the cell origin. They may extend past the cell; the solver
            periodises them. May be JAX values, for optimisation.
        n_cells: ``(n₁, n₂)`` cells drawn in the layout, along ``a₁`` and ``a₂``.
        origin_um: Layout position of the first cell's origin, so the stack
            beneath it can be sampled.
        lattice_um: Primitive vectors ``((a1x, a1y), (a2x, a2y))`` in µm, for
            any lattice (hexagonal, oblique); overrides ``period_um``.
    """

    period_um: tuple[Any, Any] | None
    polygons: tuple[Any, ...]
    n_cells: tuple[int, int] = (1, 1)
    origin_um: tuple[float, float] = (0.0, 0.0)
    lattice_um: tuple[tuple[Any, Any], tuple[Any, Any]] | None = None

    @property
    def lattice(self) -> rcwa2d.Lattice:
        """The cell's lattice, for :func:`hyperbolic_optics.rcwa2d.solve_crossed_grating`."""
        if self.lattice_um is not None:
            return rcwa2d.Lattice(tuple(self.lattice_um[0]), tuple(self.lattice_um[1]))
        return rcwa2d.Lattice.rectangular(*self.period_um)

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        lattice = self.lattice
        return np.asarray(lattice.a1, dtype=float), np.asarray(lattice.a2, dtype=float)


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
    lattice_um: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> gf.Component:
    """PCell: the ``polygons`` of one cell, repeated ``n_cells`` times on a lattice.

    The lattice is rectangular at ``period_um``, or given by its primitive
    vectors ``lattice_um = ((a1x, a1y), (a2x, a2y))`` -- e.g. a hexagonal one,
    ``((a, 0), (a/2, a·√3/2))``. The lattice vectors and every vertex are
    snapped to the layout grid (the database unit), so every cell drawn is an
    exact translate of the first; the snapped values are in ``component.info``
    and are what :func:`extract_unit_cell` recovers. Pass polygon vertices as
    nested tuples (a PCell's arguments must be hashable), e.g.
    ``tuple(map(tuple, rcwa2d.ellipse(...).tolist()))``.
    """
    component = gf.Component()
    dbu = component.kcl.dbu
    if lattice_um is None:
        lattice_um = ((period_um[0], 0.0), (0.0, period_um[1]))
    a1, a2 = (np.round(np.asarray(v, dtype=float) / dbu).astype(int) for v in lattice_um)
    if a1[0] * a2[1] - a1[1] * a2[0] == 0:
        raise ValueError(f"lattice {lattice_um} is degenerate on this grid")
    snapped = [np.round(np.asarray(polygon, dtype=float) / dbu).astype(int) for polygon in polygons]
    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            for vertices in snapped:
                shifted = (vertices + i * a1 + j * a2) * dbu
                component.add_polygon([tuple(v) for v in shifted], layer=layer)
    rectangular = a1[1] == 0 and a2[0] == 0
    component.info["period_um"] = (a1[0] * dbu, a2[1] * dbu) if rectangular else None
    component.info["lattice_um"] = (tuple(a1 * dbu), tuple(a2 * dbu))
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


def _decompose(corners: np.ndarray, a1: np.ndarray, a2: np.ndarray):
    """Exact lattice coordinates of integer points: ``c = (q₁·a₁ + q₂·a₂)/det``.

    Returns ``(q, det)`` with ``q`` integer ``[K, 2]``: ``q // det`` is the cell
    index and ``q % det`` the sublattice (residue) a point belongs to.
    """
    det = int(a1[0] * a2[1] - a1[1] * a2[0])
    adjugate = np.array([[a2[1], -a2[0]], [-a1[1], a1[0]]], dtype=np.int64)
    q = corners.astype(np.int64) @ adjugate.T
    if det < 0:
        q, det = -q, -det
    return q, det


def _complete_lattices(groups, corners, a1, a2):
    """Split each shape's copies into complete lattices; ``None`` if they are not."""
    counts, motif = None, []
    for key, members in groups.items():
        q, det = _decompose(corners[key], a1, a2)
        for residue in {tuple(r) for r in q % det}:
            chosen = np.all(q % det == residue, axis=1)
            index = q[chosen] // det
            index = index - index.min(axis=0)
            shape = tuple(int(n) + 1 for n in index.max(axis=0))
            if len({tuple(i) for i in index}) != len(index) or shape[0] * shape[1] != len(index):
                return None
            if counts is None:
                counts = shape
            elif counts != shape:
                return None
            first_row = np.flatnonzero(chosen)[np.lexsort((index[:, 1], index[:, 0]))[0]]
            first = corners[key][first_row]
            member = next(p for p in members if np.array_equal(p.min(axis=0), first))
            motif.append((member, first))
    return counts, motif


def _steps(a1, a2, dbu) -> tuple[np.ndarray, np.ndarray]:
    """Lattice vectors in whole grid steps."""
    return tuple(np.rint(np.asarray(a, float) / dbu).astype(np.int64) for a in (a1, a2))


def _altitude(a1, a2) -> float:
    """The cell's smallest height: its area over its longer side."""
    return abs(a1[0] * a2[1] - a1[1] * a2[0]) / max(np.hypot(*a1), np.hypot(*a2))


def _tiled(region, a1, a2, reach: int = 1) -> kf.kdb.Region:
    """``region`` and its lattice images up to ``reach`` steps away, unmerged (grid steps)."""
    out = kf.kdb.Region()
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            dx, dy = i * a1 + j * a2
            out += region.transformed(kf.kdb.Trans(int(dx), int(dy)))
    return out


def _self_overlaps(region, a1, a2, reach: int = 1) -> bool:
    """Whether ``region``, tiled on the lattice, overlaps itself or its images."""
    copies = (2 * reach + 1) ** 2
    return _tiled(region, a1, a2, reach).area() < copies * sum(p.area() for p in region.each())


def _hulls(region) -> list[np.ndarray]:
    """Each polygon's vertices in grid steps ``[V, 2]``; holes are refused."""
    if any(p.holes() for p in region.each()):
        raise ValueError(
            "a polygon has a hole; draw the hole's material as its own shape on the background"
        )
    return [np.array([(p.x, p.y) for p in q.each_point_hull()], np.int64) for q in region.each()]


def _overlaps(cell_polygons, a1: np.ndarray, a2: np.ndarray) -> bool:
    """Whether the motif, tiled on the lattice, overlaps itself (within or across cells)."""
    region = kf.kdb.Region()
    for points in cell_polygons:
        region.insert(kf.kdb.Polygon([kf.kdb.Point(int(x), int(y)) for x, y in points]))
    extent = np.ptp(np.concatenate(cell_polygons), axis=0)
    return _self_overlaps(region, a1, a2, int(np.hypot(*extent) // _altitude(a1, a2)) + 2)


def _candidate_bases(reference: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Primitive-basis candidates from the shortest translations between identical shapes.

    Each is right-handed (``a₁ × a₂ > 0``) with ``a₁`` pointing into ``x > 0``.
    They are ranked shortest first (a reduced basis), then with ``a₁`` closest
    to x, so a rectangular array comes back as ``((Λx, 0), (0, Λy))`` and a
    hexagonal one with ``a₁`` along x.
    """
    steps = (reference[None, :, :] - reference[:, None, :]).reshape(-1, 2)
    steps = np.unique(steps[(steps != 0).any(axis=1)], axis=0)
    steps = steps[(steps[:, 0] > 0) | ((steps[:, 0] == 0) & (steps[:, 1] > 0))]
    steps = steps[np.argsort(np.hypot(steps[:, 0], steps[:, 1]), kind="stable")][:24]
    ranked = []
    for d1 in steps:
        for d2 in steps:
            cross = int(d1[0] * d2[1] - d1[1] * d2[0])
            if cross == 0:
                continue
            a2 = d2 if cross > 0 else -d2
            length = np.hypot(*d1) + np.hypot(*a2)
            alignment = abs(d1[0]) / np.hypot(*d1)
            ranked.append((round(length, 6), -alignment, a2[1] < 0, tuple(d1), tuple(a2)))
    ranked.sort()
    return [(np.array(a1), np.array(a2)) for *_, a1, a2 in ranked]


def extract_unit_cell(
    component: gf.Component,
    layer: gf.typings.LayerSpec,
    period_um: tuple[float, float] | None = None,
    lattice_um: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> UnitCell:
    """Read a 2D-periodic pattern back from the polygons of ``component`` on ``layer``.

    The polygons are grouped into translated copies of one another. Each group
    must fill the same complete ``n₁ × n₂`` lattice, so every cell holds the
    same motif. Shapes must not overlap, even across cell edges, and must not
    have holes. Anything else raises ``ValueError``, because a solver fed a
    guessed geometry returns a confident, wrong answer.

    Args:
        component: A gdsfactory component, e.g. from ``gf.import_gds``.
        layer: The layer the pattern is drawn on.
        period_um: A rectangular lattice's period.
        lattice_um: Any lattice's primitive vectors ``((a1x, a1y), (a2x, a2y))``.
            Without either, the lattice is inferred -- rectangular, hexagonal or
            oblique, motifs that repeat a shape (a dimer) included: the
            shortest basis under which the drawing is one motif on a complete
            parallelogram of cells. Pass it to impose a lattice instead.
    """
    dbu = component.kcl.dbu
    polygons = _hulls(kf.kdb.Region(component.begin_shapes_rec(gf.get_layer(layer))))
    if len(polygons) < 2:
        raise ValueError(
            f"need a repeated pattern on layer {layer}, found {len(polygons)} polygons"
        )

    groups: dict[tuple, list[np.ndarray]] = {}
    for points in polygons:
        groups.setdefault(_shape_key(points), []).append(points)
    corners = {key: np.array([p.min(axis=0) for p in members]) for key, members in groups.items()}

    if lattice_um is None and period_um is not None:
        lattice_um = ((period_um[0], 0.0), (0.0, period_um[1]))
    if lattice_um is not None:
        bases = [
            tuple(np.round(np.asarray(v, dtype=float) / dbu).astype(np.int64) for v in lattice_um)
        ]
    else:
        reference = max(corners.values(), key=len)
        bases = _candidate_bases(reference)
        if not bases:
            raise ValueError(
                "cannot infer the lattice: the pattern does not repeat in two directions. "
                "Pass period_um or lattice_um, or draw at least two cells each way."
            )

    found, overlapping = None, None
    for a1, a2 in bases:
        split = _complete_lattices(groups, corners, a1, a2)
        if split is None:
            continue
        counts, motif = split
        origin = np.min([first for _, first in motif], axis=0)
        drawn = {(_shape_key(points), tuple(points.min(axis=0))) for points in polygons}
        rebuilt = {
            (_shape_key(points), tuple(first + i * a1 + j * a2))
            for points, first in motif
            for i in range(counts[0])
            for j in range(counts[1])
        }
        if drawn != rebuilt or len(polygons) != len(motif) * counts[0] * counts[1]:
            continue
        if _overlaps(tuple(points - origin for points, _ in motif), a1, a2):
            overlapping = overlapping or (a1, a2)
            continue
        found = (a1, a2, counts, motif, origin)
        break
    if found is None and overlapping is not None:
        a1, a2 = overlapping
        raise ValueError(
            f"shapes on layer {layer} overlap (within a cell or across its edges) on the lattice "
            f"{tuple(a1 * dbu)}, {tuple(a2 * dbu)} µm; the solver needs non-overlapping shapes"
        )
    if found is None:
        described = "the given lattice" if lattice_um is not None else "any lattice inferred"
        raise ValueError(
            f"the pattern on layer {layer} does not fill a complete lattice at {described}: it "
            "is not one motif repeated on a periodic array. If a cell repeats a shape, pass "
            "period_um or lattice_um"
        )
    a1, a2, counts, motif, origin = found
    cell_polygons = tuple((points - origin) for points, _ in motif)

    rectangular = a1[1] == 0 and a2[0] == 0
    return UnitCell(
        period_um=(a1[0] * dbu, a2[1] * dbu) if rectangular else None,
        polygons=tuple(p.astype(float) * dbu for p in cell_polygons),
        n_cells=counts,
        origin_um=(float(origin[0] * dbu), float(origin[1] * dbu)),
        lattice_um=None if rectangular else (tuple(a1 * dbu), tuple(a2 * dbu)),
    )


# ---------------------------------------------------------------------------
# Stack and solver layers
# ---------------------------------------------------------------------------


def _inside_periodic(point: np.ndarray, cell: UnitCell) -> bool:
    a1, a2 = cell._vectors()
    for polygon in cell.polygons:
        polygon = np.asarray(polygon, dtype=float)
        for s1 in (-2, -1, 0, 1, 2):
            for s2 in (-2, -1, 0, 1, 2):
                if _point_in_polygon(point - s1 * a1 - s2 * a2, polygon):
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
    a1, a2 = cell._vectors()
    fractions = (np.arange(4 * samples) + 0.5) / (4 * samples)
    grid = np.array([a * a1 + b * a2 for a in fractions for b in fractions])
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
    lattice_um: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[UnitCell, VerticalStack]:
    """A 2D-periodic pattern and the patterned stack its etch makes in the PDK's LayerStack.

    The stack is sampled at points inside every shape and around them, in the
    first cell and in one in the middle of the array; every inside sample must
    agree, and every outside one, or another layer is patterned differently and
    a ``ValueError`` says so. Where the two columns differ, the layer is
    patterned (one etch shared by every patterned layer).
    """
    cell = extract_unit_cell(component, pattern_layer, period_um, lattice_um)
    a1, a2 = cell._vectors()
    origin = np.asarray(cell.origin_um)
    sites = sorted({(0, 0), (cell.n_cells[0] // 2, cell.n_cells[1] // 2)})
    inside, outside = _sample_points(cell, samples)

    def column(points, where):
        profiles = {
            tuple(vertical_profile(component, layer_stack, tuple(origin + i * a1 + j * a2 + p)))
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
    cell: UnitCell | None,
    stack: VerticalStack | Sequence[StackLayer],
    wavelength_um: Any,
) -> list[rcwa2d.PatternedLayer]:
    """:mod:`~hyperbolic_optics.rcwa2d` layers for a unit cell etched into a stack.

    A patterned layer uses its own ``polygons`` if it has them, else the
    cell's shapes (a single etch); ``cell`` may be ``None`` when every
    patterned layer has its own. Polygons, thicknesses, orientations and the
    wavelength may be JAX values.
    """
    layers = stack.layers if isinstance(stack, VerticalStack) else stack
    out = []
    for spec in layers:
        eps, background = spec.tensors(wavelength_um)
        if background is None:
            out.append(rcwa2d.uniform(eps, spec.thickness_um))
        else:
            polygons = spec.polygons if spec.polygons is not None else cell.polygons
            shapes = tuple(rcwa2d.Shape(jnp.asarray(p), eps) for p in polygons)
            out.append(rcwa2d.PatternedLayer(spec.thickness_um, background, shapes))
    return out
