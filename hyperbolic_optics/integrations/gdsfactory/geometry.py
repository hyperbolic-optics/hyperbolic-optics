"""Line-grating geometry between gdsfactory layouts and the RCWA solver.

- :func:`line_grating`, a PCell for a grating of identical parallel ridges, with
  its period and ridge width snapped to the layout grid so that the exported
  geometry is exactly periodic;
- :func:`extract_line_grating`, which reads that geometry back from any
  gdsfactory component (or imported GDS), and refuses anything that is not a
  clean line grating rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gdsfactory as gf
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "hyperbolic_optics.integrations.gdsfactory needs gdsfactory: "
        'pip install "hyperbolic-optics[gdsfactory]"'
    ) from error

__all__ = [
    "LineGrating",
    "extract_line_grating",
    "line_grating",
]


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
        origin_um: Layout position ``(x, y)`` of the first ridge's lower-left
            corner, so the stack beneath it can be sampled.
    """

    period_um: float
    fill: float
    offset: float = 0.0
    n_periods: int = 1
    length_um: float = 0.0
    origin_um: tuple[float, float] = (0.0, 0.0)


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
    component: gf.Component,
    layer: gf.typings.LayerSpec,
    tolerance_um: float | None = None,
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
        origin_um=(float(x0[0]), float(np.mean(y0))),
    )
