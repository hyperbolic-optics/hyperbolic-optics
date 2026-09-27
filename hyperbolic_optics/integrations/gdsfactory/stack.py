"""Vertical stacks: hand-written, or read from a gdsfactory PDK's LayerStack.

A :class:`VerticalStack` is an incident half-space, layers from the incident side
down, and an exit half-space. Layers are uniform films or, for a line grating,
patterned: ridges of one material with grooves of another.

:func:`vertical_stack` reads the stack at one point of a layout from the PDK's
``LayerStack``: the levels present there (``WAFER`` levels everywhere, others
where their -- possibly derived, boolean -- layer has shapes), overlaps resolved
by ``mesh_order`` as gdsfactory defines it (lower wins). :func:`grating_stack`
compares the stack inside the ridges of a line grating with the stack in its
gaps: where they differ, the layer is patterned. So etch depth, partial etches
and what fills the grooves all come from the PDK, not from the user.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gdsfactory as gf
import kfactory as kf
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack, LogicalLayer

from hyperbolic_optics import rcwa
from hyperbolic_optics.integrations.gdsfactory.geometry import LineGrating, extract_line_grating
from hyperbolic_optics.integrations.gdsfactory.materials import (
    MaterialSpec,
    Oriented,
    material_tensor,
    resolve_material,
)

__all__ = [
    "StackLayer",
    "VerticalStack",
    "grating_layers",
    "grating_stack",
    "patterned_stack",
    "vertical_profile",
    "vertical_stack",
]


@dataclass(frozen=True)
class StackLayer:
    """One layer of the vertical stack, from the incident side down.

    Attributes:
        thickness_um: Thickness in µm (may be a JAX value for optimisation).
        material: The layer's material -- the ridges or shapes, if ``background`` is set.
        background: Material around them. ``None`` makes the layer unpatterned
            (a uniform film of ``material``).
        rotation_deg: Orientation of the layer's crystals, ``material`` and
            ``background`` alike (payload-style ``(x, y, z)``, degrees), applied
            after an :class:`Oriented` material's own cut: a twist.
        name: Label, e.g. the PDK level it came from.
        polygons: The layer's own shapes (vertices ``[V, 2]`` in µm, cell
            coordinates) for a 2D-periodic cell whose layers are patterned
            differently; ``None`` to use the unit cell's.
    """

    thickness_um: Any
    material: MaterialSpec | Oriented
    background: MaterialSpec | Oriented | None = None
    rotation_deg: Sequence = (0.0, 0.0, 0.0)
    name: str = ""
    polygons: tuple | None = None


@dataclass(frozen=True)
class VerticalStack:
    """Half-spaces and layers, incident side first.

    Attributes:
        layers: The finite layers, from the incident side down.
        incident: Material of the incident half-space (isotropic, lossless).
        exit: Material of the exit half-space (isotropic, lossless).
    """

    layers: tuple[StackLayer, ...]
    incident: MaterialSpec = 1.0
    exit: MaterialSpec = 1.0

    @property
    def patterned(self) -> bool:
        return any(layer.background is not None for layer in self.layers)

    def half_space_permittivity(self, which: str, wavelength_um: Any) -> np.ndarray:
        """Real, scalar ε of the incident or exit half-space; refuses anything else."""
        spec = self.incident if which == "incident" else self.exit
        tensor = np.asarray(material_tensor(spec, np.asarray(wavelength_um, dtype=float)))
        eps = tensor[..., 0, 0]
        isotropic = np.allclose(tensor, eps[..., None, None] * np.eye(3), atol=1e-12)
        if not isotropic or np.abs(eps.imag).max() > 1e-12 or eps.real.min() <= 0:
            raise ValueError(
                f"The {which} half-space ({spec!r}) must be isotropic and lossless; the solvers "
                "need real, positive ε there. Put an absorbing or anisotropic material in a "
                "finite layer instead."
            )
        return eps.real


def grating_layers(
    grating: LineGrating,
    stack: VerticalStack | Sequence[StackLayer],
    wavelength_um: Any,
    edge_width: float = 0.0,
) -> list[rcwa.GratingLayer]:
    """:mod:`~hyperbolic_optics.rcwa` layers for a line grating etched into a stack.

    Every patterned layer shares the grating's geometry (a single etch). Fields
    of ``grating`` may be JAX values, so fill and period can be optimised.
    """
    layers = stack.layers if isinstance(stack, VerticalStack) else stack
    out = []
    for spec in layers:
        eps = material_tensor(spec.material, wavelength_um, spec.rotation_deg)
        if spec.background is None:
            out.append(rcwa.uniform_layer(eps, spec.thickness_um))
        else:
            groove = material_tensor(spec.background, wavelength_um, spec.rotation_deg)
            out.append(
                rcwa.lamellar_layer(
                    eps,
                    groove,
                    grating.fill,
                    spec.thickness_um,
                    offset=grating.offset,
                    edge_width=edge_width,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reading a PDK LayerStack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Interval:
    level: str
    material: str
    zmin: float
    zmax: float


def _mesh_order(level) -> int:
    info = level.info or {}
    return info.get("mesh_order", level.mesh_order)


def vertical_profile(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    everywhere_layers: Sequence = ("WAFER",),
) -> list[_Interval]:
    """The PDK's material column at one point of a layout, bottom to top.

    Levels on an ``everywhere_layers`` layer (gdsfactory's ``WAFER``) are
    present at every point; others where their layer -- logical or derived --
    has shapes. Where present levels overlap in z, the lowest ``mesh_order``
    wins; a tie between different materials, or a gap no level covers, raises.
    """
    dbu = component.kcl.dbu
    point = kf.kdb.Point(round(point_um[0] / dbu), round(point_um[1] / dbu))
    everywhere = {gf.get_layer(layer) for layer in everywhere_layers}

    present = []
    for name, level in layer_stack.layers.items():
        if not level.thickness:
            continue
        logical = isinstance(level.layer, LogicalLayer)
        if logical and gf.get_layer(level.layer.layer) in everywhere:
            inside = True
        else:
            inside = any(p.inside(point) for p in level.layer.get_shapes(component).each())
        if inside:
            low, high = sorted((level.zmin, level.zmin + level.thickness))
            present.append((name, level, low, high))
    if not present:
        raise ValueError(f"No layer-stack level is present at {point_um}.")

    edges = sorted({z for _, _, low, high in present for z in (low, high)})
    profile: list[_Interval] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        covering = [(n, lv) for n, lv, lo, hi in present if lo <= low and hi >= high]
        if not covering:
            raise ValueError(f"No layer-stack level covers z = {low:g}-{high:g} µm at {point_um}.")
        best = min(_mesh_order(lv) for _, lv in covering)
        winners = [(n, lv) for n, lv in covering if _mesh_order(lv) == best]
        if len({lv.material for _, lv in winners}) > 1:
            names = ", ".join(n for n, _ in winners)
            raise ValueError(f"Levels {names} tie on mesh_order at z = {low:g}-{high:g} µm.")
        name, level = winners[0]
        if profile and profile[-1].level == name:
            profile[-1] = _Interval(name, level.material, profile[-1].zmin, high)
        else:
            profile.append(_Interval(name, level.material, low, high))
    return profile


def vertical_stack(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
) -> VerticalStack:
    """The unpatterned stack at ``point_um``: ambient above, lowest level as the exit.

    Args:
        component: The layout.
        layer_stack: The PDK's ``LayerStack`` (e.g. ``gf.get_active_pdk().layer_stack``).
        point_um: Layout position ``(x, y)`` in µm.
        materials: PDK material name → specification, over
            :data:`~hyperbolic_optics.integrations.gdsfactory.materials.DEFAULT_MATERIALS`.
        ambient: The incident half-space above the stack.
    """
    profile = vertical_profile(component, layer_stack, point_um)
    exit_material = resolve_material(profile[0].material, materials)
    layers = tuple(
        StackLayer(
            thickness_um=interval.zmax - interval.zmin,
            material=resolve_material(interval.material, materials),
            name=interval.level,
        )
        for interval in reversed(profile[1:])
    )
    return VerticalStack(layers=layers, incident=ambient, exit=exit_material)


def grating_stack(
    component: gf.Component,
    grating_layer: gf.typings.LayerSpec,
    layer_stack: LayerStack,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    samples: int = 3,
) -> tuple[LineGrating, VerticalStack]:
    """A line grating and the patterned stack its etch makes in the PDK's LayerStack.

    The stack is sampled at ``samples`` points across a ridge and across a gap,
    in two periods; every ridge sample must agree, and every gap sample, or the
    layout is not a single line grating in this stack and a ``ValueError`` says
    so. Where the ridge and gap columns differ, the layer is patterned.
    """
    grating = extract_line_grating(component, grating_layer)
    period, width = grating.period_um, grating.fill * grating.period_um
    x0, y0 = grating.origin_um
    y = y0 + grating.length_um / 2
    periods = sorted({0, grating.n_periods // 2})

    def column(x_in_period, span):
        profiles = [
            tuple(vertical_profile(component, layer_stack, (x0 + k * period + x_in_period(i), y)))
            for k in periods
            for i in range(samples)
        ]
        if len(set(profiles)) != 1:
            raise ValueError(
                f"The stack varies within the {span} of the grating on {grating_layer}: "
                "another layer is patterned differently there, so this is not a single "
                "line grating in this layer stack."
            )
        return list(profiles[0])

    ridge = column(lambda i: width * (i + 0.5) / samples, "ridges")
    gap = column(lambda i: width + (period - width) * (i + 0.5) / samples, "gaps")

    return grating, patterned_stack(ridge, gap, materials, ambient)


def patterned_stack(
    inside: Sequence[_Interval],
    outside: Sequence[_Interval],
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
) -> VerticalStack:
    """The stack from the PDK columns inside and outside a pattern's shapes.

    Where the two columns differ, the layer is patterned: the inside material
    in the shapes, the outside one around them. The lowest level, which must
    be the same in both, is the exit half-space.
    """
    ridge, gap = list(inside), list(outside)
    edges = sorted({z for part in (ridge, gap) for iv in part for z in (iv.zmin, iv.zmax)})

    def material_at(profile, low, high):
        for interval in profile:
            if interval.zmin <= low and interval.zmax >= high:
                return interval.level, interval.material
        return "ambient", None  # above this column's top

    pieces = []  # (ridge (level, material), gap (level, material), zmin, zmax), bottom up
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        key = (material_at(ridge, low, high), material_at(gap, low, high))
        if pieces and pieces[-1][0] == key:
            pieces[-1] = (key, pieces[-1][1], high)
        else:
            pieces.append((key, low, high))

    (exit_ridge, exit_gap), _, _ = pieces[0]
    if exit_ridge[1] != exit_gap[1]:
        raise ValueError(
            "The lowest level differs between ridges and gaps; it must be the substrate."
        )

    def spec(level_material):
        _, material = level_material
        return ambient if material is None else resolve_material(material, materials)

    layers = []
    for (ridge_part, gap_part), low, high in reversed(pieces[1:]):
        same = ridge_part[1] == gap_part[1]
        layers.append(
            StackLayer(
                thickness_um=high - low,
                material=spec(ridge_part),
                background=None if same else spec(gap_part),
                name=ridge_part[0] if same else f"{ridge_part[0]} / {gap_part[0]}",
            )
        )
    exit_spec = spec(exit_ridge)
    return VerticalStack(layers=tuple(layers), incident=ambient, exit=exit_spec)
