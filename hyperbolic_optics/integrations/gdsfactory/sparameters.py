"""gplugins-style S-parameters: ``write_sparameters_*(component, …)``, cached.

Like gdsfactory's simulation plugins (``gplugins.tidy3d.write_sparameters``
and friends), each function takes a component and the PDK's ``LayerStack``,
computes S-parameters over wavelength, saves them as ``.npz`` in gdsfactory's
S-parameter directory (``gf.config.PATH.sparameters``), keyed by a hash of the
geometry and the settings, and returns them. A second call with the same
component and settings loads the file.

The file's keys are ``"port@mode,port@mode"`` (input first, as in SAX and
gplugins) and ``"wavelengths"`` (µm), so ``gplugins.sax.read.model_from_npz``
reads single-mode results as they are. For several modes per port (p and s,
TE and TM), use :func:`model_from_npz` here, which keeps them apart. A JSON
file beside the ``.npz`` records the settings and the physics behind the
numbers (emission angles, directionality, effective indices).

- :func:`write_sparameters_grating_coupler`: a grating-coupler cell from the
  PDK (e.g. ``gf.components.grating_coupler_rectangular``), waveguide port to
  fibre port, from its leaky mode (:mod:`hyperbolic_optics.couplers`);
- :func:`write_sparameters_stack`: the four-port plane-wave S-matrix of the
  stack at a point of a layout.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
from gdsfactory.technology.layer_stack import LayerStack

from hyperbolic_optics import couplers, resonances
from hyperbolic_optics.integrations.gdsfactory.geometry import LineGrating, extract_grating_teeth
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec, material_tensor
from hyperbolic_optics.integrations.gdsfactory.simulate import stack_sax_model
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    grating_layers,
    teeth_stack,
    vertical_stack,
)

__all__ = [
    "component_hash",
    "get_sparameters_path",
    "model_from_npz",
    "write_sparameters_grating_coupler",
    "write_sparameters_stack",
]


def component_hash(component: gf.Component) -> str:
    """A hash of a component's geometry: every polygon on every layer, in dbu."""
    digest = hashlib.md5()
    dbu = component.kcl.dbu
    for layer, polygons in sorted(
        component.get_polygons_points().items(), key=lambda kv: str(kv[0])
    ):
        digest.update(str(layer).encode())
        points = sorted(np.round(np.asarray(p) / dbu).astype(np.int64).tobytes() for p in polygons)
        for item in points:
            digest.update(item)
    return digest.hexdigest()


def _settings_hash(settings: Mapping[str, Any]) -> str:
    return hashlib.md5(json.dumps(_jsonable(settings), sort_keys=True).encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return _jsonable(np.asarray(value).tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def get_sparameters_path(
    component: gf.Component, dirpath: Path | str | None = None, **settings: Any
) -> Path:
    """Where a component's S-parameters for these settings live (gplugins' layout)."""
    dirpath = Path(dirpath) if dirpath is not None else Path(gf.config.PATH.sparameters)
    dirpath = dirpath / "hyperbolic_optics"
    dirpath.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5((component_hash(component) + _settings_hash(settings)).encode()).hexdigest()
    return dirpath / f"{component.name}_{key}.npz"


def _save(path: Path, sparameters: dict, metadata: dict) -> None:
    np.savez(path, **sparameters)
    path.with_suffix(".json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def model_from_npz(filepath: Path | str | Mapping[str, np.ndarray]):
    """A SAX model from saved S-parameters, interpolated over wavelength, modes kept.

    Returns ``model(wl) → SDict`` keyed ``("port@mode", "port@mode")``. Real and
    imaginary parts are interpolated linearly (``jnp.interp``), as
    ``gplugins.sax.read.model_from_npz`` does, which instead drops the mode.
    """
    data = _load(Path(filepath)) if not isinstance(filepath, Mapping) else dict(filepath)
    x = jnp.asarray(data["wavelengths"])
    order = jnp.argsort(x)
    x = x[order]
    entries = {
        tuple(key.split(",")): jnp.asarray(value)[order]
        for key, value in data.items()
        if key != "wavelengths"
    }

    def model(wl=1.55):
        wl = jnp.asarray(wl, dtype=float)
        return {
            ports: jnp.interp(wl, x, jnp.real(v)) + 1j * jnp.interp(wl, x, jnp.imag(v))
            for ports, v in entries.items()
        }

    return model


# ---------------------------------------------------------------------------
# Grating couplers
# ---------------------------------------------------------------------------


def _uniform(stack: VerticalStack, which: str) -> list[StackLayer]:
    """The stack with each patterned layer replaced by its ridge or groove material."""
    out = []
    for layer in stack.layers:
        material = (
            layer.material if which == "ridge" or layer.background is None else layer.background
        )
        out.append(StackLayer(layer.thickness_um, material, rotation_deg=layer.rotation_deg,
                              name=layer.name))  # fmt: skip
    return out


def _guided_indices(stack, wavelength_um) -> list[float]:
    """Effective indices of the fundamental mode under the ridges and under the grooves.

    Only a starting guess for the leaky-mode search. The outer layers (the
    cladding above, the box below) stand in for the half-spaces, because a
    substrate can be of higher index than the mode (silicon under a BOX): the
    mode is then leaky, and the search with the true half-spaces finds that.
    """

    def scalar(spec):
        return float(np.real(np.asarray(material_tensor(spec, wavelength_um))[0, 0]))

    top = min(float(stack.half_space_permittivity("incident", wavelength_um)),
              scalar(stack.layers[0].material))  # fmt: skip
    bottom = min(float(stack.half_space_permittivity("exit", wavelength_um)),
                 scalar(stack.layers[-1].material))  # fmt: skip
    indices = []
    for which in ("ridge", "groove"):
        layers = _uniform(stack, which)
        eps = jnp.stack([material_tensor(layer.material, wavelength_um, layer.rotation_deg)
                         for layer in layers])  # fmt: skip
        thickness = jnp.array([float(layer.thickness_um) for layer in layers])
        modes = resonances.slab_modes(eps, thickness, wavelength_um, top, bottom)
        if modes:
            indices.append(max(m.effective_index.real for m in modes))
    if not indices:
        raise ValueError("no guided mode in this stack at this wavelength; pass index_range")
    return indices


def write_sparameters_grating_coupler(
    component: gf.Component,
    layer_stack: LayerStack,
    wavelengths_um,
    grating_layer: gf.typings.LayerSpec = "WG",
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    waveguide_port: str = "o1",
    fiber_port: str = "o2",
    waist_um: float = 5.2,
    index_range: tuple[float, float] | None = None,
    max_order: int = 10,
    dirpath: Path | str | None = None,
    overwrite: bool = False,
    fiber_angle_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """S-parameters of a grating-coupler cell, waveguide port ↔ fibre port.

    The teeth are read along the coupler's axis (its waveguide port's y), so
    uniform, apodized, chirped and focusing (elliptical) couplers are all read
    tooth by tooth; the etched stack under them comes from the PDK's
    ``LayerStack``. The coupling is the local leaky-mode model
    (:func:`hyperbolic_optics.couplers.apodized_grating_coupler`): each tooth's
    leaky mode from the full-vector RCWA, the whole grating's emission
    overlapped with a fibre mode at its best position (and angle, unless
    ``fiber_angle_deg`` is given). Anisotropic layers (an x-cut LiNbO₃ film,
    ``materials={"ln": Oriented("LiNbO3", x_cut(…))}``) are solved in full.
    A focusing coupler's curvature is not modelled (the fibre mode is taken to
    match it across the grating).

    Args:
        component: The grating-coupler cell.
        layer_stack: The PDK's ``LayerStack``.
        wavelengths_um: Wavelengths to compute.
        grating_layer: The layer the teeth are drawn on.
        materials, ambient: As in :func:`.simulate_grating`.
        waveguide_port, fiber_port: The component's port names for the keys.
        waist_um: The fibre mode's waist (5.2 µm: SMF-28 at 1550 nm).
        index_range: Where to look for the leaky mode's ``Re kx₀/k₀`` at the
            first wavelength; by default around the guided modes under the
            ridges and grooves.
        max_order: Harmonics of the grating solve.
        dirpath: Where to save (``gf.config.PATH.sparameters`` by default).
        overwrite: Recompute even if saved.
        fiber_angle_deg: The fibre's angle in the top medium; by default the best.

    Returns:
        ``{"o1@TE,o2@TE": …, "o2@TE,o1@TE": …, "o1@TE,o2@TM": …, "o2@TM,o1@TE": …,
        "wavelengths": …}`` (with the given port names): amplitude transfer
        between the waveguide's TE mode and the fibre's TE (s) and TM (p)
        polarizations, reciprocal, zero phase (a lumped model). The JSON
        beside the file has the teeth, and at each wavelength the fibre's
        best position and angle and the first tooth's leaky mode.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    settings = dict(
        tool="hyperbolic_optics.grating_coupler", wavelengths_um=wavelengths,
        grating_layer=grating_layer, materials=materials, ambient=ambient,
        waveguide_port=waveguide_port, fiber_port=fiber_port, waist_um=waist_um,
        index_range=index_range, max_order=max_order, fiber_angle_deg=fiber_angle_deg,
    )  # fmt: skip
    path = get_sparameters_path(component, dirpath, **settings)
    if path.exists() and not overwrite:
        return _load(path)

    teeth = extract_grating_teeth(component, grating_layer)
    stack = teeth_stack(component, teeth, layer_stack, materials, ambient)
    eps_top = float(stack.half_space_permittivity("incident", wavelengths[0]))
    eps_bottom = float(stack.half_space_permittivity("exit", wavelengths[0]))
    guided = _guided_indices(stack, wavelengths[0])
    expected = float(np.mean(guided))  # the fundamental mode sees ridges and gaps in turn
    if index_range is None:
        index_range = (min(guided) - 0.1, max(guided) + 0.05)
    periods, fills = teeth.periods_um, teeth.fills

    te, tm, records = [], [], []
    window = tuple(index_range)
    for wavelength in wavelengths:
        cache: dict = {}

        def layers_for(i, wavelength=wavelength, cache=cache):
            key = (round(float(periods[i]), 6), round(float(fills[i]), 6))
            if key not in cache:
                grating = LineGrating(period_um=float(periods[i]), fill=float(fills[i]))
                cache[key] = grating_layers(grating, stack, wavelength)
            return cache[key]

        result = couplers.apodized_grating_coupler(
            layers_for, teeth.starts_um, periods, wavelength, eps_top, eps_bottom, window,
            waist_um, fiber_angle_deg, max_order, expected_index=expected,
        )  # fmt: skip
        te.append(np.sqrt(result.efficiency_te))
        tm.append(np.sqrt(result.efficiency_tm))
        first = result.modes[0]
        records.append(
            dict(
                wavelength_um=wavelength,
                efficiency=result.efficiency,
                efficiency_te=result.efficiency_te,
                efficiency_tm=result.efficiency_tm,
                fiber_angle_deg=result.fiber_angle_deg,
                fiber_offset_um=result.fiber_offset_um,
                first_tooth=dict(
                    bloch_index=first.bloch_index,
                    alpha_per_um=first.alpha_per_um,
                    directionality=first.directionality,
                    emission_deg=first.emission_deg.get(-1),
                ),
            )  # fmt: skip
        )
        expected = first.bloch_index.real  # follow it to the next wavelength
        window = (expected - 0.05, expected + 0.05)

    w, f = f"{waveguide_port}@TE", fiber_port
    te, tm = np.asarray(te, dtype=complex), np.asarray(tm, dtype=complex)
    sparameters = {
        f"{w},{f}@TE": te, f"{f}@TE,{w}": te,
        f"{w},{f}@TM": tm, f"{f}@TM,{w}": tm,
        "wavelengths": wavelengths,
    }  # fmt: skip
    metadata = dict(
        settings=settings,
        teeth=dict(starts_um=teeth.starts_um, periods_um=periods, fills=fills),
        stack=[(layer.name, layer.thickness_um) for layer in stack.layers],
        results=records,
    )  # fmt: skip
    _save(path, sparameters, metadata)
    return sparameters


# ---------------------------------------------------------------------------
# Planar stacks
# ---------------------------------------------------------------------------


def write_sparameters_stack(
    component: gf.Component,
    layer_stack: LayerStack,
    point_um: tuple[float, float],
    wavelengths_um,
    angle_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    materials: Mapping[str, MaterialSpec] | None = None,
    ambient: MaterialSpec = 1.0,
    dirpath: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """The plane-wave S-matrix of the PDK stack at a point of a layout, over wavelength.

    Returns ``{"top@p,bottom@s": …, …, "wavelengths": …}``: all 16 entries of
    :func:`hyperbolic_optics.bcm.stack_smatrix`, power-normalised, for light at
    ``angle_deg`` in the ambient (``kx`` fixed from it), the sample turned by
    ``azimuth_deg``.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    settings = dict(
        tool="hyperbolic_optics.stack", point_um=point_um, wavelengths_um=wavelengths,
        angle_deg=angle_deg, azimuth_deg=azimuth_deg, materials=materials, ambient=ambient,
    )  # fmt: skip
    path = get_sparameters_path(component, dirpath, **settings)
    if path.exists() and not overwrite:
        return _load(path)
    stack = vertical_stack(component, layer_stack, point_um, materials, ambient)
    eps_ambient = float(stack.half_space_permittivity("incident", wavelengths[0]))
    kx = np.sqrt(eps_ambient) * np.sin(np.radians(angle_deg))
    sd = stack_sax_model(stack)(wl=jnp.asarray(wavelengths), kx=kx, azimuth=np.radians(azimuth_deg))
    sparameters = {f"{a},{b}": np.asarray(v) for (a, b), v in sd.items()}
    sparameters["wavelengths"] = wavelengths
    metadata = dict(
        settings=settings, stack=[(layer.name, layer.thickness_um) for layer in stack.layers]
    )
    _save(path, sparameters, metadata)
    return sparameters
