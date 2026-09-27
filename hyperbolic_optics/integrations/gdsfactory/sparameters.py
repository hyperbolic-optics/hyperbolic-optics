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
file beside the ``.npz`` records the settings.

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

from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec
from hyperbolic_optics.integrations.gdsfactory.simulate import stack_sax_model
from hyperbolic_optics.integrations.gdsfactory.stack import (
    vertical_stack,
)

__all__ = [
    "component_hash",
    "get_sparameters_path",
    "model_from_npz",
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
