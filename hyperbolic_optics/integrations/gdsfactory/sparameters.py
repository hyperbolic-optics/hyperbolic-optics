"""Cached four-port S-parameters for PDK stacks and metasurface cells.

Writers save ``.npz`` under ``gf.config.PATH.sparameters``, keyed by geometry,
evaluated material tensors and solver settings, with JSON metadata alongside.
Keys are ``"port@mode,port@mode"`` (input first) and ``"wavelengths"`` (µm).
Use :func:`model_from_npz` to preserve polarization modes when loading into SAX.
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

from hyperbolic_optics import __version__
from hyperbolic_optics.integrations.gdsfactory.materials import MaterialSpec, material_tensor
from hyperbolic_optics.integrations.gdsfactory.metasurface import cell_sax_model
from hyperbolic_optics.integrations.gdsfactory.simulate import stack_sax_model
from hyperbolic_optics.integrations.gdsfactory.stack import (
    VerticalStack,
    vertical_stack,
)

__all__ = [
    "component_hash",
    "get_sparameters_path",
    "model_from_npz",
    "write_sparameters_stack",
    "write_sparameters_cell",
]


def component_hash(component: gf.Component) -> str:
    """A hash of a component's geometry: every polygon on every layer, in dbu."""
    digest = hashlib.md5()
    dbu = component.kcl.dbu
    digest.update(str(dbu).encode())
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


def _stack_settings(stack, wavelengths):
    """Cache the evaluated physics, including callable materials and crystal cuts."""
    return dict(
        version=__version__,
        schema=2,
        incident=stack.half_space_permittivity("incident", wavelengths),
        exit=stack.half_space_permittivity("exit", wavelengths),
        layers=[
            dict(
                thickness_um=layer.thickness_um,
                eps=material_tensor(layer.material, wavelengths, layer.rotation_deg),
                background=None
                if layer.background is None
                else material_tensor(layer.background, wavelengths, layer.rotation_deg),
                polygons=layer.polygons,
            )
            for layer in stack.layers
        ],
    )


def _cached_model(path, settings, model, wavelengths, kx, azimuth, overwrite):
    if path.exists() and not overwrite:
        return _load(path)
    sd = model(wl=jnp.asarray(wavelengths), kx=kx, azimuth=azimuth)
    data = {f"{a},{b}": np.asarray(v) for (a, b), v in sd.items()}
    data["wavelengths"] = wavelengths
    _save(path, data, settings)
    return data


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
    ``angle_deg`` in the ambient (including its dispersion), the sample turned by
    ``azimuth_deg``.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    stack = vertical_stack(component, layer_stack, point_um, materials, ambient)
    settings = dict(
        tool="hyperbolic_optics.stack", point_um=point_um, wavelengths_um=wavelengths,
        angle_deg=angle_deg, azimuth_deg=azimuth_deg, stack=_stack_settings(stack, wavelengths),
    )  # fmt: skip
    path = get_sparameters_path(component, dirpath, **settings)
    eps_ambient = settings["stack"]["incident"]
    kx = np.sqrt(eps_ambient) * np.sin(np.radians(angle_deg))
    return _cached_model(
        path, settings, stack_sax_model(stack), wavelengths, kx, np.radians(azimuth_deg), overwrite
    )


def write_sparameters_cell(
    cell: VerticalStack,
    lattice_um,
    wavelengths_um,
    max_order: tuple[int, int] = (4, 4),
    kx=0.0,
    azimuth=0.0,
    dirpath: Path | str | None = None,
    overwrite: bool = False,
    **solver_settings,
) -> dict[str, np.ndarray]:
    """Cache a cell's four-port S-parameters; angles and solver settings follow cell_sax_model.

    The key includes evaluated tensors, thicknesses, polygons, lattice and
    truncation. Named or callable material changes therefore invalidate it.
    Read the returned dictionary with :func:`model_from_npz`.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
    settings = dict(
        tool="hyperbolic_optics.cell",
        wavelengths_um=wavelengths,
        stack=_stack_settings(cell, wavelengths),
        lattice_um=lattice_um,
        max_order=max_order,
        kx=kx,
        azimuth=azimuth,
        solver_settings=solver_settings,
    )
    directory = Path(dirpath if dirpath is not None else gf.config.PATH.sparameters)
    directory = directory / "hyperbolic_optics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cell_{_settings_hash(settings)}.npz"
    return _cached_model(
        path,
        settings,
        cell_sax_model(cell, lattice_um, max_order, **solver_settings),
        wavelengths,
        kx,
        azimuth,
        overwrite,
    )
