"""SAX models of planar anisotropic stacks.

`SAX <https://github.com/flaport/sax>`_ composes S-matrix models into circuits.
:func:`planar_model` turns a stack of anisotropic layers into one: a function
of the wavelength (``wl``, µm, SAX's convention), the in-plane wavevector
``kx`` and an azimuth, returning SAX's ``SDict`` over four multimode ports,

    ``top@p``, ``top@s``, ``bottom@p``, ``bottom@s``

keyed ``(in, out)`` as SAX expects. The values come from
:func:`hyperbolic_optics.bcm.stack_smatrix`, so they are differentiable, and
power-normalized between propagating ports (see
:class:`~hyperbolic_optics.bcm.PlanarSMatrix`).

Connecting one model's ``bottom`` to another's ``top`` in a SAX netlist stacks
them, provided the two share that half-space: its permittivity, and the same
``kx`` (pass ``kx`` as a global setting, not an angle, since the angle changes
from medium to medium and ``kx`` does not). Evanescent channels are carried, so
a gap beyond total internal reflection tunnels. ``bcm.cascade`` does the same
connection without SAX.

Plane-wave ports are not waveguide modes: connecting these to a guided-mode
model needs an overlap or mode-matching step that this module does not
provide.

For gradients through a circuit use ``sax.circuit(..., backend="filipsson_gunnar")``
(pure JAX, exact); SAX's default sparse (KLU) backend evaluates these models
but fails in the backward pass.

Payload users: :func:`structure_smatrix` / :func:`structure_sdict` take an
executed :class:`~hyperbolic_optics.structure.Structure` and give the same
four-port matrix on its scenario grid ``[angle, azimuth, frequency, thickness]``
-- every payload feature included (named materials and their rotations, custom
tensors, magnetic layers, thickness sweeps).

This module does not import SAX (it pins JAX below 0.10); the models are
plain functions returning dictionaries, which SAX accepts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp

from hyperbolic_optics import bcm

__all__ = [
    "PORTS",
    "planar_model",
    "sdict",
    "structure_sdict",
    "structure_smatrix",
]

PORTS = bcm.PORTS

Material = Any  # a number (isotropic ε), a 3×3 tensor, or a callable wavelength_um → either


def _tensor(material: Material, wavelength_um) -> jnp.ndarray:
    value = material(wavelength_um) if callable(material) else material
    value = jnp.asarray(value, dtype=complex)
    if value.ndim < 2 or value.shape[-2:] != (3, 3):
        value = value[..., None, None] * jnp.eye(3)
    return value


def _scalar(material: Material, wavelength_um) -> jnp.ndarray:
    value = material(wavelength_um) if callable(material) else material
    value = jnp.asarray(value)
    if value.ndim >= 2 and value.shape[-2:] == (3, 3):
        value = value[..., 0, 0]
    return jnp.real(value)


def sdict(smatrix: bcm.PlanarSMatrix) -> dict[tuple[str, str], jnp.ndarray]:
    """SAX's ``SDict`` for a :class:`~hyperbolic_optics.bcm.PlanarSMatrix`: ``(in, out)`` keys."""
    s = smatrix.s
    return {
        (port_in, port_out): s[..., i_out, i_in]
        for i_in, port_in in enumerate(PORTS)
        for i_out, port_out in enumerate(PORTS)
    }


def planar_model(
    layers: Sequence[tuple[Material, Any]],
    eps_top: Material,
    eps_bottom: Material,
    doublings: int = bcm.DEFAULT_DOUBLINGS,
) -> Callable[..., dict[tuple[str, str], jnp.ndarray]]:
    """A SAX model of a planar stack.

    Args:
        layers: ``(material, thickness_um)`` from the top down. A material is
            a number, a 3×3 tensor in the lab frame, or a callable
            ``wavelength_um → ε`` (for dispersion, e.g.
            ``lambda wl: dispersion.at_wavelength("hBN", wl)``).
        eps_top, eps_bottom: The half-spaces (real, positive, isotropic), as
            numbers or callables of the wavelength.
        doublings: Doublings per layer (see :data:`hyperbolic_optics.bcm.DEFAULT_DOUBLINGS`).

    Returns:
        ``model(wl=1.55, kx=0.0, azimuth=0.0) -> SDict``. ``azimuth`` (radians)
        turns every layer about the stack normal, as the payload API's
        ``rotationZ`` does; the plane of incidence stays x-z.
    """
    layers = list(layers)

    def model(wl=1.55, kx=0.0, azimuth=0.0):
        wl = jnp.asarray(wl, dtype=float)
        rotation = bcm.rotation_matrix(0.0, 0.0, jnp.asarray(azimuth, dtype=float))
        if layers:
            tensors = [bcm.rotate_tensor(_tensor(m, wl), rotation) for m, _ in layers]
            tensors = jnp.stack(jnp.broadcast_arrays(*tensors), axis=-3)
            thickness = jnp.stack(
                jnp.broadcast_arrays(*(jnp.asarray(d, dtype=float) for _, d in layers)), axis=-1
            )
        else:
            tensors = jnp.zeros((0, 3, 3), dtype=complex)
            thickness = jnp.zeros((0,))
        smatrix = bcm.stack_smatrix(
            tensors, thickness, wl, jnp.asarray(kx, dtype=float),
            _scalar(eps_top, wl), _scalar(eps_bottom, wl), doublings=doublings,
        )  # fmt: skip
        return sdict(smatrix)

    return model


def structure_smatrix(structure) -> bcm.PlanarSMatrix:
    """The four-port S-matrix of an executed payload, on its scenario grid.

    ``Structure.execute`` computes the response to light from the incident side
    only: 8 of the 16 entries. A SAX circuit also needs light arriving from
    below, so the whole matrix is built here from the executed structure's own
    layer tensors and thicknesses, with the JAX solver (which matches the NumPy
    engine to ~1e-14). Its incident-side entries are the engine's r and t,
    scaled by √(kz_out / kz_in).

    Args:
        structure: A :class:`~hyperbolic_optics.structure.Structure` after
            ``execute``, ending in a ``Semi Infinite Isotropic Layer``.

    Returns:
        :class:`~hyperbolic_optics.bcm.PlanarSMatrix` with ``s`` shaped
        ``[A, B, F, T, 4, 4]``: the canonical axes (incident angle, azimuth,
        frequency, swept thickness), each of size 1 where the scenario does not
        sweep it. The azimuth is already in the tensors, and ``kx`` comes from
        the incident angle in the prism.
    """
    from hyperbolic_optics.layers import IsotropicSemiInfiniteLayer

    if structure.k_0 is None:
        raise ValueError("execute the structure first")
    exit_layer = structure.layers[-1]
    if not isinstance(exit_layer, IsotropicSemiInfiniteLayer):
        raise NotImplementedError(
            f"the exit is a {type(exit_layer).__name__}; the S-matrix needs an isotropic, "
            "lossless exit half-space (a Semi Infinite Isotropic Layer)"
        )
    layers = structure.layers[1:-1]
    thickness_um = [jnp.asarray(layer.thickness) * 1e4 for layer in layers]  # cm -> µm
    grid = jnp.broadcast_shapes(
        jnp.shape(structure.k_x),
        jnp.shape(structure.k_0),
        *(jnp.shape(layer.eps_tensor)[:-2] for layer in layers),
        *(jnp.shape(layer.mu_tensor)[:-2] for layer in layers),
        *(jnp.shape(d) for d in thickness_um),
    )

    def stacked(tensors):
        if not tensors:
            return jnp.zeros(grid + (0, 3, 3), dtype=complex)
        return jnp.stack([jnp.broadcast_to(jnp.asarray(t), grid + (3, 3)) for t in tensors], -3)

    eps = stacked([layer.eps_tensor for layer in layers])
    mu = stacked([layer.mu_tensor for layer in layers])
    thickness = (
        jnp.stack([jnp.broadcast_to(d, grid) for d in thickness_um], -1)
        if layers
        else jnp.zeros(grid + (0,))
    )
    wavelength_um = jnp.broadcast_to(2 * jnp.pi / jnp.asarray(structure.k_0) * 1e4, grid)
    kx = jnp.broadcast_to(jnp.real(jnp.asarray(structure.k_x)), grid)
    return bcm.stack_smatrix(
        eps, thickness, wavelength_um, kx, float(structure.eps_prism),
        float(exit_layer.eps_exit), mu=mu,
    )  # fmt: skip


def structure_sdict(structure) -> dict[tuple[str, str], jnp.ndarray]:
    """SAX's ``SDict`` for an executed payload, values on its grid ``[A, B, F, T]``.

    SAX combines S-matrices elementwise, so these arrays compose with any other
    model evaluated on the same grid (same wavelengths, same ``kx``). Wrap it
    as a SAX model with ``lambda: structure_sdict(structure)``.
    """
    return sdict(structure_smatrix(structure))
