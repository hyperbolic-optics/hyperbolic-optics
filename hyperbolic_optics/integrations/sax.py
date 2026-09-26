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

This module does not import SAX (it pins an older JAX than this package's);
the models are plain functions returning dictionaries, which SAX accepts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp

from hyperbolic_optics import bcm

__all__ = ["PORTS", "kx_from_angle", "planar_model", "sdict"]

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


def kx_from_angle(angle_rad, eps) -> jnp.ndarray:
    """``kx = n·sin θ`` for an angle ``θ`` in a medium of permittivity ``eps``."""
    return jnp.sqrt(jnp.asarray(eps)) * jnp.sin(jnp.asarray(angle_rad))


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
