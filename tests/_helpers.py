"""Helpers shared by the JAX solver tests (import after ``pytest.importorskip("jax")``)."""

import jax.numpy as jnp
import numpy as np


def uniaxial(eps_o, eps_e, tilt, azimuth):
    """Uniaxial tensor with its axis at ``tilt`` out of the interface, ``azimuth`` in it."""
    axis = jnp.array(
        [jnp.cos(tilt) * jnp.cos(azimuth), jnp.cos(tilt) * jnp.sin(azimuth), jnp.sin(tilt)]
    )
    return (eps_o * jnp.eye(3) + (eps_e - eps_o) * jnp.outer(axis, axis)).astype(complex)


def total(result):
    """Summed efficiency over orders and outgoing polarizations, per input."""
    return np.asarray(result.R.sum(axis=(-3, -2)) + result.T.sum(axis=(-3, -2)))
