"""Gaussian beams on planar stacks: Goos–Hänchen and Imbert–Fedorov shifts.

A finite beam is a superposition of plane waves. On a planar stack each keeps
its in-plane wavevector ``q``, so the beam's angular spectrum passes through
component by component, each with the plane-wave S-matrix of
:func:`hyperbolic_optics.bcm.stack_smatrix` at its own ``q``.
:func:`reflected_beam` rebuilds the reflected field on the interface plane, and
:func:`beam_shift` measures its centroid: the Goos–Hänchen shift along the
plane of incidence and the Imbert–Fedorov shift across it, which anisotropic
and hyperbolic media make large and asymmetric.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hyperbolic_optics import bcm

__all__ = ["beam_shift", "reflected_beam"]


class _BeamGrid(NamedTuple):
    """The angular-spectrum samples of a Gaussian beam.

    Attributes:
        qx, qy: In-plane wavevector over k₀, ``[K]``.
        weight: ``g(q)²·Δq``, normalised to sum to 1, ``[K]``.
        q0: The central wave's ``(qx, qy)``.
    """

    qx: jax.Array
    qy: jax.Array
    weight: jax.Array
    q0: tuple


def _beam_grid(
    wavelength_um, waist_um, angle_rad, eps_top, samples: int = 31, span: float = 4.0
) -> _BeamGrid:
    """Angular-spectrum samples of a Gaussian beam incident at ``angle_rad`` from the top.

    Args:
        wavelength_um: Vacuum wavelength.
        waist_um: The beam's waist ``w₀`` (field ``∝ exp(−r²/w₀²)`` in its own frame).
        angle_rad: Angle of incidence of the central wave in the top medium.
        eps_top: The top medium's permittivity.
        samples: Points along each of qx and qy (odd, so the centre is sampled).
        span: Half-width of the grid in units of the envelope's 1/e amplitude width.
    """
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    n = jnp.sqrt(jnp.asarray(eps_top, dtype=float))
    q0x = n * jnp.sin(angle_rad)
    width = 2.0 / (k0 * jnp.asarray(waist_um))  # 1/e half-width of |g| in q, beam frame
    u = jnp.linspace(-span, span, samples)
    dx, dy = jnp.meshgrid(u * width * jnp.cos(angle_rad), u * width, indexing="ij")
    envelope = jnp.exp(-(((dx / jnp.cos(angle_rad)) ** 2 + dy**2) / width**2))
    weight = (envelope**2).reshape(-1)
    return _BeamGrid(
        qx=(q0x + dx).reshape(-1), qy=dy.reshape(-1), weight=weight / weight.sum(),
        q0=(q0x, jnp.zeros(())),
    )  # fmt: skip


def _mode_vectors(eps, qx, qy, forward: bool):
    """E-field unit vectors of the p and s waves at in-plane ``q``, ``[K, 3]`` each.

    As in :func:`hyperbolic_optics.bcm._isotropic_modes`: s along ``ẑ × q̂`` and p
    completing the triad, forward (into the stack, +z) or backward.
    """
    n = jnp.sqrt(jnp.asarray(eps, dtype=float))
    q = jnp.sqrt(qx**2 + qy**2)
    moving = q > 1e-12
    ux = jnp.where(moving, qx / jnp.where(moving, q, 1.0), 1.0)
    uy = jnp.where(moving, qy / jnp.where(moving, q, 1.0), 0.0)
    kz = jnp.sqrt(jnp.asarray(eps, dtype=float) - q**2 + 0j)
    sign = 1.0 if forward else -1.0
    p = jnp.stack([sign * kz * ux / n, sign * kz * uy / n, -q / n + 0j], -1)
    s = jnp.stack([-uy + 0j, ux + 0j, jnp.zeros_like(ux) + 0j], -1)
    k_hat = jnp.stack([qx / n + 0j, qy / n + 0j, sign * kz / n], -1)
    return p, s, k_hat


def _jones(grid: _BeamGrid, eps, forward: bool):
    """Each component's local (p, s) amplitudes for the beam's central p and s: ``[K, 2, 2]``.

    ``[..., local, beam]``: the beam's polarization vector (the central wave's
    p or s), projected transverse to the component's direction and onto its
    own p and s, normalised.
    """
    p0, s0, _ = _mode_vectors(eps, jnp.atleast_1d(grid.q0[0]), jnp.atleast_1d(grid.q0[1]), forward)
    p, s, k_hat = _mode_vectors(eps, grid.qx, grid.qy, forward)
    columns = []
    for central in (p0[0], s0[0]):
        transverse = central[None, :] - jnp.sum(central[None, :] * k_hat, -1)[:, None] * k_hat
        local = jnp.stack(
            [jnp.sum(jnp.conj(p) * transverse, -1), jnp.sum(jnp.conj(s) * transverse, -1)], -1
        )
        columns.append(local / jnp.linalg.norm(local, axis=-1, keepdims=True))
    return jnp.stack(columns, -1)


def _component_smatrices(eps, thickness_um, wavelength_um, grid, eps_top, eps_bottom, mu):
    """The plane-wave S-matrix at every component's ``q``, in its own p/s: ``[K, 4, 4]``."""
    q = jnp.sqrt(grid.qx**2 + grid.qy**2)
    azimuth = jnp.arctan2(grid.qy, grid.qx)
    rotation = bcm.rotation_matrix(0.0, 0.0, -azimuth)  # [K, 3, 3]: the sample, turned
    eps = jnp.asarray(eps, dtype=complex)
    rotated = bcm.rotate_tensor(eps[None], rotation[:, None])
    rotated_mu = None if mu is None else bcm.rotate_tensor(jnp.asarray(mu)[None], rotation[:, None])
    return bcm.stack_smatrix(
        rotated, thickness_um, wavelength_um, q, eps_top, eps_bottom, mu=rotated_mu
    ).s


def reflected_beam(
    eps, thickness_um, wavelength_um, angle_rad, eps_top, eps_bottom, waist_um,
    polarization=(1.0, 0.0), mu=None, samples: int = 61, span: float = 5.0, points: int = 129,
    extent: float = 4.0,
):  # fmt: skip
    """The reflected beam's field on the reference plane, from its angular spectrum.

    Args:
        polarization: The incident beam's Jones vector ``(p, s)`` on the central wave.
        points, extent: The real-space grid: ``points`` per side over ``±extent``
            footprint widths.
        eps, thickness_um, mu: The layers, as in :func:`hyperbolic_optics.bcm.stack_smatrix`
            (lab frame; the plane of incidence of the central wave is x-z).
        angle_rad: Angle of incidence of the central wave in the top medium.
        eps_top, eps_bottom: The half-spaces' permittivities.
        waist_um: The beam waist ``w₀``.
        samples, span: The angular-spectrum grid (see :func:`_beam_grid`).

    Returns:
        ``(x, y, field)``: coordinates in µm on the interface plane, ``[P]`` each,
        and the reflected field's p and s amplitudes (in the reflected central
        wave's basis) ``[P, P, 2]``.
    """
    grid = _beam_grid(wavelength_um, waist_um, angle_rad, eps_top, samples, span)
    s = _component_smatrices(eps, thickness_um, wavelength_um, grid, eps_top, eps_bottom, mu)
    c_in = _jones(grid, eps_top, forward=True) @ jnp.asarray(polarization, dtype=complex)
    c_out = _jones(grid, eps_top, forward=False)
    local = s[:, :2, :2] @ c_in[..., None]
    amplitude = (jnp.conj(jnp.swapaxes(c_out, -1, -2)) @ local)[..., 0]  # [K, 2]
    spectrum = jnp.sqrt(grid.weight)[:, None] * amplitude
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    footprint = jnp.asarray(waist_um) / jnp.cos(angle_rad)
    x = jnp.linspace(-extent, extent, points) * footprint
    y = jnp.linspace(-extent, extent, points) * jnp.asarray(waist_um)
    dqx, dqy = grid.qx - grid.q0[0], grid.qy - grid.q0[1]
    phase = jnp.exp(
        1j * k0 * (dqx[:, None, None] * x[None, :, None] + dqy[:, None, None] * y[None, None, :])
    )
    field = jnp.einsum("kxy,kc->xyc", phase, spectrum)
    return x, y, field


def beam_shift(x, y, field) -> tuple[jax.Array, jax.Array]:
    """Centroid ``(Δx, Δy)`` of ``|field|²``: the Goos–Hänchen and Imbert–Fedorov shifts."""
    intensity = jnp.sum(jnp.abs(field) ** 2, -1)
    total = jnp.sum(intensity)
    return (
        jnp.sum(intensity * x[:, None]) / total,
        jnp.sum(intensity * y[None, :]) / total,
    )
