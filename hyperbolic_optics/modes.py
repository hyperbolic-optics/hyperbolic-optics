"""Guided modes of planar stacks of anisotropic layers.

Full-vector: any 3×3 ε and μ in any layer, so the modes of a rotated crystal
film are hybrid (neither TE nor TM), and absorbing layers give complex
effective indices.

A mode is a field that exists with nothing incident. At a plane inside the
stack, let ``A`` be the scattering matrix of everything above it and ``B`` of
everything below. Then an eigenvalue of ``A₀₁·B₁₀`` equals 1 at a mode. The
claddings' waves are continued analytically to their outgoing branch, so the
round trip is analytic in ``n_eff``. The roots are found by Beyn's contour
method and refined by the secant method.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import bcm
from hyperbolic_optics.bcm import _interface, _layer, _longitudinal, _star

__all__ = ["GuidedMode", "slab_mode_fields", "slab_modes"]


class GuidedMode(NamedTuple):
    """A guided (or lossy, or leaky) mode of a planar stack, propagating along x.

    Attributes:
        effective_index: Complex ``n_eff = β/k₀``; ``Im > 0`` for a mode that
            decays as it propagates (absorption, or leakage).
        loss_db_per_cm: Power attenuation, ``20·log₁₀(e)·k₀·Im(n_eff)`` per cm.
        multiplicity: How many eigenvalues of the round trip reach 1 (2 for a
            degenerate pair).
        residual: ``|1 − μ|`` at the root.
    """

    effective_index: complex
    loss_db_per_cm: float
    multiplicity: int
    residual: float


def _planar_parts(eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, kx):
    """Analytic bases and each layer's operator at complex ``kx`` (real frequency)."""
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    eps = jnp.asarray(eps, dtype=complex)
    mu = jnp.broadcast_to(jnp.eye(3, dtype=complex), eps.shape) if mu is None else jnp.asarray(mu)
    eps_ref = 2 * kx**2 + 1.0
    w_ref = bcm._isotropic_modes(eps_ref, kx, jnp.sqrt(1.0 + kx**2 + 0j))
    w_ref = w_ref * ((1.0 + kx**2 + 0j) ** -0.25)[..., None, None]

    def half_space(eps_half):
        eps_half = jnp.asarray(eps_half)
        return bcm._isotropic_modes(eps_half, kx, bcm._kz_outgoing(eps_half, kx, k0))

    deltas = [
        bcm._solve(w_ref, bcm.system_matrix(eps[..., i, :, :], mu[..., i, :, :], kx) @ w_ref)
        for i in range(eps.shape[-3])
    ]
    slices = [k0 * jnp.asarray(thickness_um)[..., i] for i in range(eps.shape[-3])]
    return w_ref, half_space(eps_top), half_space(eps_bottom), deltas, slices, mu, k0


def _planar_round_trip(eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, split, doublings):
    """``n_eff ↦ A₀₁·B₁₀`` (2 × 2) at a plane inside a planar stack, compiled."""
    index, fraction = split

    def matrix(n_eff):
        w_ref, w_top, w_bottom, deltas, k0d, _, _ = _planar_parts(
            eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, n_eff
        )
        above = _interface(w_top, w_ref)
        for i in range(index):
            above = _star(above, _layer(deltas[i], k0d[i], doublings))
        above = _star(above, _layer(deltas[index], k0d[index] * fraction, doublings))
        below = _layer(deltas[index], k0d[index] * (1 - fraction), doublings)
        for i in range(index + 1, len(deltas)):
            below = _star(below, _layer(deltas[i], k0d[i], doublings))
        below = _star(below, _interface(w_ref, w_bottom))
        return above[1] @ below[2]

    return jax.jit(matrix)


def _index_bounds(eps, eps_top, eps_bottom):
    cladding = max(float(np.sqrt(np.real(eps_top))), float(np.sqrt(np.real(eps_bottom))))
    core = max(
        float(np.max(np.sqrt(np.real(np.linalg.eigvals(np.asarray(e)))))) for e in np.asarray(eps)
    )
    return cladding, core


def slab_modes(
    eps,
    thickness_um,
    wavelength_um,
    eps_top,
    eps_bottom,
    index_range: tuple[float, float] | None = None,
    max_loss: float | None = None,
    mu=None,
    split: tuple[int, float] | None = None,
    quadrature: int = 64,
    probes: int = 8,
    tolerance: float = 1e-12,
    doublings: int = 20,
    seed: int = 0,
) -> list[GuidedMode]:
    """Guided modes of a planar stack of anisotropic layers, propagating along x.

    Full-vector: any 3×3 ε and μ in any layer, so the modes of a rotated
    crystal film are hybrid (neither TE nor TM), and absorbing layers give
    complex effective indices. Rotate the tensors to change the propagation
    direction in the plane.

    Args:
        eps: Layer tensors ``[L, 3, 3]`` from the top down (lab frame, x the
            propagation direction).
        thickness_um: ``[L]``.
        wavelength_um: Vacuum wavelength (real).
        eps_top, eps_bottom: Cladding permittivities (isotropic, real).
        index_range: Real ``n_eff`` to search; by default from just above the
            higher cladding index to the highest index in the core. Keep it
            above the claddings' indices: the branch point at each is where a
            mode stops being guided.
        max_loss: Largest ``Im n_eff`` searched; by default 5 % of the range.
        mu: Layer permeabilities; identity if omitted.
        split: ``(layer, fraction)`` of the internal plane; by default the
            middle of the highest-index layer.
        quadrature, probes: Contour quadrature points, and random probe
            vectors (at least the number of modes in the range).
        tolerance: ``|1 − μ|`` at which a root is accepted.
        doublings: Doublings per layer, as in :func:`hyperbolic_optics.bcm.solve_stack`.
        seed: Seed of the probe vectors.

    Returns:
        :class:`GuidedMode` s, highest effective index first (fundamental first).
    """
    eps = jnp.asarray(eps, dtype=complex)
    thickness_um = jnp.asarray(thickness_um, dtype=float)
    cladding, core = _index_bounds(eps, eps_top, eps_bottom)
    low, high = index_range if index_range is not None else (cladding * (1 + 1e-6), core)
    depth = 0.05 * (high - low) if max_loss is None else float(max_loss)
    if split is None:
        peak = [float(np.max(np.real(np.linalg.eigvals(np.asarray(e))))) for e in eps]
        split = (int(np.argmax(peak)), 0.5)
    round_trip = _planar_round_trip(
        eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, split, doublings
    )
    roots = _contour_roots(round_trip, (low, high), depth, quadrature, probes, tolerance, seed)
    k0 = 2 * np.pi / float(wavelength_um)
    return [
        GuidedMode(
            effective_index=n,
            loss_db_per_cm=float(20 * np.log10(np.e) * k0 * n.imag * 1e4),
            multiplicity=multiplicity,
            residual=residual,
        )
        for n, residual, multiplicity in sorted(roots, key=lambda r: -r[0].real)
    ]


def slab_mode_fields(
    eps,
    thickness_um,
    wavelength_um,
    eps_top,
    eps_bottom,
    effective_index,
    mu=None,
    samples_per_layer: int = 50,
    cladding_um: tuple[float, float] = (1.0, 1.0),
    cladding_samples: int = 50,
    split: tuple[int, float] | None = None,
    doublings: int = 20,
):
    """The field profile of a guided mode, normalised to unit power along x.

    Args:
        effective_index: A root from :func:`slab_modes`.
        samples_per_layer: Samples through each layer, both faces included.
        cladding_um: How far into the top and bottom claddings to sample.
        cladding_samples: Samples in each cladding.
        Others: As in :func:`slab_modes`.

    Returns:
        ``(z_um, fields)``: depth below the top of the first layer ``[Z]`` and
        ``(Ex, Ey, Ez, Hx, Hy, Hz)`` there ``[Z, 6]`` (H scaled by the free-space
        impedance), with ``½ Re ∫ (Ey·Hz* − Ez·Hy*) dz = 1``: unit power along x.

    The fields are marched from the internal plane with bounded scattering
    matrices, so thick claddings or strongly evanescent layers stay stable.
    """
    eps = jnp.asarray(eps, dtype=complex)
    thickness_um = jnp.asarray(thickness_um, dtype=float)
    n_eff = jnp.asarray(effective_index, dtype=complex)
    w_ref, w_top, w_bottom, deltas, k0d, mu_full, k0 = _planar_parts(
        eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, n_eff
    )
    count = len(deltas)
    fractions = np.linspace(0.0, 1.0, samples_per_layer)
    step = [_layer(deltas[i], k0d[i] / (samples_per_layer - 1), doublings) for i in range(count)]

    # planes: every sample of every layer, top down; slices between consecutive ones
    planes = [(i, j) for i in range(count) for j in range(samples_per_layer)]
    identity = (jnp.eye(2, dtype=complex), jnp.zeros((2, 2), complex),
                jnp.zeros((2, 2), complex), jnp.eye(2, dtype=complex))  # fmt: skip

    def between(a, b):  # the slice from plane a to plane b = a + 1
        return identity if planes[b][0] != planes[a][0] else step[planes[a][0]]

    above = [_interface(w_top, w_ref)]
    for k in range(1, len(planes)):
        above.append(_star(above[-1], between(k - 1, k)))
    below = [_interface(w_ref, w_bottom)]
    for k in range(len(planes) - 2, -1, -1):
        below.insert(0, _star(between(k, k + 1), below[0]))

    if split is None:
        peak = [float(np.max(np.real(np.linalg.eigvals(np.asarray(e))))) for e in eps]
        split = (int(np.argmax(peak)), 0.5)
    s = split[0] * samples_per_layer + int(round(split[1] * (samples_per_layer - 1)))
    mu_values, vectors = np.linalg.eig(np.asarray(above[s][1] @ below[s][2]))
    down = {s: jnp.asarray(vectors[:, np.argmin(np.abs(1 - mu_values))])}
    up = {s: below[s][2] @ down[s]}
    eye = jnp.eye(2, dtype=complex)
    for k in range(s + 1, len(planes)):  # march down: f_k = (I − L01·B10)⁻¹ L00 f_{k−1}
        l00, l01, _, _ = between(k - 1, k)
        down[k] = jnp.linalg.solve(eye - l01 @ below[k][2], l00 @ down[k - 1])
        up[k] = below[k][2] @ down[k]
    for k in range(s - 1, -1, -1):  # march up: b_k = (I − L10·A01)⁻¹ L11 b_{k+1}
        _, _, l10, l11 = between(k, k + 1)
        up[k] = jnp.linalg.solve(eye - l10 @ above[k][1], l11 @ up[k + 1])
        down[k] = above[k][1] @ up[k]

    tops = np.concatenate([[0.0], np.cumsum(np.asarray(thickness_um))])
    z_list, field_list = [], []

    def full(tangential, eps_here, mu_here):
        a, b = _longitudinal(eps_here, mu_here, n_eff)
        ez, hz = a @ tangential, b @ tangential
        return jnp.stack([tangential[0], tangential[1], ez, tangential[2], tangential[3], hz])

    # the top cladding: outgoing (upward, decaying) waves only
    eye3 = jnp.eye(3, dtype=complex)
    top_amplitudes = jnp.linalg.solve(w_top, w_ref @ jnp.concatenate([down[0], up[0]]))
    kz_top = bcm._kz_outgoing(jnp.asarray(eps_top), n_eff, k0)
    for z in np.linspace(-cladding_um[0], 0.0, cladding_samples, endpoint=False):
        tangential = w_top[:, 2:] @ (top_amplitudes[2:] * jnp.exp(-1j * k0 * kz_top * z))
        z_list.append(z)
        field_list.append(full(tangential, jnp.asarray(eps_top) * eye3, eye3))
    for k, (i, j) in enumerate(planes):
        tangential = w_ref @ jnp.concatenate([down[k], up[k]])
        z_list.append(tops[i] + fractions[j] * float(thickness_um[i]))
        field_list.append(full(tangential, eps[i], mu_full[i]))
    bottom_amplitudes = jnp.linalg.solve(
        w_bottom, w_ref @ jnp.concatenate([down[len(planes) - 1], up[len(planes) - 1]])
    )
    kz_bottom = bcm._kz_outgoing(jnp.asarray(eps_bottom), n_eff, k0)
    for z in np.linspace(0.0, cladding_um[1], cladding_samples + 1)[1:]:
        tangential = w_bottom[:, :2] @ (bottom_amplitudes[:2] * jnp.exp(1j * k0 * kz_bottom * z))
        z_list.append(tops[-1] + z)
        field_list.append(full(tangential, jnp.asarray(eps_bottom) * eye3, eye3))

    z_um = np.asarray(z_list)
    fields = np.asarray(jnp.stack(field_list))
    flux = 0.5 * np.real(
        fields[:, 1] * np.conj(fields[:, 5]) - fields[:, 2] * np.conj(fields[:, 4])
    )
    order = np.argsort(z_um, kind="stable")
    power = np.trapezoid(flux[order], z_um[order])
    return z_um, fields / np.sqrt(power)


def _contour_roots(round_trip, real_range, depth, quadrature, probes, tolerance, seed):
    """Every ``z`` near a real range where ``A₀₁·B₁₀(z)`` has eigenvalue 1.

    The range is tiled by ellipses centred on the real axis, each reaching
    ``±depth`` across it and overlapping its neighbours, with aspect ratio at
    most 2 (the trapezoid rule converges slowly on a long, flat contour). In
    each, Beyn's method estimates the roots, which the secant method refines.
    Returns ``[(z, residual, multiplicity)]`` in order of ``Re z``, degenerate
    roots once.
    """
    x0, x1 = real_range
    width = x1 - x0
    tiles = max(1, int(np.ceil(width / (4 * depth))))
    segment = width / tiles
    found = []
    for tile in range(tiles):
        # neighbours overlap by 10 %, but no tile reaches past the range's ends:
        # an end may be a branch point (a cladding's index, for guided modes)
        left = max(x0, x0 + (tile - 0.1) * segment)
        right = min(x1, x0 + (tile + 1.1) * segment)
        centre = complex(0.5 * (left + right), 0.0)
        found += _beyn(round_trip, centre, 0.5 * (right - left), depth, quadrature, probes,
                       tolerance, seed + tile, width)  # fmt: skip

    roots = []
    for z, residual, mu in sorted(found, key=lambda r: r[0].real):
        if not x0 <= z.real <= x1:
            continue
        if roots and abs(z - roots[-1][0]) < 1e-8 * max(1.0, abs(z)):
            continue  # found twice (overlapping tiles), or a degenerate partner
        multiplicity = int(np.sum(np.abs(1 - mu) < max(1e-7, 1e3 * residual)))
        roots.append((z, residual, multiplicity))
    return roots


def _beyn(round_trip, centre, semi_real, semi_imag, quadrature, probes, tolerance, seed, scale):
    """Beyn's method in one ellipse, then secant refinement: ``[(z, residual, μ)]``."""
    theta = 2 * np.pi * (np.arange(quadrature) + 0.5) / quadrature
    points = centre + semi_real * np.cos(theta) + 1j * semi_imag * np.sin(theta)
    tangents = -semi_real * np.sin(theta) + 1j * semi_imag * np.cos(theta)  # dz/dθ

    size = np.asarray(round_trip(points[0])).shape[-1]
    probes = min(probes, size)
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((size, probes)) + 1j * rng.standard_normal((size, probes))
    moment0 = np.zeros((size, probes), dtype=complex)
    moment1 = np.zeros((size, probes), dtype=complex)
    for z, dz in zip(points, tangents, strict=True):
        t = np.eye(size) - np.asarray(round_trip(z))
        solved = np.linalg.solve(t, probe) * dz / quadrature / 1j
        moment0 += solved
        moment1 += z * solved
    u, sigma, vh = np.linalg.svd(moment0, full_matrices=False)
    rank = int(np.sum(sigma > 1e-9 * sigma[0])) if sigma[0] > 0 else 0
    if rank == 0:
        return []
    if rank == probes and probes < size:
        warnings.warn(
            f"{rank} roots fill all {probes} probes: some may be missing. Raise probes, "
            "or narrow the region.",
            UserWarning,
            stacklevel=4,
        )
    u, sigma, v = u[:, :rank], sigma[:rank], vh[:rank].conj().T
    estimates = np.linalg.eigvals(u.conj().T @ moment1 @ v / sigma)

    def mismatch(z):
        matrix = np.asarray(round_trip(z))
        if not np.all(np.isfinite(matrix)):
            return None, None
        mu = np.linalg.eigvals(matrix)
        return 1 - mu[np.argmin(np.abs(1 - mu))], mu

    reach = max(semi_real, semi_imag)
    found = []
    for estimate in estimates:
        z_prev, z = complex(estimate), complex(estimate) + 1e-6 * scale
        g_prev, _ = mismatch(z_prev)
        if g_prev is None:
            continue
        for _ in range(30):
            g, _ = mismatch(z)
            if g is None or abs(g) < tolerance or g == g_prev:
                break
            z, z_prev, g_prev = z - g * (z - z_prev) / (g - g_prev), z, g
            if abs(z - centre) > 2 * reach:  # wandered off: a poor estimate
                break
        if abs(z - centre) > 2 * reach:
            continue
        g, mu = mismatch(z)
        if g is None:
            continue
        inside = ((z.real - centre.real) / semi_real) ** 2 + (z.imag / semi_imag) ** 2 <= 1 + 1e-9
        if abs(g) < 1e3 * tolerance and inside:
            found.append((z, abs(g), mu))
    return found
