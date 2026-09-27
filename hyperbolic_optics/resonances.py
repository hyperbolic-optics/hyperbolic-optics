"""Resonances of layered and periodic structures: band-edge modes and their Q.

A resonance is a field that exists with nothing incident: a pole of the
scattering matrix. At a plane inside the structure, let ``A`` be the scattering
matrix of everything above it and ``B`` of everything below. The downward and
upward amplitudes there satisfy ``f = A₀₁·b`` and ``b = B₁₀·f``, so at a
resonance an eigenvalue ``μ`` of ``A₀₁·B₁₀`` equals 1. The half-spaces' waves
are continued analytically to their outgoing branch, so leaky resonances --
complex frequency, radiating into the open diffraction channels -- are found as
well as bound ones.

:func:`band_edge_modes` finds the band-edge modes of a photonic-crystal slab at
a Bloch wavevector (Γ by default): the modes a photonic-crystal surface-emitting
laser (PCSEL) lases on, which coupled-wave theory computes for square and
triangular lattices (M. N. Robinson, *Developing coupled wave theory for
photonic crystal surface emitting lasers*, PhD thesis, Queen's University
Belfast, 2026). Here they come from the full-vector RCWA of
:mod:`hyperbolic_optics.rcwa2d`, on any lattice and with any tensor in any
layer. Each has a complex normalised frequency ``a/λ`` (``a = |a₁|``), its
quality factor ``Q = Re f / (2 |Im f|)``, and its multiplicity (degenerate
pairs appear together at Γ on symmetric lattices).

:func:`spectrum_poles` extracts poles from any sampled response (a reflection
spectrum, from this package or another code) by AAA rational approximation
(Nakatsukasa, Sète & Trefethen, SIAM J. Sci. Comput. 40, A1494 (2018)): an
independent check on the resonances found here, from real-frequency data only.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg  # a dependency of JAX, which this module needs

from hyperbolic_optics import bcm, rcwa2d
from hyperbolic_optics.bcm import _interface, _layer, _longitudinal, _star
from hyperbolic_optics.rcwa import default_doublings

__all__ = [
    "GuidedMode",
    "Resonance",
    "band_edge_modes",
    "crystal_eigenvalues",
    "slab_mode_fields",
    "slab_modes",
    "spectrum_poles",
]


class Resonance(NamedTuple):
    """One resonance.

    Attributes:
        frequency: Complex normalised frequency ``a/λ``, ``a = |a₁|``; the
            imaginary part is negative (the field decays in time, ``exp(−iωt)``).
        wavelength_um: The complex vacuum wavelength ``a / frequency``.
        q_factor: ``Re f / (2 |Im f|)``; ``inf`` for a bound mode.
        multiplicity: How many eigenvalues of the round trip reach 1 there:
            2 for a degenerate pair.
        residual: ``|1 − μ|`` at the root.
    """

    frequency: complex
    wavelength_um: complex
    q_factor: float
    multiplicity: int
    residual: float


def _default_split(layers) -> tuple[int, float]:
    """The middle of the first patterned layer (else of the thickest)."""
    for index, layer in enumerate(layers):
        if layer.shapes:
            return index, 0.5
    thickness = [float(np.real(np.asarray(layer.thickness_um))) for layer in layers]
    return int(np.argmax(thickness)), 0.5


def _round_trip_matrix(
    layers: Sequence[rcwa2d.PatternedLayer],
    lattice,
    frequency,
    eps_top,
    eps_bottom,
    max_order: tuple[int, int],
    bloch=(0.0, 0.0),
    split: tuple[int, float] | None = None,
    factorization: str = "normal",
    truncation: str = "parallelogram",
    doublings: int | None = None,
    grid: tuple[int, int] | None = None,
) -> jax.Array:
    """The round-trip matrix ``A₀₁·B₁₀`` at a plane inside a crystal slab.

    Args:
        layers: The stack, from the top down; at least one layer.
        lattice: A :class:`~hyperbolic_optics.rcwa2d.Lattice` (or ``(Λx, Λy)``).
        frequency: Normalised frequency ``a/λ``, ``a = |a₁|``; may be complex.
        eps_top, eps_bottom: The half-spaces' permittivities (isotropic).
        max_order: Harmonics, as in :func:`~hyperbolic_optics.rcwa2d.solve_crossed_grating`.
        bloch: In-plane Bloch wavevector in units of ``2π/a``; ``(0, 0)`` is Γ.
        split: ``(layer index, fraction of its thickness)`` for the plane;
            by default the middle of the first patterned layer.
        factorization, truncation, doublings, grid: As in
            :func:`~hyperbolic_optics.rcwa2d.solve_crossed_grating`.

    Returns:
        ``[2N, 2N]``: ``A₀₁·B₁₀``.
    """
    lattice = rcwa2d._as_lattice(lattice)
    a1, _, b1, b2, _ = rcwa2d._vectors(lattice)
    a = jnp.linalg.norm(a1)
    orders = rcwa2d._orders(*max_order, lattice, truncation)
    mx, my = int(np.abs(orders[:, 0]).max()), int(np.abs(orders[:, 1]).max())
    if doublings is None:
        doublings = default_doublings(max(mx, my))
    if grid is None:
        grid = (max(32, 8 * mx + 8), max(32, 8 * my + 8))
    k0 = 2 * jnp.pi * jnp.asarray(frequency) / a  # complex in general
    bloch = 2 * jnp.pi / a * jnp.asarray(bloch, dtype=float)
    gx = orders[:, 0] * b1[0] + orders[:, 1] * b2[0]
    gy = orders[:, 0] * b1[1] + orders[:, 1] * b2[1]
    kx, ky = (bloch[0] + gx) / k0, (bloch[1] + gy) / k0
    direction = (jnp.ones(()), jnp.zeros(()))
    w_ref, w_in, _, w_out, _, deltas = rcwa2d._operators(
        list(layers), lattice, orders, (mx, my), factorization, grid, kx, ky, direction,
        eps_top, eps_bottom, k0=k0,
    )  # fmt: skip

    index, fraction = split if split is not None else _default_split(layers)
    above = _interface(w_in, w_ref)
    for i, (layer, delta) in enumerate(zip(layers, deltas, strict=True)):
        k0d = k0 * jnp.asarray(layer.thickness_um)
        if i < index:
            above = _star(above, _layer(delta, k0d, doublings))
        elif i == index:
            above = _star(above, _layer(delta, k0d * fraction, doublings))
            below = _layer(delta, k0d * (1 - fraction), doublings)
        else:
            below = _star(below, _layer(delta, k0d, doublings))
    below = _star(below, _interface(w_ref, w_out))
    return above[1] @ below[2]


def _round_trip(layers, lattice, eps_top, eps_bottom, max_order, bloch, split, factorization,
                truncation, grid):  # fmt: skip
    """``f ↦ A₀₁·B₁₀`` at the split plane, one frequency at a time (compiled once)."""

    def matrix(frequency):
        return _round_trip_matrix(
            layers, lattice, frequency, eps_top, eps_bottom, max_order, bloch=bloch,
            split=split, factorization=factorization, truncation=truncation, grid=grid,
        )  # fmt: skip

    return jax.jit(matrix)


def band_edge_modes(
    layers: Sequence[rcwa2d.PatternedLayer],
    lattice,
    frequency_range: tuple[float, float],
    eps_top,
    eps_bottom,
    max_order: tuple[int, int],
    bloch=(0.0, 0.0),
    max_loss: float | None = None,
    split: tuple[int, float] | None = None,
    factorization: str = "normal",
    truncation: str = "parallelogram",
    grid: tuple[int, int] | None = None,
    quadrature: int = 64,
    probes: int = 16,
    tolerance: float = 1e-11,
    seed: int = 0,
) -> list[Resonance]:
    """Band-edge (or any) modes of a photonic-crystal slab, with their Q factors.

    Every resonance inside a region of the complex frequency plane is found at
    once by Beyn's contour-integral method (W.-J. Beyn, Linear Algebra Appl.
    436, 3839 (2012)) applied to ``T(f) = I − A₀₁·B₁₀``: the region is an ellipse
    spanning ``frequency_range`` on the real axis and reaching ``max_loss``
    below it, so bound modes (real ``f``) and leaky ones (``Im f < 0``) are found
    together, degenerate pairs as two. (The ellipse is symmetric about the real
    axis; the upper half holds no resonances of a passive structure.) Each is
    then refined by the secant method on its eigenvalue of ``A₀₁·B₁₀``.

    Args:
        layers, lattice, eps_top, eps_bottom, max_order, bloch, split,
        factorization, truncation, grid: As in :func:`crystal_eigenvalues`.
        frequency_range: ``(f_min, f_max)``, real normalised frequencies ``a/λ``.
        max_loss: How far below the real axis to look (``|Im f|``); by default
            a quarter of the range's width (``Q`` down to about
            ``2·f/(range width)``).
        quadrature: Points on the contour; raise it if the region is crowded.
        probes: Random probe vectors: at least the number of resonances
            expected inside (the result warns if they were too few).
        tolerance: Convergence of ``|1 − μ|`` when refining.
        seed: For the probe vectors.

    Returns:
        :class:`Resonance` s inside the region, in order of frequency.
    """
    f_min, f_max = (float(f) for f in frequency_range)
    width = f_max - f_min
    depth = width / 4 if max_loss is None else float(max_loss)
    a = float(np.linalg.norm(np.asarray(rcwa2d._as_lattice(lattice).a1, dtype=float)))
    round_trip = _round_trip(
        layers, lattice, eps_top, eps_bottom, max_order, bloch, split, factorization,
        truncation, grid,
    )  # fmt: skip
    roots = _contour_roots(round_trip, (f_min, f_max), depth, quadrature, probes, tolerance, seed)
    return [
        Resonance(
            frequency=f,
            wavelength_um=a / f,
            q_factor=float(f.real / (2 * abs(f.imag))) if f.imag != 0 else float("inf"),
            multiplicity=multiplicity,
            residual=residual,
        )
        for f, residual, multiplicity in roots
    ]


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


def crystal_eigenvalues(
    layers: Sequence[rcwa2d.PatternedLayer],
    lattice,
    frequency,
    eps_top,
    eps_bottom,
    max_order: tuple[int, int],
    bloch=(0.0, 0.0),
    split: tuple[int, float] | None = None,
    factorization: str = "normal",
    truncation: str = "parallelogram",
    doublings: int | None = None,
    grid: tuple[int, int] | None = None,
) -> jax.Array:
    """Eigenvalues of the round trip ``A₀₁·B₁₀`` at a plane inside a crystal slab.

    Args:
        layers: The stack, from the top down; at least one layer.
        lattice: A :class:`~hyperbolic_optics.rcwa2d.Lattice` (or ``(Λx, Λy)``).
        frequency: Normalised frequency ``a/λ``, ``a = |a₁|``; may be complex.
        eps_top, eps_bottom: The half-spaces' permittivities (isotropic).
        max_order: Harmonics, as in :func:`~hyperbolic_optics.rcwa2d.solve_crossed_grating`.
        bloch: In-plane Bloch wavevector in units of ``2π/a``; ``(0, 0)`` is Γ.
        split: ``(layer index, fraction of its thickness)`` for the plane;
            by default the middle of the first patterned layer.
        factorization, truncation, doublings, grid: As in
            :func:`~hyperbolic_optics.rcwa2d.solve_crossed_grating`.

    Returns:
        ``[2N]`` complex eigenvalues; a resonance has one equal to 1.
    """
    return jnp.linalg.eigvals(
        _round_trip_matrix(
            layers,
            lattice,
            frequency,
            eps_top,
            eps_bottom,
            max_order,
            bloch=bloch,
            split=split,
            factorization=factorization,
            truncation=truncation,
            doublings=doublings,
            grid=grid,
        )  # fmt: skip
    )


def spectrum_poles(x, values, tolerance: float = 1e-11, max_terms: int = 60):
    """Poles of a sampled response by AAA rational approximation.

    Args:
        x: Real sample points (e.g. frequencies) ``[K]``.
        values: Complex response there ``[K]`` (e.g. a reflection amplitude).
        tolerance: Relative accuracy of the fit.
        max_terms: Most support points.

    Returns:
        ``(poles, residues)``, complex. Only poles near the sampled interval
        mean anything; spurious ones (tiny residues, or far away) are the
        usual AAA artefacts.
    """
    z = np.asarray(x, dtype=complex)
    f = np.asarray(values, dtype=complex)
    mask = np.ones(len(z), dtype=bool)
    support, weights = [], None
    approximation = np.full_like(f, np.mean(f))
    for _ in range(min(max_terms, len(z) - 1)):
        j = int(np.argmax(np.abs(f - approximation) * mask))
        support.append(j)
        mask[j] = False
        zs, fs = z[support], f[support]
        cauchy = 1.0 / (z[mask, None] - zs[None, :])
        loewner = f[mask, None] * cauchy - cauchy * fs[None, :]
        _, _, vh = np.linalg.svd(loewner, full_matrices=False)
        weights = vh[-1].conj()
        numerator = cauchy @ (weights * fs)
        denominator = cauchy @ weights
        approximation = f.copy()
        approximation[mask] = numerator / denominator
        if np.max(np.abs(f - approximation)) <= tolerance * np.max(np.abs(f)):
            break
    zs, fs = z[support], f[support]
    # poles: generalised eigenvalues of the arrowhead pencil
    m = len(zs)
    b = np.eye(m + 1, dtype=complex)
    b[0, 0] = 0
    e = np.zeros((m + 1, m + 1), dtype=complex)
    e[0, 1:] = weights
    e[1:, 0] = 1
    e[1:, 1:] = np.diag(zs)
    poles = scipy.linalg.eigvals(e, b)
    poles = poles[np.isfinite(poles)]

    # residues: numerator / d(denominator)/dz at each pole
    def rational_parts(p):
        c = 1.0 / (p - zs)
        return c @ (weights * fs), -(c**2) @ weights

    residues = np.array([n / d for n, d in (rational_parts(p) for p in poles)])
    return poles, residues


# ---------------------------------------------------------------------------
# Guided modes of planar stacks
# ---------------------------------------------------------------------------


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


def _planar_parts(eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, kx, doublings):
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
            eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, n_eff, doublings
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
    complex effective indices. Found as the roots of the round-trip condition
    in the complex ``n_eff`` plane by Beyn's contour method, then refined (see
    :func:`band_edge_modes`). Rotate the tensors to change the propagation
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
        quadrature, probes, tolerance, doublings, seed: As in
            :func:`band_edge_modes`.

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
        eps, thickness_um, wavelength_um, eps_top, eps_bottom, mu, n_eff, doublings
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


# ---------------------------------------------------------------------------
# Leaky modes of 1D gratings (grating couplers)
# ---------------------------------------------------------------------------


class LeakyMode(NamedTuple):
    """A leaky Bloch mode of a 1D grating waveguide, propagating along +x.

    Attributes:
        bloch_index: Complex ``kx₀/k₀`` of the mode's fundamental harmonic;
            ``Im > 0``: it decays as it propagates, by radiating.
        alpha_per_um: Field decay rate ``k₀·Im(bloch_index)`` (power: ``2α``).
        orders: The diffraction orders that radiate into either half-space.
        emission_deg: ``{order: angle}`` of emission into the top half-space,
            from the normal, positive towards +x.
        directionality: Fraction of the radiated power going into the top.
        top_polarization: ``{order: (p, s)}`` unit Jones vector of the emission
            into the top (in that order's own p/s).
        multiplicity: As for :class:`Resonance`.
        residual: ``|1 − μ|`` at the root.
    """

    bloch_index: complex
    alpha_per_um: float
    orders: tuple[int, ...]
    emission_deg: dict
    directionality: float
    top_polarization: dict
    multiplicity: int
    residual: float


def _grating_parts(layers, period_um, wavelength_um, eps_top, eps_bottom, max_order,
                   factorization, kx0):  # fmt: skip
    """Analytic bases, operators and half-space kz of a 1D grating at complex ``kx₀``."""
    from hyperbolic_optics import rcwa

    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    orders = jnp.arange(-max_order, max_order + 1)
    kx = kx0 + orders * (jnp.asarray(wavelength_um) / jnp.asarray(period_um))
    ky = jnp.zeros_like(kx)
    direction = (jnp.ones(()), jnp.zeros(()))
    size = 2 * max_order + 1
    eps_ref = 2 * kx**2 + 1.0
    w_ref, _ = rcwa._harmonic_modes(eps_ref, kx, ky, direction, k0)
    kz_ref = jnp.sqrt(1.0 + kx**2 + 0j)
    w_ref = w_ref / jnp.concatenate([jnp.sqrt(kz_ref)] * 4)[None, :]
    w_top, _ = rcwa._harmonic_modes(jnp.asarray(eps_top), kx, ky, direction, k0)
    w_bottom, _ = rcwa._harmonic_modes(jnp.asarray(eps_bottom), kx, ky, direction, k0)
    deltas = []
    for layer in layers:
        eps = rcwa._factorize(jnp.asarray(layer.eps) + 0j, layer.boundaries, max_order,
                              factorization, layer.edge_width)  # fmt: skip
        if layer.mu is None:
            eye = jnp.eye(size, dtype=eps.dtype)
            mu = jnp.einsum("ij,mn->ijmn", jnp.eye(3, dtype=eps.dtype), eye)
        else:
            mu = rcwa._factorize(jnp.asarray(layer.mu) + 0j, layer.boundaries, max_order,
                                 factorization, layer.edge_width)  # fmt: skip
        delta = rcwa._fourier_system_matrix(eps, mu, kx, ky)
        deltas.append(bcm._solve(w_ref, delta @ w_ref))
    kz = {
        "top": bcm._kz_outgoing(jnp.asarray(eps_top), kx, k0),
        "bottom": bcm._kz_outgoing(jnp.asarray(eps_bottom), kx, k0),
    }
    return w_ref, w_top, w_bottom, deltas, k0, kx, kz


def _grating_cascades(layers, parts, split, doublings):
    w_ref, w_top, w_bottom, deltas, k0, _, _ = parts
    index, fraction = split
    above = _interface(w_top, w_ref)
    for i, (layer, delta) in enumerate(zip(layers, deltas, strict=True)):
        k0d = k0 * jnp.asarray(layer.thickness_um)
        if i < index:
            above = _star(above, _layer(delta, k0d, doublings))
        elif i == index:
            above = _star(above, _layer(delta, k0d * fraction, doublings))
            below = _layer(delta, k0d * (1 - fraction), doublings)
        else:
            below = _star(below, _layer(delta, k0d, doublings))
    below = _star(below, _interface(w_ref, w_bottom))
    return above, below


def grating_leaky_modes(
    layers,
    period_um,
    wavelength_um,
    eps_top,
    eps_bottom,
    index_range: tuple[float, float],
    max_order: int = 10,
    max_loss: float | None = None,
    split: tuple[int, float] | None = None,
    factorization: str = "li",
    doublings: int | None = None,
    quadrature: int = 64,
    probes: int = 16,
    tolerance: float = 1e-11,
    seed: int = 0,
) -> list[LeakyMode]:
    """Leaky Bloch modes of a 1D grating waveguide: the modes a grating coupler uses.

    The guided mode of the waveguide, periodically perturbed, radiates through
    the diffraction orders that propagate in a half-space. Its Bloch
    wavevector is then complex; it is found as a root of the round-trip
    condition in the complex ``kx₀/k₀`` plane (Beyn's method, as in
    :func:`band_edge_modes`), with the half-spaces on their outgoing branch.

    Args:
        layers: :mod:`hyperbolic_optics.rcwa` layers from the top down, the
            grating and the waveguide under it; full tensors anywhere.
        period_um: The grating period.
        wavelength_um: Vacuum wavelength (real).
        eps_top, eps_bottom: The half-spaces (isotropic).
        index_range: Real ``kx₀/k₀`` to search: near the guided mode's
            effective index, inside ``(λ/Λ − n, n_eff_max)`` so that the
            radiating order is the −1st.
        max_order: Harmonics ``|m| ≤ max_order`` (static).
        max_loss: Largest ``Im kx₀/k₀`` searched; by default a quarter of the
            range (grating couplers are strongly leaky by design).
        split: ``(layer, fraction)`` of the internal plane; by default the
            middle of the layer of highest index (the waveguide core), where
            the mode is strong.
        factorization: ``"li"`` (default) or ``"laurent"``.
        doublings, quadrature, probes, tolerance, seed: As in :func:`band_edge_modes`.

    Returns:
        :class:`LeakyMode` s, highest ``Re kx₀`` first.
    """
    from hyperbolic_optics.rcwa import default_doublings

    layers = list(layers)
    doublings = default_doublings(max_order) if doublings is None else doublings
    if split is None:
        peak = [float(np.max(np.real(np.asarray(layer.eps)))) for layer in layers]
        split = (int(np.argmax(peak)), 0.5)
    low, high = (float(v) for v in index_range)
    depth = 0.25 * (high - low) if max_loss is None else float(max_loss)

    def matrix(kx0):
        parts = _grating_parts(layers, period_um, wavelength_um, eps_top, eps_bottom,
                               max_order, factorization, kx0)  # fmt: skip
        above, below = _grating_cascades(layers, parts, split, doublings)
        return above[1] @ below[2]

    round_trip = jax.jit(matrix)
    roots = _contour_roots(round_trip, (low, high), depth, quadrature, probes, tolerance, seed)

    size = 2 * max_order + 1
    k0 = 2 * np.pi / float(wavelength_um)
    modes = []
    for kx0, residual, multiplicity in sorted(roots, key=lambda r: -r[0].real):
        if kx0.imag < -1e-9 * abs(kx0):
            continue  # grows as it propagates: not a mode of a passive grating
        parts = _grating_parts(layers, period_um, wavelength_um, eps_top, eps_bottom,
                               max_order, factorization, kx0)  # fmt: skip
        above, below = _grating_cascades(layers, parts, split, doublings)
        values, vectors = np.linalg.eig(np.asarray(above[1] @ below[2]))
        down = vectors[:, np.argmin(np.abs(1 - values))]
        up = np.asarray(below[2]) @ down
        top = np.asarray(above[3]) @ up  # outgoing into the top half-space: [p·N, s·N]
        bottom = np.asarray(below[0]) @ down  # outgoing into the bottom half-space
        kx = np.asarray(parts[5])
        kz_top, kz_bottom = (np.asarray(parts[6][side]) for side in ("top", "bottom"))

        def radiated(amplitudes, kz):
            # a unit-field mode's normal flux is ½ Re(kz) for p and s alike
            return 0.5 * np.real(kz) * (np.abs(amplitudes[:size]) ** 2
                                        + np.abs(amplitudes[size:]) ** 2)  # fmt: skip

        up_power, down_power = radiated(top, kz_top), radiated(bottom, kz_bottom)
        # an order radiates where it propagates on the real axis: |Re kx_m| < n
        open_top = np.abs(kx.real) < np.sqrt(float(np.real(eps_top)))
        open_bottom = np.abs(kx.real) < np.sqrt(float(np.real(eps_bottom)))
        total = up_power[open_top].sum() + down_power[open_bottom].sum()
        orders = tuple(int(m) for m in np.arange(-max_order, max_order + 1)[open_top | open_bottom])
        emission, polarization = {}, {}
        n_top = np.sqrt(float(np.real(eps_top)))
        for i in np.flatnonzero(open_top):
            m = int(i - max_order)
            emission[m] = float(np.degrees(np.arcsin(np.clip(kx[i].real / n_top, -1, 1))))
            jones = np.array([top[i], top[size + i]])
            polarization[m] = tuple(jones / np.linalg.norm(jones))
        modes.append(
            LeakyMode(
                bloch_index=complex(kx0),
                alpha_per_um=float(k0 * kx0.imag),
                orders=orders,
                emission_deg=emission,
                directionality=float(up_power[open_top].sum() / total) if total > 0 else 0.0,
                top_polarization=polarization,
                multiplicity=multiplicity,
                residual=residual,
            )
        )
    return modes
