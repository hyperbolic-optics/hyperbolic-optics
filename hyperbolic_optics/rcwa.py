"""Differentiable RCWA for 1D-periodic gratings of full-tensor anisotropic media.

Rigorous coupled-wave analysis (the Fourier modal method) for structures that
are periodic in x, uniform in y and layered in z. Each layer is piecewise
constant in x, every segment carrying its own 3×3 permittivity and permeability
-- any orientation, including the xz and yz components a tilted crystal axis
produces. Incidence may be conical (a plane of incidence at any azimuth to the
grating vector).

Layers are solved with the eigendecomposition-free cascade of
:mod:`hyperbolic_optics.bcm`, now on ``4N × 4N`` Fourier-space operators for
``N = 2·max_order + 1`` harmonics, so the whole calculation -- including the
Fourier coefficients of the geometry -- is differentiable with ``jax.grad``:
with respect to segment boundaries (fill factor, ridge position), thicknesses,
period, tensors and their orientations, wavelength and incidence.

Fourier factorisation
---------------------
Across an x-discontinuity the components ``Dx, Ey, Ez`` (and ``Bx, Hy, Hz``) are
continuous. The constitutive relation is therefore first solved for
``(Ex, Dy, Dz)`` in terms of ``(Dx, Ey, Ez)``: every product in that form pairs
a discontinuous coefficient with a continuous field, for which Laurent's rule
(a Toeplitz matrix of Fourier coefficients) converges. The result is then
rearranged back into ``D = [[ε]]·E`` with the inverse taken only in Fourier
space. This is Li's factorisation for anisotropic lamellar gratings (L. Li,
J. Mod. Opt. 45, 1313 (1998)); for an isotropic medium it reduces to the
familiar inverse rule for ``Dx`` and Laurent's rule for ``Dy`` and ``Dz``.
``factorization="laurent"`` uses plain Toeplitz matrices of ε instead, which is
correct but converges slowly for TM light on high-contrast gratings; it is kept
for comparison.

Conventions
-----------
Fields and time dependence follow :mod:`hyperbolic_optics.bcm`. Harmonic ``m``
has in-plane wavevector ``(kx₀ + m·λ/Λ, ky)`` (normalised by ``k₀``). Every
order's amplitudes are resolved into that order's own p and s: ``s`` is
perpendicular to its plane of diffraction, along ``ẑ × û`` with ``û`` the unit
in-plane wavevector, and p completes the triad with the sign convention of the
planar solver. (At exactly zero in-plane wavevector ``û`` is taken along the
plane of incidence.) Results are indexed ``[..., order, out, in]`` with 0 = p and
1 = s; ``R`` and ``T`` are diffraction efficiencies, zero for evanescent orders.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hyperbolic_optics.bcm import (
    DEFAULT_DOUBLINGS,
    _check_x64,
    _interface,
    _kz,
    _layer,
    _solve,
    _star,
)

__all__ = [
    "GratingLayer",
    "GratingResult",
    "default_doublings",
    "fourier_coefficients",
    "lamellar_layer",
    "sampled_layer",
    "solve_grating",
    "uniform_layer",
]


class GratingLayer(NamedTuple):
    """One layer: ``S`` segments of constant material across one period.

    Attributes:
        thickness_um: Layer thickness in µm, ``[...]``.
        boundaries: Segment edges as fractions of the period, ``[..., S + 1]``,
            increasing, with ``boundaries[S] − boundaries[0] = 1``.
        eps: Segment permittivity tensors in the lab frame, ``[..., S, 3, 3]``.
        mu: Segment permeability tensors, same layout; ``None`` for μ = 1.
        edge_width: Optional smoothing, as a fraction of the period: the profile
            is convolved with a box this wide, which multiplies Fourier
            coefficient ``m`` by ``sinc(m · edge_width)``. ``0`` keeps hard
            edges. A few percent removes the truncation ripple that makes
            geometry gradients of hyperbolic (high-k) ridges unusable; see
            docs/user-guide/gratings.md.
    """

    thickness_um: jax.Array
    boundaries: jax.Array
    eps: jax.Array
    mu: jax.Array | None = None
    edge_width: jax.Array | float = 0.0


class GratingResult(NamedTuple):
    """Complex amplitudes and efficiencies, each ``[..., order, out, in]``."""

    orders: jax.Array
    r: jax.Array
    t: jax.Array
    R: jax.Array
    T: jax.Array


def uniform_layer(eps, thickness_um, mu=None) -> GratingLayer:
    """A homogeneous layer, ``eps`` ``[..., 3, 3]``."""
    eps = jnp.asarray(eps)
    return GratingLayer(
        jnp.asarray(thickness_um),
        jnp.array([0.0, 1.0]),
        eps[..., None, :, :],
        None if mu is None else jnp.asarray(mu)[..., None, :, :],
    )


def lamellar_layer(
    eps_ridge,
    eps_groove,
    fill,
    thickness_um,
    offset=0.0,
    mu_ridge=None,
    mu_groove=None,
    edge_width=0.0,
) -> GratingLayer:
    """Ridge of width ``fill`` (fraction of the period) starting at ``offset``.

    ``fill`` and ``offset`` enter the Fourier coefficients analytically, so the
    layer is differentiable in both. ``edge_width`` smooths the edges (see
    :class:`GratingLayer`).
    """
    fill, offset = jnp.asarray(fill), jnp.asarray(offset)
    boundaries = jnp.stack([offset, offset + fill, offset + 1.0], axis=-1)
    eps_ridge, eps_groove = jnp.broadcast_arrays(jnp.asarray(eps_ridge), jnp.asarray(eps_groove))
    eps = jnp.stack([eps_ridge, eps_groove], axis=-3)
    mu = None
    if mu_ridge is not None or mu_groove is not None:
        eye = jnp.eye(3)
        mu_ridge = eye if mu_ridge is None else jnp.asarray(mu_ridge)
        mu_groove = eye if mu_groove is None else jnp.asarray(mu_groove)
        mu = jnp.stack(jnp.broadcast_arrays(mu_ridge, mu_groove), axis=-3)
    return GratingLayer(jnp.asarray(thickness_um), boundaries, eps, mu, edge_width)


def sampled_layer(eps_samples, thickness_um, mu_samples=None) -> GratingLayer:
    """A profile sampled at ``S`` equal-width segments, ``[..., S, 3, 3]``."""
    eps_samples = jnp.asarray(eps_samples)
    boundaries = jnp.linspace(0.0, 1.0, eps_samples.shape[-3] + 1)
    mu = None if mu_samples is None else jnp.asarray(mu_samples)
    return GratingLayer(jnp.asarray(thickness_um), boundaries, eps_samples, mu)


# ---------------------------------------------------------------------------
# Fourier space
# ---------------------------------------------------------------------------


def fourier_coefficients(values: jax.Array, boundaries: jax.Array, max_index: int) -> jax.Array:
    """Exact Fourier coefficients ``c_m``, ``|m| ≤ max_index``, of a step function.

    ``f(x) = Σ_m c_m exp(2πi·m·x)`` over one period, with ``f = values[s]`` on
    ``[boundaries[s], boundaries[s + 1])``. Analytic in the boundaries.

    Args:
        values: ``[..., S]``.
        boundaries: ``[..., S + 1]``.
        max_index: Highest ``|m|``.

    Returns:
        ``[..., 2·max_index + 1]``, index ``m + max_index``.
    """
    m = jnp.arange(-max_index, max_index + 1)
    phase = jnp.exp(-2j * jnp.pi * m * boundaries[..., :, None])  # [..., S + 1, M]
    zero = m == 0
    denominator = jnp.where(zero, 1.0, -2j * jnp.pi * m)  # safe at m = 0
    integrals = jnp.where(
        zero,
        (boundaries[..., 1:] - boundaries[..., :-1])[..., None],
        (phase[..., 1:, :] - phase[..., :-1, :]) / denominator,
    )  # [..., S, M]
    return jnp.sum(values[..., None] * integrals, axis=-2)


def _toeplitz(
    values: jax.Array, boundaries: jax.Array, max_order: int, edge_width=0.0
) -> jax.Array:
    """Laurent-rule matrix ``[[f]]_{mn} = c_{m−n}``, ``[..., N, N]``.

    ``edge_width`` convolves ``f`` with a box of that width (fraction of the
    period): ``c_m → c_m · sinc(m · edge_width)``.
    """
    coefficients = fourier_coefficients(values, boundaries, 2 * max_order)
    m = jnp.arange(-2 * max_order, 2 * max_order + 1)
    coefficients = coefficients * jnp.sinc(m * jnp.asarray(edge_width)[..., None])
    size = 2 * max_order + 1
    index = jnp.arange(size)[:, None] - jnp.arange(size)[None, :] + 2 * max_order
    return coefficients[..., index]


def _factorize(
    tensor: jax.Array, boundaries: jax.Array, max_order: int, rule: str, edge_width=0.0
) -> jax.Array:
    """Fourier-space constitutive matrix ``[[ε]]`` as ``[..., 3, 3, N, N]`` blocks."""

    def t(values):
        return _toeplitz(values, boundaries, max_order, edge_width)

    if rule == "laurent":
        rows = [[t(tensor[..., i, j]) for j in range(3)] for i in range(3)]
        return jnp.stack([jnp.stack(row, axis=-3) for row in rows], axis=-4)
    if rule != "li":
        raise ValueError(f"Unknown factorization {rule!r}; use 'li' or 'laurent'.")

    e = [[tensor[..., i, j] for j in range(3)] for i in range(3)]
    # (Ex, Dy, Dz) in terms of the x-continuous (Dx, Ey, Ez); Laurent's rule
    # applies to every product here.
    inv_xx = t(1.0 / e[0][0])
    a = _solve(inv_xx, jnp.broadcast_to(jnp.eye(inv_xx.shape[-1]), inv_xx.shape))  # [[1/εxx]]⁻¹
    xy, xz = t(e[0][1] / e[0][0]), t(e[0][2] / e[0][0])
    yx, zx = t(e[1][0] / e[0][0]), t(e[2][0] / e[0][0])

    def schur(i, j):
        return t(e[i][j] - e[i][0] * e[0][j] / e[0][0])

    rows = [
        [a, a @ xy, a @ xz],
        [yx @ a, schur(1, 1) + yx @ a @ xy, schur(1, 2) + yx @ a @ xz],
        [zx @ a, schur(2, 1) + zx @ a @ xy, schur(2, 2) + zx @ a @ xz],
    ]
    return jnp.stack([jnp.stack(jnp.broadcast_arrays(*row), axis=-3) for row in rows], axis=-4)


def _fourier_system_matrix(eps: jax.Array, mu: jax.Array, kx: jax.Array, ky: jax.Array):
    """Fourier-space Berreman operator on ``F = [Ex, Ey, Hx, Hy]`` (N each), ``[..., 4N, 4N]``.

    The planar derivation of :func:`hyperbolic_optics.bcm.system_matrix` with
    ``∂x → i·diag(kx)``, ``∂y → i·diag(ky)`` and each tensor component a block.
    ``kx`` and ``ky`` are per harmonic, ``[..., N]``; a scalar ``ky`` (a 1D grating's
    shared one) broadcasts.
    """
    size = kx.shape[-1]
    ky = jnp.asarray(ky)
    ky_batch = ky.shape[:-1] if ky.ndim else ()
    batch = jnp.broadcast_shapes(eps.shape[:-4], mu.shape[:-4], kx.shape[:-1], ky_batch)
    eps = jnp.broadcast_to(eps, batch + eps.shape[-4:])
    mu = jnp.broadcast_to(mu, batch + mu.shape[-4:])
    eye = jnp.eye(size, dtype=eps.dtype)
    big_kx = jnp.broadcast_to(kx, batch + (size,))[..., :, None] * eye
    big_ky = jnp.broadcast_to(ky, batch + (size,))[..., :, None] * eye
    zero = jnp.zeros_like(big_kx)

    def e(i, j):
        return eps[..., i, j, :, :]

    def m(i, j):
        return mu[..., i, j, :, :]

    def row(*blocks):
        return jnp.concatenate(blocks, axis=-1)

    # Ez = a·F from (εE)_z = ky·Hx − Kx·Hy; Hz = b·F from (μH)_z = Kx·Ey − ky·Ex.
    a = _solve(e(2, 2), row(-e(2, 0), -e(2, 1), big_ky, -big_kx))
    b = _solve(m(2, 2), row(-big_ky, big_kx, -m(2, 0), -m(2, 1)))

    row_ex = big_kx @ a + row(zero, zero, m(1, 0), m(1, 1)) + m(1, 2) @ b
    row_ey = big_ky @ a - (row(zero, zero, m(0, 0), m(0, 1)) + m(0, 2) @ b)
    row_hx = big_kx @ b - (row(e(1, 0), e(1, 1), zero, zero) + e(1, 2) @ a)
    row_hy = big_ky @ b + row(e(0, 0), e(0, 1), zero, zero) + e(0, 2) @ a
    return jnp.concatenate([row_ex, row_ey, row_hx, row_hy], axis=-2)


def _harmonic_modes(eps, qx, qy, direction):
    """Isotropic-medium modes for every harmonic: ``W`` ``[..., 4N, 4N]`` and flux ``[..., 4N]``.

    Rows are ``[Ex, Ey, Hx, Hy]`` × harmonics; columns ``[p→, s→, p←, s←]`` ×
    harmonics. Unit total field; each mode's normal flux is ``±½ Re(kz)`` for
    real ε.

    Args:
        eps: Permittivity per harmonic (broadcasts to ``qx``).
        qx: In-plane x wavevector per harmonic ``[..., N]``.
        qy: In-plane y wavevector per harmonic ``[..., N]``; a scalar broadcasts.
        direction: ``(ux, uy)`` used where the in-plane wavevector vanishes.
    """
    shape = jnp.broadcast_shapes(jnp.shape(qx), jnp.shape(qy))
    qx, qy = jnp.broadcast_to(qx, shape), jnp.broadcast_to(qy, shape)
    k_par_sq = qx**2 + qy**2
    moving = jnp.abs(k_par_sq) > 1e-24
    k_par = jnp.where(moving, jnp.sqrt(jnp.where(moving, k_par_sq, 1.0)), 0.0)
    ux = jnp.where(moving, qx / jnp.where(moving, k_par, 1.0), direction[0])
    uy = jnp.where(moving, qy / jnp.where(moving, k_par, 1.0), direction[1])

    eps = jnp.broadcast_to(eps, qx.shape) + 0j
    n = jnp.sqrt(eps)
    kz = _kz(eps, k_par)
    zero = jnp.zeros_like(kz)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    forward_k, backward_k = (qx, qy, kz), (qx, qy, -kz)
    s_field = (-uy + zero, ux + zero, zero)
    modes = [
        (forward_k, (kz * ux / n, kz * uy / n, -k_par / n)),
        (forward_k, s_field),
        (backward_k, (-kz * ux / n, -kz * uy / n, -k_par / n)),
        (backward_k, s_field),
    ]
    columns, fluxes = [], []
    for k, field in modes:
        h = cross(k, field)
        tangential = (field[0], field[1], h[0], h[1])  # [Ex, Ey, Hx, Hy], each [..., N]
        columns.append([t * jnp.eye(qx.shape[-1]) for t in (x[..., None] for x in tangential)])
        fluxes.append(0.5 * jnp.real(field[0] * jnp.conj(h[1]) - field[1] * jnp.conj(h[0])))
    rows = [jnp.concatenate([columns[c][r] for c in range(4)], axis=-1) for r in range(4)]
    return jnp.concatenate(rows, axis=-2), jnp.concatenate(fluxes, axis=-1)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def default_doublings(max_order: int) -> int:
    """Doublings that keep the slice error at the planar level as orders are added.

    The operator's norm grows as ``kx_max² ∝ max_order²`` and the fourth-order
    start leaves an error ``∝ (norm · k₀d)⁵ / 2⁴ᴺ``, so holding it fixed takes
    ``2.5 · log₂(max_order / 10)`` extra doublings beyond ten orders. At 161
    orders this is 28 doublings, which brings a lossless grating's energy error
    from ~1e-7 (at 20) down to its ~1e-9 rounding floor.
    """
    if max_order <= 10:
        return DEFAULT_DOUBLINGS
    return DEFAULT_DOUBLINGS + math.ceil(2.5 * math.log2(max_order / 10))


def solve_grating(
    layers: list[GratingLayer],
    period_um: jax.Array,
    wavelength_um: jax.Array,
    incidence_rad: jax.Array,
    eps_incident: jax.Array,
    eps_exit: jax.Array,
    max_order: int,
    azimuth_rad: jax.Array = 0.0,
    factorization: str = "li",
    doublings: int | None = None,
) -> GratingResult:
    """Diffraction of a plane wave by a stack of 1D-periodic anisotropic layers.

    Args:
        layers: :class:`GratingLayer` s from the incident side down; build them
            with :func:`uniform_layer`, :func:`lamellar_layer` or
            :func:`sampled_layer`.
        period_um: Grating period in µm, along x.
        wavelength_um: Vacuum wavelength in µm.
        incidence_rad: Polar angle of incidence in the incident medium.
        eps_incident: Real, positive permittivity of the incident half-space.
        eps_exit: Real, positive permittivity of the exit half-space.
        max_order: Keep diffraction orders ``−max_order … max_order`` (static).
        azimuth_rad: Angle of the plane of incidence from the grating vector
            (x); nonzero gives conical diffraction.
        factorization: ``"li"`` (default) or ``"laurent"`` (static).
        doublings: Doublings per layer (static), as in :mod:`hyperbolic_optics.bcm`.
            Defaults to :func:`default_doublings` for ``max_order``.

    Returns:
        :class:`GratingResult`; ``orders`` lists the orders, and ``r``, ``t``,
        ``R``, ``T`` are ``[..., order, out, in]`` with 0 = p and 1 = s. Only the
        zeroth order is incident.
    """
    _check_x64()
    if doublings is None:
        doublings = default_doublings(max_order)
    size = 2 * max_order + 1
    orders = jnp.arange(-max_order, max_order + 1)
    eps_incident, eps_exit = jnp.asarray(eps_incident), jnp.asarray(eps_exit)
    wavelength_um, azimuth_rad = jnp.asarray(wavelength_um), jnp.asarray(azimuth_rad)

    k_par = jnp.sqrt(eps_incident) * jnp.sin(jnp.asarray(incidence_rad))
    kx = (k_par * jnp.cos(azimuth_rad))[..., None] + orders * (
        wavelength_um / jnp.asarray(period_um)
    )[..., None]
    ky = k_par * jnp.sin(azimuth_rad)
    direction = (jnp.cos(azimuth_rad)[..., None], jnp.sin(azimuth_rad)[..., None])
    k0 = 2 * jnp.pi / wavelength_um

    # Reference medium per harmonic with kz = √(1 + k∥²), scaled to unit power:
    # always propagating, and impedance-matched to the evanescent and high-k
    # waves of the high orders (see bcm._reference_modes).
    ky = jnp.asarray(ky)[..., None]  # shared by every harmonic of a 1D grating
    k_par_sq = kx**2 + ky**2
    w_ref, flux_ref = _harmonic_modes(2.0 * k_par_sq + 1.0, kx, ky, direction)
    w_ref = w_ref / jnp.sqrt(2.0 * jnp.abs(flux_ref))[..., None, :]
    w_in, flux_in = _harmonic_modes(eps_incident[..., None], kx, ky, direction)
    w_out, flux_out = _harmonic_modes(eps_exit[..., None], kx, ky, direction)

    scattering = _interface(w_in, w_ref)
    for layer in layers:
        eps = _factorize(
            jnp.asarray(layer.eps) + 0j,
            layer.boundaries,
            max_order,
            factorization,
            layer.edge_width,
        )
        if layer.mu is None:
            eye = jnp.eye(size, dtype=eps.dtype)
            mu = jnp.einsum("ij,mn->ijmn", jnp.eye(3, dtype=eps.dtype), eye)
        else:
            mu = _factorize(
                jnp.asarray(layer.mu) + 0j,
                layer.boundaries,
                max_order,
                factorization,
                layer.edge_width,
            )
        delta = _fourier_system_matrix(eps, mu, kx, ky)
        delta_ref = _solve(w_ref, delta @ w_ref)
        k0d = k0 * jnp.asarray(layer.thickness_um)
        scattering = _star(scattering, _layer(delta_ref, k0d, doublings))
    scattering = _star(scattering, _interface(w_ref, w_out))

    t_full, _, r_full, _ = scattering  # [..., 2N (p then s, by order), 2N]
    incident = jnp.array([max_order, size + max_order])  # zeroth-order p, s

    def by_order(block):
        # [..., 2N, 2] -> [..., order, out, in]
        amplitude = block[..., :, incident]
        return jnp.stack([amplitude[..., :size, :], amplitude[..., size:, :]], axis=-2)

    r, t = by_order(r_full), by_order(t_full)
    forward_in = flux_in[..., : 2 * size]
    power_in = jnp.stack([forward_in[..., max_order], forward_in[..., size + max_order]], -1)

    def per_order(flux):
        forward = flux[..., : 2 * size]
        return jnp.stack([forward[..., :size], forward[..., size:]], axis=-1)  # [..., order, out]

    R = jnp.abs(r) ** 2 * per_order(flux_in)[..., :, :, None] / power_in[..., None, None, :]
    T = jnp.abs(t) ** 2 * per_order(flux_out)[..., :, :, None] / power_in[..., None, None, :]
    return GratingResult(orders=orders, r=r, t=t, R=R, T=T)
