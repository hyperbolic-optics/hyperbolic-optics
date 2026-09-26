"""Differentiable planar solver: an eigendecomposition-free cascade in JAX.

The NumPy engine (:mod:`hyperbolic_optics.structure`) diagonalises each layer's
4×4 Berreman matrix. Eigenvectors are awkward to differentiate -- their phase is
arbitrary and their derivative diverges where two modes are degenerate, as they
are in every isotropic layer -- so this module builds each layer's scattering
matrix without them, from matrix products and linear solves only. Every step is
then an ordinary JAX operation, and ``jax.grad`` gives exact derivatives with
respect to thicknesses, tensor components, orientations, incidence angle and
wavelength.

Method
------
The tangential field ``F = [Ex, Ey, Hx, Hy]`` (H scaled by the free-space
impedance, time dependence ``exp(-iωt)``) obeys ``dF/dz̃ = iΔF`` with ``z̃ = k₀z``
and ``Δ`` the Berreman matrix -- the same matrix, component for component, as
:meth:`hyperbolic_optics.waves.Wave.delta_matrix_calc`. For a slice thin enough,
``exp(iΔτ) − I`` is a short Taylor series. The slice is converted to a
scattering matrix, and ``2ᴺ`` slices are assembled by ``N`` doublings, each the
Redheffer star product of the current interval with itself, carrying the
transmission blocks as increments from the identity so the small early updates
are not lost to rounding. This is the bisection-doubling scheme of the precise
integration method (Zhong & Williams 1994) used by the boundary-value cascade of
Yang, Song & Cai, arXiv:2608.06185.

One choice here differs from that paper. It represents an interval by E at one
end and H at the other; that relation is singular whenever a lossless layer is
an odd number of quarter-waves thick (E at one face then fixes H at the other).
Here intervals are represented instead by wave amplitudes in a reference medium
of permittivity ``2kx² + 1``, whose waves always propagate, with
``kz = √(1 + kx²)``, and are scaled to unit power. In that basis the scattering
matrix of any passive interval is power-normalised (norm at most 1), so no
doubling step can meet a singular inverse. Letting ``kz`` grow with ``kx`` keeps
the reference impedance close to that of the waves it represents: evanescent
waves at large ``kx`` have ``|kz| ≈ kx``, and a fixed ``kz = 1`` reference
reflects them almost totally, which amplifies rounding (by ~1e3 for the
high-order harmonics of :mod:`hyperbolic_optics.rcwa`). The initialisation is
also derived independently rather than taken from the paper's supplement; this
is a related implementation, not a reproduction.

Scope
-----
- Any number of layers, each with full 3×3 permittivity and permeability.
- Isotropic, lossless incident and exit half-spaces (real positive ε).
- The plane of incidence is x-z. Rotate tensors (:func:`rotation_matrix`) to
  change the in-plane orientation, as the payload API does.
- All inputs broadcast over any leading batch dimensions.
- Grazing cutoff in the exit medium (``kx`` exactly equal to its index) is
  singular, as in any plane-wave basis.

Coefficients are the 2×2 Jones matrices ``r[..., out, in]`` and
``t[..., out, in]`` with index 0 = p and 1 = s, in total-field amplitudes on the
same sign convention as the NumPy engine: ``r[..., 0, 0]`` is ``r_pp`` and
``r[..., 1, 0]`` is its ``r_ps`` (p in, s out). ``R`` and ``T`` are the
corresponding powers; an evanescent exit gives ``T = 0``.

Precision: reference accuracy needs 64-bit JAX, which this module does not
switch on for you -- call ``jax.config.update("jax_enable_x64", True)`` first.
Without it everything runs in complex64 and a warning says so.
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["PlanarResult", "rotate_tensor", "rotation_matrix", "solve_stack", "system_matrix"]

#: Doublings per layer unless told otherwise: each layer is split into 2**20
#: slices. The fourth-order start then leaves a relative error of order
#: ``(|Δ| k₀d)⁵ / (120 · 2⁸⁰)`` -- below 1e-12 up to ``|Δ| k₀d ≈ 1e3``.
DEFAULT_DOUBLINGS = 20


class PlanarResult(NamedTuple):
    """Complex Jones matrices and powers, each ``[..., out, in]`` with 0 = p, 1 = s."""

    r: jax.Array
    t: jax.Array
    R: jax.Array
    T: jax.Array


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def rotation_matrix(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Euler rotation ``Rz(z) · Ry(y) · Rx(x)`` in radians, ``[..., 3, 3]``.

    The payload API's ``rotationX``/``rotationY``/``rotationZ`` convention, so a
    tensor rotated with :func:`rotate_tensor` matches the one a ``Crystal Layer``
    would use for the same angles.
    """
    x, y, z = jnp.broadcast_arrays(*(jnp.asarray(angle, dtype=float) for angle in (x, y, z)))
    zero, one = jnp.zeros_like(x), jnp.ones_like(x)

    def matrix(rows):
        return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)

    cx, sx, cy, sy, cz, sz = jnp.cos(x), jnp.sin(x), jnp.cos(y), jnp.sin(y), jnp.cos(z), jnp.sin(z)
    rot_x = matrix([[one, zero, zero], [zero, cx, -sx], [zero, sx, cx]])
    rot_y = matrix([[cy, zero, sy], [zero, one, zero], [-sy, zero, cy]])
    rot_z = matrix([[cz, -sz, zero], [sz, cz, zero], [zero, zero, one]])
    return rot_z @ rot_y @ rot_x


def rotate_tensor(tensor: jax.Array, rotation: jax.Array) -> jax.Array:
    """Similarity transform ``R · tensor · Rᵀ``, broadcasting over batch dimensions."""
    return rotation @ tensor @ jnp.swapaxes(rotation, -1, -2)


def system_matrix(eps: jax.Array, mu: jax.Array, kx: jax.Array) -> jax.Array:
    """Berreman matrix ``Δ`` with ``dF/dz̃ = iΔF``, ``F = [Ex, Ey, Hx, Hy]``.

    Derived from Maxwell's equations with ``∂x → i·kx`` and ``∂y → 0`` by
    eliminating the longitudinal components: ``Ez = a·F`` from
    ``(εE)_z = −kx·Hy`` and ``Hz = b·F`` from ``(μH)_z = kx·Ey``.

    Args:
        eps: Permittivity tensor ``[..., 3, 3]``.
        mu: Permeability tensor ``[..., 3, 3]``.
        kx: Normalised in-plane wavevector ``[...]``.

    Returns:
        ``[..., 4, 4]``.
    """
    batch = jnp.broadcast_shapes(eps.shape[:-2], mu.shape[:-2], jnp.shape(kx))
    eps = jnp.broadcast_to(eps, batch + (3, 3))
    mu = jnp.broadcast_to(mu, batch + (3, 3))
    kx = jnp.broadcast_to(kx, batch).astype(eps.dtype)[..., None]
    zero = jnp.zeros_like(kx)

    def e(i, j):
        return eps[..., i, j][..., None]

    def m(i, j):
        return mu[..., i, j][..., None]

    unit = jnp.eye(4, dtype=eps.dtype)
    ex, ey, hx, hy = unit[0], unit[1], unit[2], unit[3]
    a = jnp.concatenate([-e(2, 0), -e(2, 1), zero, -kx], axis=-1) / e(2, 2)
    b = jnp.concatenate([zero, kx, -m(2, 0), -m(2, 1)], axis=-1) / m(2, 2)

    row_ex = kx * a + m(1, 0) * hx + m(1, 1) * hy + m(1, 2) * b
    row_ey = -(m(0, 0) * hx + m(0, 1) * hy + m(0, 2) * b)
    row_hx = kx * b - (e(1, 0) * ex + e(1, 1) * ey + e(1, 2) * a)
    row_hy = e(0, 0) * ex + e(0, 1) * ey + e(0, 2) * a
    return jnp.stack([row_ex, row_ey, row_hx, row_hy], axis=-2)


# ---------------------------------------------------------------------------
# Scattering matrices, carried as blocks (s00, s01, s10, s11)
# ---------------------------------------------------------------------------
#
# Inputs are the forward amplitude entering from the left and the backward
# amplitude entering from the right; outputs are the forward amplitude leaving
# right (s00 from left, s01 from right) and the backward amplitude leaving left
# (s10 from left, s11 from right). Each block is [..., n, n]: n = 2 over (p, s)
# for a planar stack, n = 2N over (p, s) x N harmonics in hyperbolic_optics.rcwa.


def _solve(a: jax.Array, b: jax.Array) -> jax.Array:
    """``a⁻¹·b`` with the batch dimensions of ``a`` and ``b`` broadcast."""
    batch = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return jnp.linalg.solve(
        jnp.broadcast_to(a, batch + a.shape[-2:]), jnp.broadcast_to(b, batch + b.shape[-2:])
    )


def _kz(eps: jax.Array, kx: jax.Array) -> jax.Array:
    """Normal wavevector of an isotropic medium, on the decaying (Im ≥ 0) branch."""
    kz = jnp.sqrt(eps - kx**2 + 0j)
    return jnp.where(jnp.imag(kz) < 0, -kz, kz)


def _isotropic_modes(eps: jax.Array, kx: jax.Array) -> jax.Array:
    """Mode columns ``[p→, s→, p←, s←]`` of an isotropic medium, ``[..., 4, 4]``.

    Unit total-field amplitudes on the NumPy engine's sign convention: the p
    waves share the sign of Hy, which gives ``r_p = (n₂c₁ − n₁c₂)/(n₂c₁ + n₁c₂)``.
    For real ε every mode carries normal flux ``±½ Re(kz)``.
    """
    n = jnp.sqrt(eps + 0j)
    kz = _kz(eps, kx)
    n = jnp.broadcast_to(n, kz.shape)
    zero, one = jnp.zeros_like(kz), jnp.ones_like(kz)
    columns = [
        [kz / n, zero, zero, n],
        [zero, one, -kz, zero],
        [-kz / n, zero, zero, n],
        [zero, one, kz, zero],
    ]
    return jnp.stack([jnp.stack(column, axis=-1) for column in columns], axis=-1)


def _reference_modes(kx: jax.Array) -> jax.Array:
    """Reference-medium modes: ε = 2kx² + 1, kz = √(1 + kx²), unit power.

    Always propagating (so passive intervals have norm-bounded scattering
    matrices) and impedance-matched to evanescent waves at large ``kx``.
    """
    kz_sq = 1.0 + kx**2
    return _isotropic_modes(kx**2 + kz_sq, kx) * (kz_sq**-0.25)[..., None, None]


def _blocks(matrix: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n = matrix.shape[-1] // 2
    return matrix[..., :n, :n], matrix[..., :n, n:], matrix[..., n:, :n], matrix[..., n:, n:]


def _interface(w_left: jax.Array, w_right: jax.Array):
    """Scattering matrix of the interface between two media, from tangential continuity."""
    w_left, w_right = jnp.broadcast_arrays(w_left, w_right)
    n = w_left.shape[-1] // 2
    lhs = jnp.concatenate([w_right[..., :n], -w_left[..., n:]], axis=-1)
    rhs = jnp.concatenate([w_left[..., :n], -w_right[..., n:]], axis=-1)
    return _blocks(jnp.linalg.solve(lhs, rhs))


def _star(left, right):
    """Redheffer star product: ``left`` on the incident side of ``right``."""
    a00, a01, a10, a11 = left
    b00, b01, b10, b11 = right
    eye = jnp.eye(a00.shape[-1], dtype=a00.dtype)
    loop = eye - a01 @ b10  # multiple reflections between the two
    return (
        b00 @ _solve(loop, a00),
        b01 + b00 @ _solve(loop, a01 @ b11),
        a10 + a11 @ b10 @ _solve(loop, a00),
        a11 @ _solve(eye - b10 @ a01, b11),
    )


def _layer(delta_ref: jax.Array, k0d: jax.Array, doublings: int):
    """Scattering matrix of one homogeneous layer in the reference basis.

    Args:
        delta_ref: The layer's ``Δ`` expressed in reference-mode amplitudes.
        k0d: Normalised thickness ``k₀·d`` ``[...]``.
        doublings: Number of doublings ``N``; the start slice is ``k₀d / 2ᴺ``.
    """
    size = delta_ref.shape[-1]
    eye2 = jnp.eye(size // 2, dtype=delta_ref.dtype)  # one block: (p, s) per harmonic
    eye4 = jnp.eye(size, dtype=delta_ref.dtype)
    z = 1j * (k0d / 2.0**doublings)[..., None, None] * delta_ref
    # exp(z) − I to fourth order, in Horner form; the identity is never added.
    step = z @ (eye4 + z @ (eye4 + z @ (eye4 + z / 4) / 3) / 2)
    e11, e12, e21, e22 = _blocks(step)

    # Transfer -> scattering for the slice. The transmissions are held as
    # increments x = s00 − I and y = s11 − I, both O(τ).
    k = _solve(eye2 + e22, jnp.broadcast_to(eye2, e22.shape))
    x = e11 - e12 @ k @ e21
    y = -k @ e22
    s01 = e12 @ k
    s10 = -k @ e21

    def double(_, carry):
        # S ⋆ S, rewritten on the increments using G − I = ½(G·P + P·G) for
        # G = (I − P)⁻¹, which keeps an O(τ) update from rounding away against I.
        x, y, s01, s10 = carry
        p, q = s01 @ s10, s10 @ s01
        g = _solve(eye2 - p, jnp.broadcast_to(eye2, p.shape))
        h = _solve(eye2 - q, jnp.broadcast_to(eye2, q.shape))
        s00, s11 = eye2 + x, eye2 + y
        xp, yq = x + p / 2, y + q / 2
        return (
            xp @ g + g @ xp + x @ g @ x,
            yq @ h + h @ yq + y @ h @ y,
            s01 + s00 @ g @ s01 @ s11,
            s10 + s11 @ s10 @ g @ s00,
        )

    x, y, s01, s10 = jax.lax.fori_loop(0, doublings, double, (x, y, s01, s10))
    return eye2 + x, s01, s10, eye2 + y


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def solve_stack(
    eps: jax.Array,
    thickness_um: jax.Array,
    wavelength_um: jax.Array,
    incidence_rad: jax.Array,
    eps_incident: jax.Array,
    eps_exit: jax.Array,
    mu: jax.Array | None = None,
    doublings: int = DEFAULT_DOUBLINGS,
) -> PlanarResult:
    """Reflection and transmission of a planar stack of anisotropic layers.

    Args:
        eps: Layer permittivity tensors in the lab frame, ``[..., L, 3, 3]``,
            ordered from the incident side. ``L`` may be zero.
        thickness_um: Layer thicknesses in µm, ``[..., L]``.
        wavelength_um: Vacuum wavelength in µm, ``[...]``. (From a frequency in
            cm⁻¹: ``1e4 / frequency``.)
        incidence_rad: Angle of incidence in the incident medium, radians.
        eps_incident: Real, positive permittivity of the incident half-space.
        eps_exit: Real, positive permittivity of the exit half-space.
        mu: Layer permeability tensors ``[..., L, 3, 3]``; identity if omitted.
        doublings: Doublings per layer (static). See :data:`DEFAULT_DOUBLINGS`.

    Returns:
        :class:`PlanarResult` with ``r``, ``t``, ``R``, ``T``, each
        ``[..., 2, 2]`` indexed ``[out, in]``, 0 = p and 1 = s. Wrap in
        ``jax.jit(..., static_argnames="doublings")`` for repeated calls.
    """
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "hyperbolic_optics.bcm is running in complex64 because 64-bit JAX is off; "
            'call jax.config.update("jax_enable_x64", True) for reference accuracy.',
            UserWarning,
            stacklevel=2,
        )
    complex_dtype = jnp.result_type(jnp.asarray(eps).dtype, jnp.complex64)
    eps = jnp.asarray(eps, dtype=complex_dtype)
    mu = (
        jnp.broadcast_to(jnp.eye(3, dtype=complex_dtype), eps.shape)
        if mu is None
        else jnp.asarray(mu, dtype=complex_dtype)
    )
    thickness_um = jnp.asarray(thickness_um)
    eps_incident = jnp.asarray(eps_incident)
    eps_exit = jnp.asarray(eps_exit)
    n_layers = eps.shape[-3]

    kx = jnp.sqrt(eps_incident) * jnp.sin(jnp.asarray(incidence_rad))
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    w_ref = _reference_modes(kx)

    scattering = _interface(_isotropic_modes(eps_incident, kx), w_ref)
    for index in range(n_layers):
        delta = system_matrix(eps[..., index, :, :], mu[..., index, :, :], kx)
        delta_ref = _solve(w_ref, delta @ w_ref)
        k0d = k0 * thickness_um[..., index]
        scattering = _star(scattering, _layer(delta_ref, k0d, doublings))
    scattering = _star(scattering, _interface(w_ref, _isotropic_modes(eps_exit, kx)))

    t, _, r, _ = scattering
    flux = jnp.real(_kz(eps_exit, kx)) / jnp.real(_kz(eps_incident, kx))
    return PlanarResult(r=r, t=t, R=jnp.abs(r) ** 2, T=jnp.abs(t) ** 2 * flux[..., None, None])
