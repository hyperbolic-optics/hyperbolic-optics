"""Differentiable planar solver: an eigendecomposition-free cascade in JAX.

The NumPy engine (:mod:`hyperbolic_optics.structure`) diagonalises each layer's
4×4 Berreman matrix, whose eigenvectors are awkward to differentiate (arbitrary
phase, divergent derivative at degeneracies, which every isotropic layer has).
Here each layer's scattering matrix is built from matrix products and linear
solves only, so ``jax.grad`` gives exact derivatives with respect to
thicknesses, tensors, orientations, angle and wavelength.

For a thin slice, ``exp(iΔτ) − I`` is a short Taylor series; ``2ᴺ`` slices are
assembled by ``N`` doublings, each a Redheffer star product of the interval
with itself (the precise integration method of Zhong & Williams 1994, as in
Yang, Song & Cai, arXiv:2608.06185). Intervals are carried as wave amplitudes
in a reference medium of permittivity ``2kx² + 1``, whose waves always
propagate and are scaled to unit power, so no doubling meets a singular inverse.

Coefficients are Jones matrices ``[..., out, in]`` with 0 = p and 1 = s, on the
NumPy engine's sign convention. The half-spaces are isotropic and lossless,
except in :func:`solve_stack_on_crystal`. Inputs broadcast over leading batch
dimensions. Reference accuracy needs ``jax.config.update("jax_enable_x64", True)``;
without it a warning says so.
"""

import warnings
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "PORTS",
    "PlanarResult",
    "CrystalResult",
    "PlanarSMatrix",
    "StackFields",
    "cascade",
    "rotate_tensor",
    "rotation_matrix",
    "solve_stack",
    "stack_fields",
    "solve_stack_on_crystal",
    "stack_smatrix",
    "system_matrix",
]

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


#: Port order of :class:`PlanarSMatrix`: each side of the stack, each polarization.
PORTS = ("top@p", "top@s", "bottom@p", "bottom@s")


class PlanarSMatrix(NamedTuple):
    """The full bidirectional scattering matrix of a planar stack.

    Attributes:
        s: ``[..., 4, 4]`` indexed ``[out, in]`` over :data:`PORTS`. ``top`` is
            the incident-side half-space, ``bottom`` the exit side; p and s are
            defined in each half-space's own plane of incidence (x-z). Reference
            planes are the stack's two outer interfaces.
        propagating: ``[..., 4]``, whether each port's wave propagates (it is
            evanescent in a half-space whose index is below ``kx``).

    Amplitudes are scaled by ``√kz`` of their half-space, so between propagating
    ports ``|s|²`` is the fraction of power transferred, and a lossless stack
    has a unitary ``s``. Reciprocal media (symmetric ε and μ) give
    ``s(kx) = D · s(−kx)ᵀ · D`` with ``D = diag(−1, 1, −1, 1)``: reversing a wave
    reverses its in-plane wavevector, and the p waves' sign convention (they
    share the sign of Hy) flips on reversal. For an evanescent port the same scaling still applies
    (with complex ``kz``), so matrices of stacks that share a half-space
    compose correctly through it -- frustrated total internal reflection
    included -- but ``|s|²`` is not a power there.
    """

    s: jax.Array
    propagating: jax.Array


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def _check_x64() -> None:
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "hyperbolic_optics is running in complex64 because 64-bit JAX is off; "
            'call jax.config.update("jax_enable_x64", True) for reference accuracy.',
            UserWarning,
            stacklevel=3,
        )


def _materials(eps, mu):
    """``eps`` and ``mu`` as complex arrays, ``mu`` the identity if omitted."""
    dtype = jnp.result_type(jnp.asarray(eps).dtype, jnp.complex64)
    eps = jnp.asarray(eps, dtype=dtype)
    if mu is None:
        return eps, jnp.broadcast_to(jnp.eye(3, dtype=dtype), eps.shape)
    return eps, jnp.asarray(mu, dtype=dtype)


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


def _longitudinal(eps, mu, kx):
    """Rows ``a``, ``b`` with ``Ez = a·F`` and ``Hz = b·F``, the longitudinal fields."""
    batch = jnp.broadcast_shapes(eps.shape[:-2], mu.shape[:-2], jnp.shape(kx))
    eps = jnp.broadcast_to(eps, batch + (3, 3))
    mu = jnp.broadcast_to(mu, batch + (3, 3))
    kx = jnp.broadcast_to(kx, batch)[..., None].astype(eps.dtype)
    zero = jnp.zeros_like(kx)
    a = jnp.concatenate([-eps[..., 2, 0:1], -eps[..., 2, 1:2], zero, -kx], -1) / eps[..., 2, 2:3]
    b = jnp.concatenate([zero, kx, -mu[..., 2, 0:1], -mu[..., 2, 1:2]], -1) / mu[..., 2, 2:3]
    return a, b


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
    a, b = _longitudinal(eps, mu, kx)
    kx = jnp.broadcast_to(kx, batch).astype(eps.dtype)[..., None]

    def e(i, j):
        return eps[..., i, j][..., None]

    def m(i, j):
        return mu[..., i, j][..., None]

    unit = jnp.eye(4, dtype=eps.dtype)
    ex, ey, hx, hy = unit[0], unit[1], unit[2], unit[3]

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


def _isotropic_modes(eps: jax.Array, kx: jax.Array, kz: jax.Array | None = None) -> jax.Array:
    """Mode columns ``[p→, s→, p←, s←]`` of an isotropic medium, ``[..., 4, 4]``.

    Unit total-field amplitudes on the NumPy engine's sign convention: the p
    waves share the sign of Hy, which gives ``r_p = (n₂c₁ − n₁c₂)/(n₂c₁ + n₁c₂)``.
    For real ε every mode carries normal flux ``±½ Re(kz)``. ``kz`` overrides
    the default decaying branch when analytically continuing a guided mode.
    """
    n = jnp.sqrt(eps + 0j)
    kz = _kz(eps, kx) if kz is None else kz
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


@partial(jax.jit, static_argnames="doublings")  # compiled once per shape, not per call
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
    _check_x64()
    kx = jnp.sqrt(jnp.asarray(eps_incident)) * jnp.sin(jnp.asarray(incidence_rad))
    t, _, r, _ = _scattering(
        eps, thickness_um, wavelength_um, kx, eps_incident, eps_exit, mu, doublings
    )
    flux = jnp.real(_kz(eps_exit, kx)) / jnp.real(_kz(eps_incident, kx))
    return PlanarResult(r=r, t=t, R=jnp.abs(r) ** 2, T=jnp.abs(t) ** 2 * flux[..., None, None])


def _scattering(eps, thickness_um, wavelength_um, kx, eps_incident, eps_exit, mu, doublings):
    """Scattering blocks between the half-spaces' mode amplitudes, at in-plane ``kx``."""
    kx = jnp.asarray(kx)
    w_top = _isotropic_modes(jnp.asarray(eps_incident), kx)
    w_bottom = _isotropic_modes(jnp.asarray(eps_exit), kx)
    return _cascade(eps, mu, thickness_um, wavelength_um, kx, w_top, w_bottom, doublings)


def _deltas(eps, mu, kx, w_ref) -> list:
    """Each layer's ``Δ`` in reference-mode amplitudes, from the top down."""
    return [
        _solve(w_ref, system_matrix(eps[..., i, :, :], mu[..., i, :, :], kx) @ w_ref)
        for i in range(eps.shape[-3])
    ]


def _cascade(eps, mu, thickness_um, wavelength_um, kx, w_top, w_bottom, doublings):
    """The layers between two half-spaces' mode bases, as scattering blocks."""
    eps, mu = _materials(eps, mu)
    thickness_um = jnp.asarray(thickness_um)
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    w_ref = _reference_modes(kx)
    scattering = _interface(w_top, w_ref)
    for index, delta in enumerate(_deltas(eps, mu, kx, w_ref)):
        scattering = _star(scattering, _layer(delta, k0 * thickness_um[..., index], doublings))
    return _star(scattering, _interface(w_ref, w_bottom))


def stack_smatrix(
    eps: jax.Array,
    thickness_um: jax.Array,
    wavelength_um: jax.Array,
    kx: jax.Array,
    eps_top: jax.Array,
    eps_bottom: jax.Array,
    mu: jax.Array | None = None,
    doublings: int = DEFAULT_DOUBLINGS,
) -> PlanarSMatrix:
    """The bidirectional, power-normalized scattering matrix of a planar stack.

    Where :func:`solve_stack` gives the response to light from the incident
    side, this gives all four ports: light from either side, in either
    polarization, as needed to connect stacks to one another (for instance in a
    SAX circuit, :mod:`hyperbolic_optics.integrations.sax`).

    Args:
        eps: Layer permittivity tensors ``[..., L, 3, 3]``, from the top down.
        thickness_um: Layer thicknesses in µm, ``[..., L]``.
        wavelength_um: Vacuum wavelength in µm.
        kx: In-plane wavevector over k₀ (``n·sin θ``, conserved through the
            stack), rather than an angle, so stacks sharing a half-space share
            it too.
        eps_top, eps_bottom: Real, positive permittivities of the half-spaces.
        mu: Layer permeability tensors; identity if omitted.
        doublings: Doublings per layer (static).

    Returns:
        :class:`PlanarSMatrix` over :data:`PORTS`.
    """
    _check_x64()
    kx = jnp.asarray(kx)
    s00, s01, s10, s11 = _scattering(
        eps, thickness_um, wavelength_um, kx, eps_top, eps_bottom, mu, doublings
    )
    return _normalize_smatrix((s00, s01, s10, s11), kx, eps_top, eps_bottom)


def _normalize_smatrix(blocks, kx, eps_top, eps_bottom, bottom_propagating=None):
    """The four-port matrix of scattering ``blocks``, power-normalised (complex √kz
    on evanescent ports).

    ``bottom_propagating`` (``[..., 2]``) marks bottom ports whose modes are
    already power-normalised (a crystal's, from :mod:`hyperbolic_optics.rcwa2d`).
    """
    s00, s01, s10, s11 = blocks
    raw = jnp.concatenate(
        [jnp.concatenate([s10, s11], axis=-1), jnp.concatenate([s00, s01], axis=-1)], axis=-2
    )  # rows: out at top, out at bottom; columns: in at top, in at bottom
    kz_top = _kz(jnp.asarray(eps_top), kx) + 0j
    crystal = bottom_propagating is not None
    kz_bottom = jnp.ones_like(kz_top) if crystal else _kz(jnp.asarray(eps_bottom), kx) + 0j
    kz = jnp.stack(jnp.broadcast_arrays(kz_top, kz_top, kz_bottom, kz_bottom), -1)
    tolerance = 1e-12
    # at exact cutoff (grazing, kz = 0) a port carries no power: leave it unscaled
    cutoff = jnp.abs(kz) < tolerance
    root = jnp.where(cutoff, 1.0, jnp.sqrt(jnp.where(cutoff, 1.0, kz)))
    s = raw * root[..., :, None] / root[..., None, :]
    propagating = (jnp.abs(jnp.imag(kz)) < tolerance) & ~cutoff
    propagating = jnp.broadcast_to(propagating, s.shape[:-1])
    if crystal:
        bottom = jnp.broadcast_to(bottom_propagating, s.shape[:-2] + (2,))
        propagating = jnp.concatenate([propagating[..., :2], bottom], axis=-1)
    return PlanarSMatrix(s=s, propagating=propagating)


def cascade(upper: PlanarSMatrix, lower: PlanarSMatrix) -> PlanarSMatrix:
    """Connect ``upper``'s bottom port to ``lower``'s top port.

    The two must share that half-space (same permittivity, same ``kx``): the
    result is the stack of both, with the shared medium of zero thickness --
    what a SAX circuit computes, without SAX. Evanescent channels in the shared
    medium are carried, so a gap below total internal reflection tunnels.
    """

    def blocks(m):
        s = m.s
        return s[..., :2, :2], s[..., 2:, :2], s[..., :2, 2:], s[..., 2:, 2:]

    a_tt, a_bt, a_tb, a_bb = blocks(upper)  # [out side][in side]: t = top, b = bottom
    b_tt, b_bt, b_tb, b_bb = blocks(lower)
    eye = jnp.eye(2, dtype=a_tt.dtype)
    down = _solve(eye - b_tt @ a_bb, b_tt @ a_bt)  # lower's reflection seen from above
    up = _solve(eye - a_bb @ b_tt, a_bb)
    top_top = a_tt + a_tb @ down
    bottom_top = b_bt @ _solve(eye - a_bb @ b_tt, a_bt)
    top_bottom = a_tb @ _solve(eye - b_tt @ a_bb, b_tb)
    bottom_bottom = b_bb + b_bt @ up @ b_tb
    s = jnp.concatenate(
        [
            jnp.concatenate([top_top, top_bottom], axis=-1),
            jnp.concatenate([bottom_top, bottom_bottom], axis=-1),
        ],
        axis=-2,
    )
    propagating = jnp.concatenate([upper.propagating[..., :2], lower.propagating[..., 2:]], axis=-1)
    return PlanarSMatrix(s=s, propagating=propagating)


# ---------------------------------------------------------------------------
# Fields and absorption
# ---------------------------------------------------------------------------


class StackFields(NamedTuple):
    """Fields and power flow through a planar stack, for p and s light from the top.

    The last axis of every array is the incident polarization (0 = p, 1 = s),
    each of unit incident power.

    Attributes:
        R, T: ``[..., 2]`` total reflectance and transmittance (summed over the
            outgoing polarizations).
        absorption: ``[..., L, 2]`` fraction of the incident power absorbed in
            each layer.
        interface_flux: ``[..., L + 1, 2]`` normal energy flux ``S_z`` at each
            interface, from the top of the first layer down, over the incident
            flux: ``1 − R`` at the top and ``T`` at the bottom.
        z_um: ``[..., L·S]`` depths of the field samples below the top of the
            first layer (empty unless ``samples_per_layer`` was given).
        fields: ``[..., L·S, 6, 2]`` complex ``(Ex, Ey, Ez, Hx, Hy, Hz)`` at those
            depths, H scaled by the free-space impedance.
    """

    R: jax.Array
    T: jax.Array
    absorption: jax.Array
    interface_flux: jax.Array
    z_um: jax.Array
    fields: jax.Array


def _poynting_z(field):
    """``S_z = ½ Re(Ex·Hy* − Ey·Hx*)`` of tangential fields ``[..., 4, n]``."""
    ex, ey, hx, hy = (field[..., i, :] for i in range(4))
    return 0.5 * jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))


def _plane_fields(above, below, w_ref):
    """Tangential fields at a plane from the scattering matrices above and below it.

    With unit p and s amplitudes incident from the top, the downward amplitude
    ``f`` at the plane satisfies ``f = A₀₀ + A₀₁·B₁₀·f``: every matrix involved is
    bounded, so no growing exponential enters.
    """
    a00, a01, _, _ = above
    _, _, b10, _ = below
    eye = jnp.eye(a00.shape[-1], dtype=a00.dtype)
    down = _solve(eye - a01 @ b10, a00)  # [..., 2, 2]: columns are p and s in
    up = b10 @ down
    return w_ref @ jnp.concatenate([down, up], axis=-2)  # [..., 4, 2]


def stack_fields(
    eps: jax.Array,
    thickness_um: jax.Array,
    wavelength_um: jax.Array,
    incidence_rad: jax.Array,
    eps_incident: jax.Array,
    eps_exit: jax.Array,
    mu: jax.Array | None = None,
    samples_per_layer: int = 0,
    doublings: int = DEFAULT_DOUBLINGS,
) -> StackFields:
    """Fields, interface fluxes and layer-resolved absorption of a planar stack.

    The fields at each plane come from the scattering matrices of the stack
    above and below it, never from transfer matrices, so thick, evanescent and
    strongly absorbing layers stay stable. Everything is differentiable,
    including with respect to thicknesses: the samples inside a layer sit at
    fixed fractions of its thickness.

    Args:
        samples_per_layer: Field samples per layer (static), evenly spaced from
            its top to its bottom, both included. ``0`` skips the depth profile.

    The other arguments are as in :func:`solve_stack`.

    Returns:
        :class:`StackFields`.
    """
    _check_x64()
    eps, mu = _materials(eps, mu)
    thickness_um = jnp.asarray(thickness_um)
    eps_incident, eps_exit = jnp.asarray(eps_incident), jnp.asarray(eps_exit)
    n_layers = eps.shape[-3]
    kx = jnp.sqrt(eps_incident) * jnp.sin(jnp.asarray(incidence_rad))
    k0 = 2 * jnp.pi / jnp.asarray(wavelength_um)
    w_ref = _reference_modes(kx)

    deltas = _deltas(eps, mu, kx, w_ref)
    layers = [_layer(deltas[i], k0 * thickness_um[..., i], doublings) for i in range(n_layers)]
    above = [_interface(_isotropic_modes(eps_incident, kx), w_ref)]
    for layer in layers:
        above.append(_star(above[-1], layer))
    below = [_interface(w_ref, _isotropic_modes(eps_exit, kx))]
    for layer in reversed(layers):
        below.insert(0, _star(layer, below[0]))

    incident_flux = 0.5 * jnp.real(_kz(eps_incident, kx))[..., None]
    planes = [_plane_fields(above[j], below[j], w_ref) for j in range(n_layers + 1)]
    flux = jnp.stack([_poynting_z(f) for f in planes], axis=-2) / incident_flux[..., None, :]
    absorption = flux[..., :-1, :] - flux[..., 1:, :]

    batch = flux.shape[:-2]
    z_um = jnp.zeros(batch + (0,))
    fields = jnp.zeros(batch + (0, 6, 2), dtype=eps.dtype)
    if samples_per_layer and n_layers:
        fractions = jnp.linspace(0.0, 1.0, samples_per_layer)
        tops = jnp.cumsum(thickness_um, axis=-1) - thickness_um
        depths, samples = [], []
        for i in range(n_layers):
            k0d = (k0 * thickness_um[..., i])[..., None]
            delta = deltas[i][..., None, :, :]
            upper = _layer(delta, k0d * fractions, doublings)
            lower = _layer(delta, k0d * (1 - fractions), doublings)

            def expand(blocks):
                return tuple(block[..., None, :, :] for block in blocks)

            tangential = _plane_fields(
                _star(expand(above[i]), upper), _star(lower, expand(below[i + 1])),
                w_ref[..., None, :, :],
            )  # fmt: skip
            a, b = _longitudinal(eps[..., i, :, :], mu[..., i, :, :], kx)
            ez = jnp.einsum("...j,...sjn->...sn", a, tangential)
            hz = jnp.einsum("...j,...sjn->...sn", b, tangential)
            full = jnp.stack(
                [tangential[..., 0, :], tangential[..., 1, :], ez,
                 tangential[..., 2, :], tangential[..., 3, :], hz],
                axis=-2,
            )  # fmt: skip
            samples.append(full)
            depths.append(tops[..., i, None] + thickness_um[..., i, None] * fractions)
        fields = jnp.concatenate(jnp.broadcast_arrays(*samples), axis=-3)
        z_um = jnp.concatenate(jnp.broadcast_arrays(*depths), axis=-1)

    return StackFields(
        R=1.0 - flux[..., 0, :],
        T=flux[..., -1, :],
        absorption=absorption,
        interface_flux=flux,
        z_um=z_um,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Semi-infinite anisotropic substrates
# ---------------------------------------------------------------------------


class CrystalResult(NamedTuple):
    """Response of a stack on a semi-infinite anisotropic (or lossy) substrate.

    Attributes:
        r: ``[..., 2, 2]`` reflection Jones matrix ``[out, in]`` (0 = p, 1 = s),
            on the same convention as :func:`solve_stack`.
        R: ``|r|²``.
        T: ``[..., 2]`` total power carried into the substrate, per incident
            polarization, from the normal Poynting flux just inside it
            (``1 − R − T`` is absorbed in the layers).
    """

    r: jax.Array
    R: jax.Array
    T: jax.Array


def _matrix_sign(m: jax.Array, iterations: int) -> jax.Array:
    """``sign(m)`` by the scaled Newton iteration ``X ← (X/c + c·X⁻¹)/2``."""
    eye = jnp.eye(m.shape[-1], dtype=m.dtype)

    def step(x, _):
        inverse = _solve(x, jnp.broadcast_to(eye, x.shape))
        # determinant scaling speeds the early steps
        scale = jnp.abs(jnp.linalg.det(x)) ** (-1.0 / x.shape[-1])
        scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.0)[..., None, None]
        return 0.5 * (scale * x + inverse / scale), None

    x, _ = jax.lax.scan(step, m, None, length=iterations)
    return x


def _sylvester(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """``X`` with ``a·X − X·b = c`` for small square blocks (a Kronecker solve)."""
    n, m = a.shape[-1], b.shape[-1]
    eye_n, eye_m = jnp.eye(n, dtype=a.dtype), jnp.eye(m, dtype=a.dtype)
    kron = jnp.einsum("ij,...kl->...ikjl", eye_m, a).reshape(a.shape[:-2] + (n * m, n * m))
    kron = kron - jnp.einsum("...ji,kl->...ikjl", b, eye_n).reshape(kron.shape)
    vec = jnp.swapaxes(c, -1, -2).reshape(c.shape[:-2] + (n * m,))
    x = jnp.linalg.solve(kron, vec[..., None])[..., 0]
    return jnp.swapaxes(x.reshape(c.shape[:-2] + (m, n)), -1, -2)


def _invariant(delta: jax.Array, basis: jax.Array, complement: jax.Array, steps: int = 3):
    """The invariant subspace of ``delta`` nearest ``span(basis)``, by Newton (Riccati).

    In the coordinates of ``Q = qr([basis, complement])`` the subspace is
    ``span [I; Z]``, with ``A₂₁ + A₂₂Z − ZA₁₁ − ZA₁₂Z = 0``; each Newton step is a
    Sylvester solve, and converges quadratically from a close start.
    """
    q, _ = jnp.linalg.qr(jnp.concatenate([basis, complement], axis=-1))
    a = jnp.conj(jnp.swapaxes(q, -1, -2)) @ delta @ q
    k = basis.shape[-1]
    a11, a12, a21, a22 = a[..., :k, :k], a[..., :k, k:], a[..., k:, :k], a[..., k:, k:]
    z = jnp.zeros_like(a21)
    for _ in range(steps):
        residual = a21 + a22 @ z - z @ a11 - z @ a12 @ z
        z = z + _sylvester(a22 - z @ a12, a11 + a12 @ z, -residual)
    eye = jnp.broadcast_to(jnp.eye(k, dtype=a.dtype), a11.shape)
    return q @ jnp.concatenate([eye, z], axis=-2)


def _substrate_modes(eps, mu, kx, loss: float, iterations: int):
    """Mode columns ``[forward ×2, backward ×2]`` spanning the substrate's two subspaces.

    The forward (into the substrate) subspace is that of the waves that decay
    into it: the range of ``(I + sign(−iΔ))/2``. A lossless substrate's
    propagating waves are assigned by a small causal loss, ``loss·|ε|``: their
    energy flows in, which for a hyperbolic crystal need not be the sense of
    ``kz``. The subspaces so found are then refined to those of the lossless
    ``Δ`` itself (:func:`_invariant`), so the result is exact to rounding. No
    eigenvector is formed, and degenerate modes (a uniaxial axis along z)
    cost nothing.
    """
    scale = jnp.max(jnp.abs(eps), axis=(-2, -1), keepdims=True)
    causal = eps + 1j * loss * scale * jnp.eye(3)
    # A positive reference remains nonsingular when a hyperbolic tensor's trace
    # vanishes, or its mean permittivity happens to put the guess at cutoff.
    return _split_modes(
        system_matrix(eps, mu, kx), system_matrix(causal, mu, kx), _reference_modes(kx), iterations
    )


def _split_modes(delta, delta_causal, guess, iterations):
    """Columns ``[forward ×n, backward ×n]`` of ``Δ``'s two invariant subspaces.

    The forward subspace is the range of ``(I + sign(−iΔ_causal))/2``, seeded
    from ``guess``'s columns (forward first) and refined to ``Δ`` itself.
    """
    sign = _matrix_sign(-1j * delta_causal, iterations)
    eye = jnp.eye(delta.shape[-1], dtype=sign.dtype)
    forward, backward = 0.5 * (eye + sign), 0.5 * (eye - sign)
    half = delta.shape[-1] // 2
    start_forward = forward @ guess[..., :, :half]
    start_backward = backward @ guess[..., :, half:]
    columns = jnp.concatenate(
        [
            _invariant(delta, start_forward, start_backward),
            _invariant(delta, start_backward, start_forward),
        ],
        axis=-1,
    )
    return columns / jnp.linalg.norm(columns, axis=-2, keepdims=True)


def solve_stack_on_crystal(
    eps: jax.Array,
    thickness_um: jax.Array,
    wavelength_um: jax.Array,
    incidence_rad: jax.Array,
    eps_incident: jax.Array,
    eps_substrate: jax.Array,
    mu: jax.Array | None = None,
    mu_substrate: jax.Array | None = None,
    doublings: int = DEFAULT_DOUBLINGS,
) -> CrystalResult:
    """A planar stack on a semi-infinite anisotropic, lossy or hyperbolic substrate.

    The NumPy engine's ``Semi Infinite Anisotropic Layer``, in the
    differentiable solver: bulk crystals (calcite, sapphire, hBN, SiC,
    α-MoO₃, a LiNbO₃ wafer) as substrates, in any orientation. The waves the
    substrate carries away are found without eigendecomposition, by a matrix
    sign function (see :func:`_substrate_modes`), so reflection is exact and
    differentiable even where the substrate's modes are degenerate.

    The other arguments are as in :func:`solve_stack` (``L`` may be zero: a
    bare crystal).

    Args:
        eps_substrate: The substrate's tensor ``[..., 3, 3]`` in the lab frame
            (rotate it with :func:`rotate_tensor`); lossy is fine.
        mu_substrate: Its permeability; identity if omitted.

    Returns:
        :class:`CrystalResult`: ``r``, ``R`` and the power transmitted into
        the substrate ``T``.
    """
    _check_x64()
    eps_substrate, mu_substrate = _materials(eps_substrate, mu_substrate)
    eps_incident = jnp.asarray(eps_incident)
    kx = jnp.sqrt(eps_incident) * jnp.sin(jnp.asarray(incidence_rad))
    # a lossless substrate gets a relative loss of 1e-8 to decide which waves carry energy
    # in; the subspaces are then refined to the lossless ones, so it costs no accuracy
    w_sub = _substrate_modes(eps_substrate, mu_substrate, kx, 1e-8, 60)
    w_in = _isotropic_modes(eps_incident, kx)
    t, _, r, _ = _cascade(eps, mu, thickness_um, wavelength_um, kx, w_in, w_sub, doublings)
    flux = _poynting_z(w_sub[..., :, :2] @ t)  # into the substrate, per input
    incident = 0.5 * jnp.real(_kz(eps_incident, kx))[..., None]
    return CrystalResult(r=r, R=jnp.abs(r) ** 2, T=flux / incident)
