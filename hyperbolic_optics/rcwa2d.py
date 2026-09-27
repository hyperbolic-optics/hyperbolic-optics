"""Differentiable RCWA for 2D-periodic (crossed) gratings of full-tensor media.

Layers periodic in the plane on any Bravais lattice -- square, rectangular,
hexagonal (triangular) or oblique (:class:`Lattice`) -- and layered in z. Each
patterned layer is a background material and non-overlapping polygons, each
with its own 3×3 permittivity and permeability -- any orientation. Circles and
rectangles are polygons (:func:`ellipse`, :func:`rectangle`), and a GDS
polygon is one directly.

Geometry enters through the exact Fourier transform of each polygon, a closed
sum over its edges (the divergence theorem), so every coefficient is exact and
differentiable with respect to every vertex -- a fill factor, a radius or a
polygon exported from a layout. Layers are solved by the eigendecomposition-free
cascade of :mod:`hyperbolic_optics.bcm` on ``4N × 4N`` Fourier-space operators
for ``N`` harmonics ``G = m·b₁ + n·b₂`` of the reciprocal lattice: all
``|m| ≤ M₁, |n| ≤ M₂`` (``truncation="parallelogram"``), or every ``G`` with
``|G| ≤ max(M₁, M₂)·g_min`` for the shortest reciprocal vector ``g_min``
(``"circular"``), which depends only on the lattice, not on the primitive
vectors chosen to describe it.

Fourier factorisation
---------------------
``factorization="laurent"`` uses Toeplitz matrices of ε directly: correct, but
slow to converge where the normal component of E jumps at an edge.

``factorization="normal"`` (default) is the normal-vector method generalised to
full tensors. At every point the constitutive relation is written in a local
frame ``(n, t, z)`` whose ``n`` is normal to the pattern's edges, and Li's
anisotropic rearrangement (see :mod:`hyperbolic_optics.rcwa`) is applied there:
``(E_n, D_t, D_z)`` in terms of the edge-continuous ``(D_n, E_t, E_z)``, with
the inverse rule on the normal component only. For isotropic media this is the
textbook normal-vector method; for stripes, ``n = x̂`` and it is the 1D solver's
Li rule exactly. The normal field is the gradient direction of the pattern
smoothed over a tenth of the cell.

Coefficient functions that depend on the frame are products of a shape's
indicator -- whose coefficients stay exact -- with a smooth function of the
frame, whose coefficients come from a fine grid; the product is a convolution
in Fourier space. The assembly is averaged with its product-reversed twin,
which keeps it Hermitian for lossless media: energy is conserved to rounding at
every truncation.

Convergence, for orientation (the user guide has the comparisons): at 529
harmonics, silicon pillars are within 3e-3 (square) and 3e-4 (disc) of FMMAX's
converged reflectance, against 1e-2 and 4e-3 for Laurent's rule, and a
birefringent disc within 5e-5 in every channel. Metal-like and hyperbolic
inclusions (Re ε < 0) converge slowly under every Fourier factorisation, this
one included.

Conventions follow :mod:`hyperbolic_optics.rcwa`: order ``(m, n)`` has in-plane
wavevector ``k∥₀ + (m·b₁ + n·b₂)/k₀`` (on a rectangular lattice,
``(kx₀ + m·λ/Λx, ky₀ + n·λ/Λy)``), resolved into that order's own p and s;
results are ``[..., order, out, in]`` with 0 = p and 1 = s.
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics.bcm import _interface, _layer, _solve, _star
from hyperbolic_optics.rcwa import _fourier_system_matrix, _harmonic_modes, default_doublings

__all__ = [
    "CrossedGratingResult",
    "Lattice",
    "PatternedLayer",
    "Shape",
    "ellipse",
    "polygon_fourier",
    "rectangle",
    "solve_crossed_grating",
    "uniform",
]


class Lattice(NamedTuple):
    """A 2D Bravais lattice: primitive vectors ``a1``, ``a2`` as ``(x, y)`` in µm.

    Use the constructors: :meth:`rectangular`, :meth:`square`, :meth:`hexagonal`
    (the triangular lattice) and :meth:`oblique`. A ``(Λx, Λy)`` tuple passed as
    a period means :meth:`rectangular`. Components may be JAX values (to
    optimise the lattice) except under ``truncation="circular"``.
    """

    a1: tuple
    a2: tuple

    @classmethod
    def rectangular(cls, period_x, period_y) -> "Lattice":
        return cls((period_x, 0.0), (0.0, period_y))

    @classmethod
    def square(cls, period) -> "Lattice":
        return cls.rectangular(period, period)

    @classmethod
    def hexagonal(cls, period, rotation_rad=0.0) -> "Lattice":
        """Triangular lattice of nearest-neighbour distance ``period``; ``a1`` along x, rotated."""
        c, s = jnp.cos(rotation_rad), jnp.sin(rotation_rad)
        h = jnp.sqrt(3.0) / 2

        def turn(x, y):
            return (period * (c * x - s * y), period * (s * x + c * y))

        return cls(turn(1.0, 0.0), turn(0.5, h))

    @classmethod
    def oblique(cls, a, b, angle_rad) -> "Lattice":
        """``|a1| = a`` along x and ``|a2| = b`` at ``angle_rad`` from it."""
        return cls((a, 0.0), (b * jnp.cos(angle_rad), b * jnp.sin(angle_rad)))


def _as_lattice(period_um) -> Lattice:
    if isinstance(period_um, Lattice):
        return period_um
    return Lattice.rectangular(*period_um)


def _vectors(lattice: Lattice):
    """``(a1, a2, b1, b2, area)``: primitive and reciprocal vectors, cell area."""
    a1 = jnp.asarray(lattice.a1, dtype=float)
    a2 = jnp.asarray(lattice.a2, dtype=float)
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = 2 * jnp.pi * jnp.stack([a2[1], -a2[0]]) / cross
    b2 = 2 * jnp.pi * jnp.stack([-a1[1], a1[0]]) / cross
    return a1, a2, b1, b2, jnp.abs(cross)


class Shape(NamedTuple):
    """A polygon of one material within the unit cell.

    Attributes:
        vertices: ``[..., V, 2]`` in µm, in cell coordinates (either winding).
            Shapes may cross the cell edge; they are periodised.
        eps: Permittivity tensor in the lab frame, ``[..., 3, 3]``.
        mu: Permeability tensor, or ``None`` for μ = 1.
    """

    vertices: jax.Array
    eps: jax.Array
    mu: jax.Array | None = None


class PatternedLayer(NamedTuple):
    """One layer: a background material and non-overlapping :class:`Shape` s."""

    thickness_um: jax.Array
    background: jax.Array
    shapes: tuple[Shape, ...] = ()
    background_mu: jax.Array | None = None


class CrossedGratingResult(NamedTuple):
    """Amplitudes and efficiencies ``[..., order, out, in]``.

    ``orders`` is ``[N, 2]``, a NumPy array (static, so it can index inside ``jit``).
    """

    orders: np.ndarray
    r: jax.Array
    t: jax.Array
    R: jax.Array
    T: jax.Array


def uniform(eps, thickness_um, mu=None) -> PatternedLayer:
    """An unpatterned layer."""
    return PatternedLayer(jnp.asarray(thickness_um), jnp.asarray(eps), (), mu)


def rectangle(center, size) -> jax.Array:
    """Vertices ``[4, 2]`` of an axis-aligned rectangle (differentiable in both)."""
    cx, cy = center
    hx, hy = size[0] / 2, size[1] / 2
    return jnp.stack(
        [
            jnp.stack([cx - hx, cy - hy]),
            jnp.stack([cx + hx, cy - hy]),
            jnp.stack([cx + hx, cy + hy]),
            jnp.stack([cx - hx, cy + hy]),
        ]
    )


def ellipse(center, radii, n_vertices: int = 64, rotation_rad=0.0) -> jax.Array:
    """Vertices ``[n, 2]`` of an ellipse, as a polygon of equal area.

    The radius is scaled so the polygon's area equals the ellipse's, which
    removes the leading-order polygon error in the zeroth Fourier coefficient.
    """
    angle = 2 * jnp.pi * jnp.arange(n_vertices) / n_vertices
    scale = jnp.sqrt(2 * jnp.pi / (n_vertices * jnp.sin(2 * jnp.pi / n_vertices)))
    x = radii[0] * scale * jnp.cos(angle)
    y = radii[1] * scale * jnp.sin(angle)
    c, s = jnp.cos(rotation_rad), jnp.sin(rotation_rad)
    return jnp.stack([center[0] + c * x - s * y, center[1] + s * x + c * y], axis=-1)


# ---------------------------------------------------------------------------
# Exact Fourier coefficients of polygons
# ---------------------------------------------------------------------------


def polygon_fourier(vertices: jax.Array, gx: jax.Array, gy: jax.Array) -> jax.Array:
    """``∫_P exp(−i(gx·x + gy·y)) dA`` over a polygon, exactly.

    By the divergence theorem, ``(i/|g|²) Σ_edges (g·N_j) ∫_edge exp(−i g·r) dl``
    with ``N_j`` the edge's outward normal scaled by its length; each edge
    integral is closed-form. ``g = 0`` gives the area.

    Args:
        vertices: ``[..., V, 2]`` (either winding).
        gx, gy: Wavevector components ``[K]``.

    Returns:
        ``[..., K]``.
    """
    v = vertices
    w = jnp.roll(v, -1, axis=-2)
    e = w - v  # [..., V, 2]
    signed_area = 0.5 * jnp.sum(v[..., 0] * w[..., 1] - w[..., 0] * v[..., 1], axis=-1)
    winding = jnp.sign(signed_area)[..., None]  # +1 counter-clockwise

    phase = v[..., :, 0, None] * gx + v[..., :, 1, None] * gy  # [..., V, K]
    along = e[..., :, 0, None] * gx + e[..., :, 1, None] * gy
    small = jnp.abs(along) < 1e-6
    safe = jnp.where(small, 1.0, along)
    segment = jnp.where(
        small,
        1.0 - 0.5j * along - along**2 / 6.0,
        (1.0 - jnp.exp(-1j * along)) / (1j * safe),
    )
    normal = e[..., :, 1, None] * gx - e[..., :, 0, None] * gy  # g · (e_y, −e_x)
    total = jnp.sum(normal * jnp.exp(-1j * phase) * segment, axis=-2)
    g_sq = gx**2 + gy**2
    zero = g_sq == 0
    transform = 1j * total / jnp.where(zero, 1.0, g_sq)
    return jnp.where(zero, jnp.abs(signed_area)[..., None] + 0j, winding * transform)


def _grid_wavevectors(k1_max: int, k2_max: int, lattice: Lattice):
    """``G = m·b₁ + n·b₂`` for ``|m| ≤ k1_max, |n| ≤ k2_max``, as ``[2K₁+1, 2K₂+1]`` grids."""
    _, _, b1, b2, _ = _vectors(lattice)
    m = jnp.arange(-k1_max, k1_max + 1)[:, None]
    n = jnp.arange(-k2_max, k2_max + 1)[None, :]
    return m * b1[0] + n * b2[0], m * b1[1] + n * b2[1]


def _indicators(layer: PatternedLayer, k1_max: int, k2_max: int, lattice) -> list[jax.Array]:
    """Exact coefficients of each shape's indicator on ``|m| ≤ k1_max, |n| ≤ k2_max``."""
    gx, gy = _grid_wavevectors(k1_max, k2_max, lattice)
    shape = gx.shape
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    area = _vectors(lattice)[4]
    return [
        (polygon_fourier(jnp.asarray(s.vertices), gx, gy) / area).reshape(
            jnp.shape(s.vertices)[:-2] + shape
        )
        for s in layer.shapes
    ]


def _delta(shape) -> jax.Array:
    out = np.zeros(shape)
    out[shape[0] // 2, shape[1] // 2] = 1.0
    return jnp.asarray(out)


def _scalar_coefficients(background, shape_values, indicators, grid_shape):
    """Coefficients of a piecewise-constant scalar: background + Σ (value − background)·χ."""
    total = jnp.asarray(background)[..., None, None] * _delta(grid_shape)
    for value, chi in zip(shape_values, indicators, strict=True):
        total = total + (jnp.asarray(value) - jnp.asarray(background))[..., None, None] * chi
    return total


def _toeplitz(coefficients: jax.Array, orders: np.ndarray, centre) -> jax.Array:
    """Block-Toeplitz ``[[f]]_{ij} = c_{m_i − m_j, n_i − n_j}``, ``[..., N, N]``."""
    dm = orders[:, None, 0] - orders[None, :, 0] + centre[0]
    dn = orders[:, None, 1] - orders[None, :, 1] + centre[1]
    return coefficients[..., dm, dn]


# ---------------------------------------------------------------------------
# Factorisation
# ---------------------------------------------------------------------------


def _li_functions(tensor: jax.Array) -> dict[str, jax.Array]:
    """Li's rearrangement of a tensor in its own frame (index 0 = the normal).

    ``(E_n, D_t, D_z) = F · (D_n, E_t, E_z)``; returns the coefficient functions
    in the form the operator assembly uses: ``nn`` = 1/ε_nn, ``nt`` = ε_nt/ε_nn,
    ``tn`` = ε_tn/ε_nn and ``tt`` = ε_tt − ε_tn·ε_nt/ε_nn (the Schur complements).
    """
    e = [[tensor[..., i, j] for j in range(3)] for i in range(3)]
    out = {"nn": 1.0 / e[0][0]}
    for key, j in (("nt", 1), ("nz", 2)):
        out[key] = e[0][j] / e[0][0]
    for key, i in (("tn", 1), ("zn", 2)):
        out[key] = e[i][0] / e[0][0]
    for key, (i, j) in (("tt", (1, 1)), ("tz", (1, 2)), ("zt", (2, 1)), ("zz", (2, 2))):
        out[key] = e[i][j] - e[i][0] * e[0][j] / e[0][0]
    return out


def _full(blocks: jax.Array) -> jax.Array:
    """``[..., 3, 3, N, N]`` blocks -> ``[..., 3N, 3N]``."""
    moved = jnp.swapaxes(blocks, -3, -2)  # [..., 3, N, 3, N]
    size = blocks.shape[-1]
    return moved.reshape(blocks.shape[:-4] + (3 * size, 3 * size))


def _blocks3(full: jax.Array) -> jax.Array:
    """``[..., 3N, 3N]`` -> ``[..., 3, 3, N, N]``."""
    size = full.shape[-1] // 3
    return jnp.swapaxes(full.reshape(full.shape[:-2] + (3, size, 3, size)), -3, -2)


def _frame_pieces(tensor: jax.Array, n: jax.Array, t: jax.Array, z: jax.Array):
    """The lab-frame pieces of Li's rule in the local frame, all even in n.

    ``1/ε_nn`` and three 3×3 matrix fields: ``G`` (the normal D's dependence on
    the continuous E), ``H`` (the tangential and z D's on the normal D) and ``S``
    (theirs on the continuous E). Frame vectors are ``[Gx, Gy, 3]``.
    """
    rotation = jnp.stack([n, t, z], axis=-2)  # rows n, t, z
    local = rotation @ tensor[..., None, None, :, :] @ jnp.swapaxes(rotation, -1, -2)
    f = _li_functions(local)

    def outer(a, b):
        return a[..., :, None] * b[..., None, :]

    def scale(value, matrix):
        return value[..., None, None] * matrix

    g = scale(f["nt"], outer(n, t)) + scale(f["nz"], outer(n, z))
    h = scale(f["tn"], outer(t, n)) + scale(f["zn"], outer(z, n))
    s = (
        scale(f["tt"], outer(t, t))
        + scale(f["tz"], outer(t, z))
        + scale(f["zt"], outer(z, t))
        + scale(f["zz"], outer(z, z))
    )
    # S acts only on (I − P)·E, so adding (ε n)·nᵀ changes nothing exactly and
    # makes S = ε for isotropic media: the textbook normal-vector form.
    s = s + outer((tensor[..., None, None, :, :] @ n[..., :, None])[..., 0], n)
    return f["nn"], g, h, s


def _normal_vector_assemble(inv_nn, projector, g, h, s) -> jax.Array:
    """``[[ε]] = (I + [[H]])·(A ⊗ I)·([[P]] + [[G]]·[[I − P]]) + [[S]]·[[I − P]]``.

    The continuous fields are ``(I − P)·E`` (tangential and z E) and ``n·D_n``.
    Every product pairs a discontinuous coefficient with one of them, or the
    smooth projector with E; and the inverse rule ``A = [[1/ε_nn]]⁻¹`` acts on
    the normal component, as in the isotropic normal-vector method, which this
    reduces to (as it reduces to Li's 1D rule for ``P = x̂x̂``).
    """
    p, g, h, s = (_full(m) for m in (projector, g, h, s))
    eye = jnp.eye(p.shape[-1], dtype=g.dtype)
    size = inv_nn.shape[-1]
    a = _solve(inv_nn, jnp.broadcast_to(jnp.eye(size, dtype=g.dtype), inv_nn.shape))
    a3 = jnp.kron(jnp.eye(3, dtype=g.dtype), a) if a.ndim == 2 else _kron3(a)
    tangential = eye - p
    return _blocks3((eye + h) @ a3 @ (p + g @ tangential) + s @ tangential)


def _kron3(a: jax.Array) -> jax.Array:
    """Block-diagonal ``I₃ ⊗ A`` for batched ``A`` ``[..., N, N]``."""
    zero = jnp.zeros_like(a)
    rows = [[a, zero, zero], [zero, a, zero], [zero, zero, a]]
    return _full(jnp.stack([jnp.stack(r, -3) for r in rows], -4))


def _phases(fractions: jax.Array, k_max: int, sign: float) -> jax.Array:
    """``exp(sign·2πi·f·m)`` for sample fractions ``f`` and ``|m| ≤ k_max``: ``[F, 2K+1]``."""
    m = jnp.arange(-k_max, k_max + 1)
    return jnp.exp(sign * 2j * jnp.pi * fractions[:, None] * m[None, :])


def _normal_field(indicators, lattice, grid, sigma):
    """Unit normal ``(nx, ny)`` of the pattern's edges at cell-centred points.

    The points are ``u·a₁ + v·a₂`` on a regular grid of fractions ``(u, v)``,
    which samples a cell of any shape uniformly.
    """
    chi = sum(indicators)  # union (shapes do not overlap); [..., 2K₁+1, 2K₂+1]
    k1_max, k2_max = chi.shape[-2] // 2, chi.shape[-1] // 2
    gx, gy = _grid_wavevectors(k1_max, k2_max, lattice)
    smooth = chi * jnp.exp(-0.5 * sigma**2 * (gx**2 + gy**2))
    u = (jnp.arange(grid[0]) + 0.5) / grid[0]
    v = (jnp.arange(grid[1]) + 0.5) / grid[1]
    eu, ev = _phases(u, k1_max, 1.0), _phases(v, k2_max, 1.0)
    dx = jnp.real(jnp.einsum("...mn,im,jn->...ij", 1j * gx * smooth, eu, ev))
    dy = jnp.real(jnp.einsum("...mn,im,jn->...ij", 1j * gy * smooth, eu, ev))
    squared = dx**2 + dy**2
    flat = squared < 1e-18 * jnp.max(squared)  # only the field's isolated zeros
    safe = jnp.sqrt(jnp.where(flat, 1.0, squared))  # differentiable everywhere
    # The natural gradient direction, not a sign-canonicalised one: the factorisation
    # is pointwise independent of n's sign, but any fixed half-plane cut coincides
    # with some edge direction (n = ±ŷ on a horizontal edge), putting a jump in n
    # exactly on a material boundary. The natural field flips only inside uniform
    # regions (the medial lines), where every factor is continuous.
    nx = jnp.where(flat, 1.0, dx / safe)
    ny = jnp.where(flat, 0.0, dy / safe)
    return nx, ny, u, v


def _grid_coefficients(samples, u, v, k1_max, k2_max):
    """Fourier coefficients ``|m| ≤ k1_max, |n| ≤ k2_max`` of samples at fractions ``(u, v)``."""
    eu, ev = _phases(u, k1_max, -1.0), _phases(v, k2_max, -1.0)
    return jnp.einsum("...ij,im,jn->...mn", samples + 0j, eu, ev) / (u.size * v.size)


def _convolve(chi, g_hat, out_x: int, out_y: int):
    """Coefficients ``|k| ≤ out`` of the product χ·g from χ's (wide) and ĝ's (narrow)."""
    cx, cy = chi.shape[-2] // 2, chi.shape[-1] // 2
    qx, qy = g_hat.shape[-2] // 2, g_hat.shape[-1] // 2
    kx = np.arange(-out_x, out_x + 1)
    ky = np.arange(-out_y, out_y + 1)
    px = np.arange(-qx, qx + 1)
    py = np.arange(-qy, qy + 1)
    ix = (kx[:, None] - px[None, :]) + cx  # [out, q]
    iy = (ky[:, None] - py[None, :]) + cy
    gathered = chi[..., ix[:, None, :, None], iy[None, :, None, :]]  # [..., kx, ky, qx, qy]
    return jnp.einsum("...abcd,...cd->...ab", gathered, g_hat)


def _factorize(layer: PatternedLayer, which: str, max_order, lattice, orders, rule, grid):
    """``[[ε]]`` (or ``[[μ]]``) as ``[..., 3, 3, N, N]`` blocks."""
    mx, my = max_order
    size = orders.shape[0]
    centre = (2 * mx, 2 * my)
    if which == "eps":
        background = jnp.asarray(layer.background) + 0j
        values = [jnp.asarray(s.eps) + 0j for s in layer.shapes]
    else:
        eye = jnp.eye(3, dtype=complex)
        background = eye if layer.background_mu is None else jnp.asarray(layer.background_mu) + 0j
        values = [eye if s.mu is None else jnp.asarray(s.mu) + 0j for s in layer.shapes]
        if layer.background_mu is None and all(s.mu is None for s in layer.shapes):
            return jnp.einsum("ij,mn->ijmn", eye, jnp.eye(size, dtype=complex))  # μ = 1

    if not layer.shapes:  # uniform: every rule is the tensor itself
        ident = jnp.eye(size, dtype=complex)
        return jnp.einsum("...ij,mn->...ijmn", background, ident)

    if rule == "laurent":
        indicators = _indicators(layer, 2 * mx, 2 * my, lattice)
        shape = indicators[0].shape[-2:]
        blocks = [
            [
                _toeplitz(
                    _scalar_coefficients(
                        background[..., i, j], [v[..., i, j] for v in values], indicators, shape
                    ),
                    orders,
                    centre,
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        return jnp.stack([jnp.stack(jnp.broadcast_arrays(*row), -3) for row in blocks], -4)
    if rule != "normal":
        raise ValueError(f"Unknown factorization {rule!r}; use 'normal' or 'laurent'.")

    # frame-dependent parts are smooth: keep their harmonics up to q = 2·max_order
    qx_max, qy_max = 2 * mx, 2 * my
    wide = _indicators(layer, 2 * mx + qx_max, 2 * my + qy_max, lattice)
    # smoothed over a tenth of the cell, so the field's direction is defined
    # everywhere but isolated points, and is the edge normal along straight edges
    a1, a2, _, _, _ = _vectors(lattice)
    sigma = 0.1 * jnp.minimum(jnp.linalg.norm(a1), jnp.linalg.norm(a2))
    nx, ny, u, v = _normal_field(wide, lattice, grid, sigma)
    zero, one = jnp.zeros_like(nx), jnp.ones_like(nx)
    projector_grid = jnp.stack(
        [jnp.stack([nx * nx, nx * ny, zero], -1), jnp.stack([nx * ny, ny * ny, zero], -1),
         jnp.stack([zero, zero, zero], -1)],
        -2,
    )  # fmt: skip

    chi_background = _delta(wide[0].shape[-2:]) - sum(wide)
    frame = (
        jnp.stack([nx, ny, zero], -1),
        jnp.stack([-ny, nx, zero], -1),
        jnp.stack([zero, zero, one], -1),
    )

    def coefficients(field):  # [..., Gx, Gy] -> [..., 4Mx+1, 4My+1], one material
        return _grid_coefficients(field, u, v, qx_max, qy_max)

    p_coefficients = _grid_coefficients(
        jnp.moveaxis(projector_grid, (-2, -1), (0, 1)), u, v, 2 * mx, 2 * my
    )  # [3, 3, ..., 4Mx+1, 4My+1]
    p_blocks = jnp.moveaxis(_toeplitz(p_coefficients, orders, centre), (0, 1), (-4, -3))

    def assemble(tensors):
        sums = {"inv_nn": 0.0, "g": 0.0, "h": 0.0, "s": 0.0}
        for chi, tensor in zip([chi_background, *wide], tensors, strict=True):
            inv_nn, g, h, s = _frame_pieces(tensor, *frame)
            sums["inv_nn"] = sums["inv_nn"] + _convolve(chi, coefficients(inv_nn), 2 * mx, 2 * my)
            for key, matrix in (("g", g), ("h", h), ("s", s)):
                entries = [
                    [
                        _convolve(chi, coefficients(matrix[..., i, j]), 2 * mx, 2 * my)
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
                block = jnp.stack([jnp.stack(jnp.broadcast_arrays(*r), -3) for r in entries], -4)
                sums[key] = sums[key] + block
        blocks = {key: _toeplitz(value, orders, centre) for key, value in sums.items()}
        batch = jnp.broadcast_shapes(
            p_blocks.shape[:-4], *(b.shape[:-4] for k, b in blocks.items() if k != "inv_nn")
        )
        projector = jnp.broadcast_to(p_blocks, batch + (3, 3, size, size))
        inv_nn = jnp.broadcast_to(blocks["inv_nn"], batch + (size, size))
        g, h, s = (jnp.broadcast_to(blocks[k], batch + (3, 3, size, size)) for k in ("g", "h", "s"))
        return _full(_normal_vector_assemble(inv_nn, projector, g, h, s))

    # The same rules with every product reversed are the conjugate transpose of
    # the assembly for εᴴ. Their mean is Hermitian for lossless media, so the
    # truncated model conserves energy exactly, and it keeps the 1D reduction
    # (the projector then commutes with everything it multiplies).
    tensors = [background, *values]
    forward = assemble(tensors)
    reverse = assemble([jnp.conj(jnp.swapaxes(t, -1, -2)) for t in tensors])
    return _blocks3(0.5 * (forward + jnp.conj(jnp.swapaxes(reverse, -1, -2))))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _orders(mx: int, my: int, lattice=None, truncation: str = "parallelogram") -> np.ndarray:
    """Harmonic indices ``(m, n)`` kept, ``[N, 2]``, in lexicographic order."""
    m, n = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
    orders = np.stack([m.reshape(-1), n.reshape(-1)], axis=-1)
    if truncation == "parallelogram":
        return orders
    if truncation != "circular":
        raise ValueError(f"Unknown truncation {truncation!r}; use 'parallelogram' or 'circular'.")
    try:
        a1 = np.asarray(lattice.a1, dtype=float)
        a2 = np.asarray(lattice.a2, dtype=float)
    except (TypeError, jax.errors.TracerArrayConversionError) as error:
        raise ValueError(
            "circular truncation needs a concrete lattice (the harmonics kept depend on it); "
            "use truncation='parallelogram' to optimise the lattice itself"
        ) from error
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = 2 * np.pi * np.array([a2[1], -a2[0]]) / cross
    b2 = 2 * np.pi * np.array([-a1[1], a1[0]]) / cross
    # the radius, from the lattice's shortest reciprocal vector, is basis-independent;
    # the index box is then whatever holds that circle (m = G·a₁/2π, n = G·a₂/2π)
    small = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing="ij"), -1).reshape(
        -1, 2
    )
    small = small[(small != 0).any(axis=1)]
    shortest = np.min(np.linalg.norm(small[:, :1] * b1 + small[:, 1:] * b2, axis=1))
    radius = max(mx, my) * shortest * (1 + 1e-9)
    m1 = int(np.ceil(radius * np.linalg.norm(a1) / (2 * np.pi)))
    m2 = int(np.ceil(radius * np.linalg.norm(a2) / (2 * np.pi)))
    m, n = np.meshgrid(np.arange(-m1, m1 + 1), np.arange(-m2, m2 + 1), indexing="ij")
    orders = np.stack([m.reshape(-1), n.reshape(-1)], axis=-1)
    g = orders[:, :1] * b1 + orders[:, 1:] * b2
    return orders[np.linalg.norm(g, axis=1) <= radius]


def _operators(
    layers, lattice, orders, max_order, factorization, grid, kx, ky, direction,
    eps_incident, eps_exit,
):  # fmt: skip
    """Mode bases and each layer's operator in the reference basis.

    Returns ``(w_ref, w_in, flux_in, w_out, flux_out, deltas)``.
    """
    eps_ref = 2.0 * (kx**2 + ky**2) + 1.0
    w_ref, flux_ref = _harmonic_modes(eps_ref, kx, ky, direction)
    w_ref = w_ref / jnp.sqrt(2.0 * jnp.abs(flux_ref))[..., None, :]
    w_in, flux_in = _harmonic_modes(jnp.asarray(eps_incident)[..., None], kx, ky, direction)
    w_out, flux_out = _harmonic_modes(jnp.asarray(eps_exit)[..., None], kx, ky, direction)
    deltas = []
    for layer in layers:
        eps = _factorize(layer, "eps", max_order, lattice, orders, factorization, grid)
        mu = _factorize(layer, "mu", max_order, lattice, orders, factorization, grid)
        delta = _fourier_system_matrix(eps, mu, kx, ky)
        deltas.append(_solve(w_ref, delta @ w_ref))
    return w_ref, w_in, flux_in, w_out, flux_out, deltas


def solve_crossed_grating(
    layers: list[PatternedLayer],
    period_um: tuple[float, float],
    wavelength_um: jax.Array,
    incidence_rad: jax.Array,
    eps_incident: jax.Array,
    eps_exit: jax.Array,
    max_order: tuple[int, int],
    azimuth_rad: jax.Array = 0.0,
    factorization: str = "normal",
    doublings: int | None = None,
    grid: tuple[int, int] | None = None,
    truncation: str = "parallelogram",
) -> CrossedGratingResult:
    """Diffraction of a plane wave by a stack of 2D-periodic anisotropic layers.

    Args:
        layers: :class:`PatternedLayer` s from the incident side down.
        period_um: A :class:`Lattice`, or ``(Λx, Λy)`` in µm for a rectangular one.
        wavelength_um: Vacuum wavelength in µm.
        incidence_rad: Polar angle of incidence in the incident medium.
        eps_incident, eps_exit: Real, positive permittivities of the half-spaces.
        max_order: ``(M₁, M₂)``: orders ``|m| ≤ M₁``, ``|n| ≤ M₂`` along ``b₁``, ``b₂`` (static).
        azimuth_rad: Plane of incidence, from the x axis.
        factorization: ``"normal"`` (default) or ``"laurent"`` (static).
        doublings: Doublings per layer (static); by default grows with the order.
        grid: Sampling ``(G₁, G₂)`` of the cell (along ``a₁``, ``a₂``) for the
            normal-vector method's smooth parts.
        truncation: ``"parallelogram"`` (default) or ``"circular"`` (static).
            Circular keeps ``|G| ≤ max(M₁, M₂)·g_min``, with ``g_min`` the
            shortest reciprocal lattice vector: it respects the lattice's
            symmetry (hexagonal included), and the results do not depend on the
            primitive vectors chosen to describe it.

    Returns:
        :class:`CrossedGratingResult`: ``orders`` ``[N, 2]`` and ``r``, ``t``,
        ``R``, ``T`` ``[..., N, out, in]``. Only order (0, 0) is incident.
    """
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "hyperbolic_optics.rcwa2d is running in complex64 because 64-bit JAX is off; "
            'call jax.config.update("jax_enable_x64", True) for reference accuracy.',
            UserWarning,
            stacklevel=2,
        )
    mx, my = max_order
    lattice = _as_lattice(period_um)
    _, _, b1, b2, _ = _vectors(lattice)
    orders = _orders(mx, my, lattice, truncation)
    # the index box the kept harmonics span (circular truncation may widen it)
    mx, my = int(np.abs(orders[:, 0]).max()), int(np.abs(orders[:, 1]).max())
    max_order = (mx, my)
    size = orders.shape[0]
    zeroth = int(np.flatnonzero((orders[:, 0] == 0) & (orders[:, 1] == 0))[0])
    if doublings is None:
        doublings = default_doublings(max(mx, my))
    if grid is None:
        grid = (max(32, 8 * mx + 8), max(32, 8 * my + 8))

    eps_incident, eps_exit = jnp.asarray(eps_incident), jnp.asarray(eps_exit)
    wavelength_um, azimuth_rad = jnp.asarray(wavelength_um), jnp.asarray(azimuth_rad)
    k_par = jnp.sqrt(eps_incident) * jnp.sin(jnp.asarray(incidence_rad))
    scale = (wavelength_um / (2 * jnp.pi))[..., None]
    gx = orders[:, 0] * b1[0] + orders[:, 1] * b2[0]
    gy = orders[:, 0] * b1[1] + orders[:, 1] * b2[1]
    kx = (k_par * jnp.cos(azimuth_rad))[..., None] + gx * scale
    ky = (k_par * jnp.sin(azimuth_rad))[..., None] + gy * scale
    direction = (jnp.cos(azimuth_rad)[..., None], jnp.sin(azimuth_rad)[..., None])
    k0 = 2 * jnp.pi / wavelength_um

    w_ref, w_in, flux_in, w_out, flux_out, deltas = _operators(
        layers, lattice, orders, max_order, factorization, grid, kx, ky, direction,
        eps_incident, eps_exit,
    )  # fmt: skip
    scattering = _interface(w_in, w_ref)
    for layer, delta_ref in zip(layers, deltas, strict=True):
        k0d = k0 * jnp.asarray(layer.thickness_um)
        scattering = _star(scattering, _layer(delta_ref, k0d, doublings))
    scattering = _star(scattering, _interface(w_ref, w_out))

    t_full, _, r_full, _ = scattering
    incident = jnp.array([zeroth, size + zeroth])

    def by_order(block):
        amplitude = block[..., :, incident]
        return jnp.stack([amplitude[..., :size, :], amplitude[..., size:, :]], axis=-2)

    def per_order(flux):
        forward = flux[..., : 2 * size]
        return jnp.stack([forward[..., :size], forward[..., size:]], axis=-1)

    r, t = by_order(r_full), by_order(t_full)
    forward_in = flux_in[..., : 2 * size]
    power_in = jnp.stack([forward_in[..., zeroth], forward_in[..., size + zeroth]], -1)
    R = jnp.abs(r) ** 2 * per_order(flux_in)[..., :, :, None] / power_in[..., None, None, :]
    T = jnp.abs(t) ** 2 * per_order(flux_out)[..., :, :, None] / power_in[..., None, None, :]
    return CrossedGratingResult(orders=orders, r=r, t=t, R=R, T=T)
