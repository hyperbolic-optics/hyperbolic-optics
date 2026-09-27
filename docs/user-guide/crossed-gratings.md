# Crossed Gratings (2D RCWA)

`hyperbolic_optics.rcwa2d` computes diffraction by structures that are periodic
in x and y on a rectangular lattice and layered in z: pillar and hole arrays,
metasurfaces, photonic-crystal slabs. Each patterned layer is a background
material and non-overlapping polygons, and every one of them can have a full
3×3 permittivity and permeability, including the xz and yz components of a
tilted crystal axis. It is written in JAX with the same layer engine as the
[1D grating solver](gratings.md), so `jax.grad` works with respect to:

- geometry: every polygon vertex (so radii, widths, positions, rotations),
  thickness, period
- materials and their orientation
- wavelength and incidence

It needs the optional JAX dependency and 64-bit JAX.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from hyperbolic_optics import bcm, rcwa2d

eye = jnp.eye(3, dtype=complex)
# LiNbO3 with the optic axis tilted 45° out of plane
crystal = jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex))
ln = bcm.rotate_tensor(crystal, bcm.rotation_matrix(0.0, jnp.radians(45.0), 0.0))

fin = rcwa2d.Shape(rcwa2d.rectangle(center=(0.17, 0.17), size=(0.2, 0.1)), ln)
disc = rcwa2d.Shape(rcwa2d.ellipse(center=(0.17, 0.17), radii=(0.08, 0.08)), 2.1 * eye)
layers = [
    rcwa2d.PatternedLayer(0.30, eye, (fin,)),      # fins in air
    rcwa2d.PatternedLayer(0.05, 1.46**2 * eye, (disc,)),
    rcwa2d.uniform(2.1 * eye, 0.1),                # an unpatterned film
]
result = rcwa2d.solve_crossed_grating(
    layers, period_um=(0.34, 0.34), wavelength_um=0.55, incidence_rad=0.2,
    eps_incident=1.0, eps_exit=1.46**2, max_order=(5, 5),
)
result.orders        # [121, 2]: (m, n) of each order
result.T.shape       # (121, 2, 2): [order, out, in], 0 = p and 1 = s
```

- **Shapes.** `Shape(vertices, eps, mu=None)` takes any simple polygon, in µm
  and in either winding. `rectangle` and `ellipse` build the common ones.
  `ellipse` scales the polygon so its area equals the ellipse's. Shapes may
  cross the cell edge; they are periodised. They must not overlap.
- **Orders.** Order `(m, n)` has in-plane wavevector
  `(kx₀ + m·λ/Λx, ky₀ + n·λ/Λy)` and is resolved into its own p and s, as in
  the 1D solver. `max_order=(Mx, My)` keeps `(2Mx + 1)(2My + 1)` harmonics.
  Cost grows as the cube of that, so a few hundred harmonics is a practical
  ceiling on a CPU.
- **Batching.** Tensors, thicknesses and vertices can carry leading batch
  dimensions, and the whole solve can be `jax.jit`-compiled (with
  `max_order` and `factorization` static).

## Lattices

Any Bravais lattice, given by its primitive vectors:

```python
hexagonal = rcwa2d.Lattice.hexagonal(0.6)                 # triangular, a1 along x
oblique = rcwa2d.Lattice.oblique(0.7, 0.6, jnp.radians(75.0))
custom = rcwa2d.Lattice((0.6, 0.0), (0.3, 0.52))

holes = rcwa2d.PatternedLayer(0.25, 12.25 * eye,
                              (rcwa2d.Shape(rcwa2d.ellipse((0.0, 0.0), (0.18, 0.18)), eye),))
result = rcwa2d.solve_crossed_grating(
    [holes], hexagonal, wavelength_um=1.0, incidence_rad=0.3, eps_incident=1.0,
    eps_exit=2.1, max_order=(6, 6), truncation="circular",
)
```

- **Harmonics.** They are `G = m·b₁ + n·b₂`, with `b₁`, `b₂` the reciprocal
  vectors. A `(Λx, Λy)` period still means a rectangular lattice.
- **Truncation.** `truncation="parallelogram"` (the default) keeps
  `|m| ≤ M₁, |n| ≤ M₂`. `"circular"` keeps every `G` with
  `|G| ≤ max(M₁, M₂)·g_min`, where `g_min` is the shortest reciprocal vector.
  - It respects the lattice's point symmetry, which matters for hexagonal
    lattices.
  - It does not depend on the primitive vectors chosen: the same lattice in
    two bases gives the same answer.
  - It needs a concrete lattice (not a traced one) because the harmonics it
    keeps depend on it. Optimise a lattice constant with the parallelogram
    truncation.
- **Shapes** are given in real space and periodised on the lattice. The normal
  field is sampled on a grid of fractions `(u, v)` of the cell, `u·a₁ + v·a₂`.

## Fourier factorisation

The two rules differ in how they expand the products of discontinuous
functions (ε and the field components that jump at an edge).

- **`factorization="normal"`** (the default) is the normal-vector method,
  generalised here to full tensors.
  - At every point, the constitutive relation is written in a local frame
    whose first axis is normal to the pattern's edges.
  - Li's anisotropic rule is applied in that frame: `E_n`, `D_t` and `D_z` in
    terms of the components that are continuous across an edge, `D_n`, `E_t`
    and `E_z`.
  - For isotropic media this is the textbook normal-vector method; for
    stripes, the normal is x̂ everywhere and it is the 1D solver's rule
    exactly.
  - The assembly is averaged with its product-reversed twin, which keeps it
    Hermitian for lossless media. Energy is then conserved to rounding at
    every truncation, which the plain normal-vector method does not do.
- **`factorization="laurent"`** takes Toeplitz matrices of ε directly: simple,
  and correct in the limit, but it converges slowly wherever the normal
  component of E jumps.

The normal field is the gradient direction of the pattern smoothed over a
tenth of the cell. Only the smooth, frame-dependent factors are sampled on a
grid (`grid=(Gx, Gy)`, by default 8× the order). The shapes' own Fourier
coefficients are always exact: a closed-form sum over each polygon's edges.

## Convergence

Zeroth-order reflectance against FMMAX (Λ = 0.6 µm,
h = 0.25 µm, λ = 1 µm). The reference is FMMAX's normal-vector formulation
(`JONES_DIRECT`) at 1605 terms.

| case | channel | FMMAX, 1605 | normal, 121 / 529 | Laurent, 121 / 529 |
|---|---|---|---|---|
| silicon square pillar | R_pp | 0.0322 | 0.0232 / 0.0293 | 0.0532 / 0.0425 |
| silicon disc | R_pp | 0.0224 | 0.0208 / 0.0226 | 0.0297 / 0.0259 |
| birefringent disc, axis in plane | R_pp | 0.00543 | 0.00533 / 0.00539 | 0.00305 / 0.00415 |
| birefringent disc, axis in plane | R_ps | 1.09e-4 | 0.96e-4 / 1.04e-4 | 0.93e-4 / 0.98e-4 |
| hyperbolic disc (ε = 4 / −3 + 0.3i) | R_ps | 0.050 (0.049–0.055 from 193 terms) | 0.057 / 0.054 | 0.062 / 0.050 |

The tilted-axis LiNbO₃ nanofins of arXiv:2608.06185 (Sec. 3A), which FMMAX
cannot represent (ε_xz ≠ 0): T_xx at 550 nm is 0.9501 / 0.9520 (normal, 121 / 289 harmonics) and
0.9450 / 0.9464 (Laurent), against the paper's 0.9496 (its BCM) and 0.9483
(FEM). The two rules bracket the paper's values, within 0.4%.

Metal-like and hyperbolic inclusions (Re ε < 0 in some direction) converge
slowly under every Fourier factorisation, because the fields concentrate at
corners and edges. Check your own design at two truncations before trusting a
third digit.

## Validation of other lattices

In `tests/test_rcwa2d.py`:

- **Exact reductions.** A uniform layer on a hexagonal lattice is the planar
  solver (1e-12). A triangular lattice equals its rectangular supercell,
  `(a, a√3)` with two discs per cell (1e-12, Laurent's rule).
- **Basis independence.** A square lattice described by a skewed basis gives
  the same answer under circular truncation: 1e-12 with Laurent's rule, 1e-5
  with the normal-vector rule.
- **Symmetry.** Hexagonal lattices are unchanged under 60° rotation: 1e-12
  with Laurent's rule. With the normal-vector rule the error is 1e-4, limited
  by its sampling grid, and falls with a finer `grid`.
- **FMMAX.** Silicon discs on a 0.6 µm triangular lattice, at 0.3 rad. FMMAX,
  at 1615 terms, gives R_p = 0.56889 and R_s = 0.99224. This package gives
  R_p = 0.575 and R_s = 0.991 at 439 harmonics, still converging towards
  FMMAX's values.
- **Gradients** with respect to the lattice constant match finite differences.

## Validation

In `tests/test_rcwa2d.py`:

- **Polygon transforms** match direct quadrature, and a rectangle's matches
  the product of two sincs exactly.
- **Reductions:**
  - uniform layers reproduce the planar solver (1e-12);
  - stripes along x and along y reproduce the 1D grating solver per order
    (1e-12), for full tensors at conical incidence, under both
    factorisations.
- **Physics:**
  - lossless patterns conserve energy to 1e-11 across all orders, including
    tilted-axis crystals;
  - metal and hyperbolic pillars absorb and never gain energy;
  - a square pillar at normal incidence is polarization-blind and does not
    convert, and incidence from φ and φ + 90° agree.
- **FMMAX:** a birefringent disc with an in-plane axis agrees in all four
  zeroth-order channels, cross-polarized included, to 1e-4 at 121 harmonics.
- **Gradients** with respect to a polygon vertex, a width, a disc radius,
  thickness and wavelength match finite differences to 1e-5, under both
  factorisations.

## Limitations

- Circular truncation needs a lattice that is not being optimised (see above).
- Shapes must not overlap. A hole in a slab is a shape of the hole's material
  on the slab's background.
- Incident and exit half-spaces must be isotropic and lossless.
- Cost: the operator is `4N × 4N` for `N` harmonics, so memory and time grow
  as `N²` and `N³`.
- Plane-wave results for an infinitely periodic structure, as in the 1D
  solver.
