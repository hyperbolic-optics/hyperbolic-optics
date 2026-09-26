# Gratings (RCWA)

`hyperbolic_optics.rcwa` computes diffraction by structures that are periodic in
x, uniform in y and layered in z. Any segment of any layer can have a full 3×3
permittivity and permeability, including the xz and yz components of a tilted
crystal axis, and incidence can be conical. It is written in JAX like the
[differentiable planar solver](differentiable.md) and uses the same
eigendecomposition-free layer engine, so `jax.grad` works with respect to:

- geometry: fill factor, ridge position, thickness, period
- materials and their orientation
- wavelength and incidence

It needs the optional JAX dependency and 64-bit JAX, as the planar solver does.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from hyperbolic_optics import bcm, rcwa

eye = jnp.eye(3, dtype=complex)
# LiNbO3 ridges with the optic axis tilted 45° out of plane, in air
crystal = jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex))
ridge = bcm.rotate_tensor(crystal, bcm.rotation_matrix(0.0, jnp.radians(45.0), 0.0))

layers = [
    rcwa.lamellar_layer(ridge, eye, fill=0.4, thickness_um=0.35),
    rcwa.uniform_layer(2.1 * eye, 0.1),         # an unpatterned film below it
]
result = rcwa.solve_grating(
    layers,
    period_um=0.9,
    wavelength_um=1.0,
    incidence_rad=jnp.radians(20.0),
    eps_incident=1.0,
    eps_exit=1.45**2,
    max_order=15,                               # keeps orders −15 … 15
)
result.orders           # [-15 … 15]
result.T[..., 1, 0]     # per order: p in, s out, as a diffraction efficiency
```

`r` and `t` are complex amplitudes, and `R` and `T` are efficiencies. Each is
indexed `[..., order, out, in]` with 0 = p and 1 = s, and each order is resolved
into its own p and s relative to its plane of diffraction. Only the zeroth order
is incident. Evanescent orders have zero efficiency.

Build layers with:

- `uniform_layer(eps, thickness)` — unpatterned
- `lamellar_layer(eps_ridge, eps_groove, fill, thickness, offset=0.0)` — a
  ridge and a groove per period
- `sampled_layer(eps_samples, thickness)` — any profile, sampled at equal
  widths
- `GratingLayer(thickness, boundaries, eps)` — segments with arbitrary edges

Edges enter through exact Fourier coefficients, so derivatives with respect to
`fill` and `offset` are exact rather than pixelated. A profile that also varies
in z, such as a slanted or twisted grating, is a stack of thin layers.

## What materials it handles

Every segment takes a full 3×3 ε and μ with no symmetry assumed. These classes
of material are validated:

| material | example | checked by |
|---|---|---|
| isotropic dielectric and metal | silicon, gold | FMMAX (same answer to 4e-7–2e-5), energy, symmetry |
| anisotropic, axis in the plane | LiNbO₃ | FMMAX (converges to the same answer, 1e-5) |
| **hyperbolic**, axis in the plane | α-MoO₃ at 900 cm⁻¹ | FMMAX (converges to the same answer, 1e-4) |
| **hyperbolic, full tensor** (xz, yz terms) | hBN at 1450 cm⁻¹, axis tilted | eigenmode solution (1e-10), NumPy engine for the unpatterned film (1e-14), mirror symmetries, energy, passivity |
| **non-symmetric** (gyrotropic, ε_xy = −ε_yx) | Hermitian gyrotropic tensor | FMMAX (converges to the same answer, 1e-5), energy conservation |
| near-zero ε_xx | ε_xx = 0.01i | stays finite and passive |

Two limits follow from the method itself:

- **ε_xx and ε_zz must be nonzero in every segment** (and μ_xx, μ_zz likewise).
  Li's rule contains 1/ε_xx and the field equations contain 1/ε_zz, so an
  *exactly* zero, lossless value gives `NaN`. Any loss at all (ε_xx = 0.01i
  was tested) keeps it finite.
- **Incident and exit half-spaces must be isotropic and lossless.**

## Convergence

Truncation is the main source of error: keep raising `max_order` until the
quantity you need stops changing. With the default Li factorisation, the change
to the 161-order result is roughly:

| grating | 41 orders | 81 orders |
|---|---|---|
| silicon ridge in air | 1e-4 | 3e-5 |
| gold ridge, TM | 1e-2 | 2e-3 |
| LiNbO₃, axis in the plane | 5e-5 | 9e-6 |
| hyperbolic MoO₃, axis in the plane | 7e-4 | 2e-4 |
| hyperbolic hBN, axis tilted (full tensor) | 2e-2 | 6e-3 |

Hyperbolic ridges converge slowly because they carry propagating waves of very
high in-plane wavevector, so plan on 81 orders or more for them.

The number of doublings per layer grows with `max_order` automatically (see
`rcwa.default_doublings`). Pass `factorization="laurent"` to compare against the
unmodified Laurent rule, which reaches the same limit much more slowly for TM
light.

### Rounding

Energy conservation measures the numerical error that is left apart from
truncation. Lossless dielectric gratings conserve energy to 1e-13 at 41 orders
and 1e-11 at 81.

Lossless hyperbolic gratings are the hardest case: 1e-11 at 41 orders, 1e-9 at
81 and 3e-8 at 161. That is still far below their truncation error.

### Gradients with respect to geometry on hyperbolic ridges

The gradient is exact for the truncated model at any `max_order`. For some
hyperbolic ridges, however, the truncated model itself depends on the edge
positions with a ripple, whose amplitude falls as orders are added while its
frequency rises. The ripple in the *slope* therefore does not fade.

- **Tilted-axis hBN ridges** (period 8 µm, first diffracted order): the ripple
  in d/d(fill) is comparable to the true slope even at 161 orders. See panel (e)
  of `benchmarks/hyperbolic_grating/`.
- **A subwavelength hBN ribbon absorber** with the flakes' natural orientation
  (`examples/hbn_absorber_gdsfactory.py`) shows no ripple, even with hard edges.
- **Dielectric ridges** show none either: silicon's d/d(fill) is identical at
  41, 81 and 161 orders.

So check before optimising geometry: compare the gradient at two truncations.
Where it ripples, smooth the edges with `edge_width` (a fraction of the period)
on `lamellar_layer` or `GratingLayer`. That convolves the profile with a box,
multiplying Fourier coefficient m by sinc(m · edge_width).

On the tilted-hBN grating:

- at 161 orders, `edge_width = 0.04` removes the ripple completely (the gradient
  changes sign 0 times, against 30 with hard edges);
- at 81 orders no width suffices, because the remaining ripple comes from the
  hBN's own high-k fields rather than the edges.

Smoothing also shifts the answer slightly (−4% in that first-order efficiency at
`edge_width = 0.04`). Re-simulate the hard-edged layout afterwards to measure
the difference.

## Validation

In `tests/test_rcwa.py`:

- **Uniform layers** reproduce the planar solver exactly (to about 1e-13), at
  any truncation and at conical incidence.
- **The doubling engine** agrees with:
  - a SciPy matrix exponential of the same Fourier-space operator on a
    patterned full-tensor layer
  - an eigenmode S-matrix on a hyperbolic full-tensor layer, lossy and lossless
- **Physics:**
  - lossless gratings conserve energy across all orders, including full-tensor,
    hyperbolic, gyrotropic and conical ones
  - lossy gratings absorb and never gain energy
  - a symmetric grating diffracts symmetrically at normal incidence
  - mirroring the structure in x or y (which flips the signs of the
    off-diagonal terms) mirrors the result
  - a lateral shift changes only the phases of the orders
  - Li's rule converges faster than Laurent's
- **Gradients** match finite differences for fill factor, offset, thickness,
  period, wavelength, incidence, azimuth and crystal tilt.

Benchmarks, each of which writes PNG figures to its own `output/` folder:

- **`benchmarks/fmmax/`** compares against
  [FMMAX](https://github.com/mfschubert/fmmax) for six gratings, including
  in-plane hyperbolic MoO₃ and a gyrotropic ridge, with convergence plots and
  efficiency spectra. Isotropic gratings agree to 4e-7–2e-5 at every
  truncation. The anisotropic ones converge to FMMAX's answer extrapolated to
  infinitely many orders (1e-5–1e-4). FMMAX cannot represent xz / yz
  components.
- **`benchmarks/hyperbolic_grating/`** runs a tilted-axis hBN grating across its
  Reststrahlen band, using hBN's real dispersion. It shows the unpatterned-film
  limit against the NumPy engine (1e-14), convergence, geometry gradients
  against finite differences, and energy conservation of the lossless variant.
- **`benchmarks/paper_fig4/`** runs the liquid-crystal polarization grating of
  arXiv:2608.06185 and records where it does and does not match the paper's FEM
  curves.

## Limitations

- Periodic in one direction only. 2D-periodic metasurfaces are not supported
  yet.
- Incident and exit half-spaces must be isotropic and lossless.
- ε_xx, ε_zz, μ_xx and μ_zz must not be exactly zero (see above).
- Results are for an infinitely periodic structure under plane-wave
  illumination. They do not give the efficiency of a finite device such as a
  grating coupler.
- Speed: one 41-order grating layer takes about 85 ms per wavelength on a CPU,
  against about 250 ms for FMMAX on the same case (both run without `jax.jit`).
  Neither has been tuned, so this is not a performance comparison.
