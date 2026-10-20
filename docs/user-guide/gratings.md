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

Every segment takes a full 3×3 ε and μ: isotropic dielectrics and metals,
in-plane anisotropic and hyperbolic crystals (LiNbO₃, α-MoO₃), full tensors
with xz and yz terms (tilted hBN) and non-symmetric, gyrotropic tensors. The
isotropic and in-plane cases agree with FMMAX to 4e-7–1e-4 (the comparison
script is in git history before `98a1f20`); full tensors, which FMMAX cannot
represent, agree with an eigenmode solution to 1e-10 and, unpatterned, with the
NumPy engine to 1e-14.

- ε_xx, ε_zz, μ_xx and μ_zz must be nonzero: Li's rule divides by them. Any
  loss keeps a near-zero value finite.
- Incident and exit half-spaces must be isotropic and lossless.

## Convergence

Truncation is the main error: raise `max_order` until the quantity you need
stops changing. With Li's factorisation, the change to the 161-order result is
roughly:

| grating | 41 orders | 81 orders |
|---|---|---|
| silicon ridge in air | 1e-4 | 3e-5 |
| gold ridge, TM | 1e-2 | 2e-3 |
| LiNbO₃, axis in the plane | 5e-5 | 9e-6 |
| hyperbolic MoO₃, axis in the plane | 7e-4 | 2e-4 |
| hyperbolic hBN, axis tilted (full tensor) | 2e-2 | 6e-3 |

Hyperbolic ridges carry very high-k waves, so plan on 81 orders or more.
Doublings grow with `max_order` automatically (`rcwa.default_doublings`).
Lossless gratings conserve energy to 1e-13 (dielectric, 41 orders) down to
3e-8 (hyperbolic, 161 orders), far below the truncation error.

### Gradients with respect to geometry on hyperbolic ridges

On some hyperbolic ridges (tilted-axis hBN) the truncated model ripples with
the edge positions; the ripple's amplitude falls with `max_order` but its
frequency rises, so the gradient's ripple persists. Compare gradients at two
truncations; if they ripple, set `edge_width` (a fraction of the period:
coefficient m is multiplied by sinc(m·edge_width)) and re-simulate the
hard-edged layout to measure the shift.

Tests: `tests/test_rcwa.py`.

## Limitations

- Periodic in one direction only; for 2D see
  [Crossed Gratings (2D RCWA)](crossed-gratings.md).
- Plane-wave results for an infinitely periodic structure.
