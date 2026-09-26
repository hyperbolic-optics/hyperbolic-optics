# Differentiable Solver (JAX)

`hyperbolic_optics.bcm` solves the same planar problem as `Structure`, written in
JAX so that `jax.grad` gives exact derivatives of any result with respect to layer
thicknesses, tensor components (including loss), crystal orientations, the angle
of incidence and the wavelength. Use it for gradient-based design. The payload
API is unchanged and remains the reference.

## Installation

JAX is an optional dependency:

```bash
pip install "hyperbolic-optics[jax]"
# or, in a checkout
uv sync --extra jax
```

## Precision

Enable 64-bit JAX before calling the solver. The module does not switch it on
for you, because that setting is global to your program:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without it everything runs in complex64 and the solver warns.

## Usage

The solver is a function of arrays, not payloads. Tensors are given in the lab
frame, with the plane of incidence x-z:

```python
import jax.numpy as jnp
from hyperbolic_optics import bcm

# LiNbO3 film, extraordinary axis in-plane at 45°, on glass
crystal = jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex))
film = bcm.rotate_tensor(crystal, bcm.rotation_matrix(0.0, 0.0, jnp.radians(45.0)))

result = bcm.solve_stack(
    eps=film[None],                      # [L, 3, 3], incident side first
    thickness_um=jnp.array([3.6]),       # [L]
    wavelength_um=jnp.linspace(0.6, 0.7, 41),
    incidence_rad=0.0,
    eps_incident=1.0,
    eps_exit=1.52**2,
)
result.T[..., 1, 0]  # p in, s out: transmitted power
```

`result` has `r`, `t` (complex) and `R`, `T` (power), each `[..., out, in]` with
index 0 = p and 1 = s. The complex coefficients use the same sign convention as
`Structure`, so `result.r[..., 0, 0]` is `r_pp` and `result.r[..., 1, 0]` is
`r_ps` (p in, s out). Every argument broadcasts over leading batch dimensions,
and `mu=` takes permeability tensors in the same layout.

For a frequency in cm⁻¹, pass `wavelength_um = 1e4 / frequency`.
`rotation_matrix(x, y, z)` follows the payload's `rotationX`/`rotationY`/
`rotationZ` convention, so a tensor rotated with it matches the one a
`Crystal Layer` would use.

### Gradients

```python
def objective(thickness):
    res = bcm.solve_stack(film[None], thickness[None], 0.65, 0.0, 1.0, 1.52**2)
    return res.T[1, 0]

jax.grad(objective)(jnp.asarray(3.6))
```

Wrap repeated calls in `jax.jit(..., static_argnames="doublings")`.
`examples/optimize_polarization_converter.py` optimises a film's thickness and
axis angle with Adam, then re-checks the result with `Structure`.

### Fields and layer-resolved absorption

```python
lossy = film + 0.05j * jnp.eye(3)
fields = bcm.stack_fields(lossy[None], jnp.array([3.6]), 0.65, 0.3, 1.0, 1.52**2,
                          samples_per_layer=100)
fields.absorption      # [..., L, 2]: absorbed fraction per layer, for p and s in
fields.R, fields.T     # [..., 2]
fields.interface_flux  # [..., L + 1, 2]: S_z at each interface over the incident flux
fields.z_um, fields.fields   # depths, and (Ex, Ey, Ez, Hx, Hy, Hz) there, per polarization
```

- **Stable.** The fields at each plane come from the scattering matrices of
  the stack above and below it, never from transfer matrices. Thick, strongly
  absorbing and evanescent layers stay finite.
- **Differentiable.** Gradients go through all of it, thicknesses included.
  The samples inside a layer sit at fixed fractions of its thickness
  (`samples_per_layer`, static), so moving a layer boundary moves its samples
  with it.
- **Validated.** `tests/test_stack_fields.py` checks:
  - R, T and each layer's absorption against the NumPy engine's
    `FieldProfile`, and all six field components against depth, to 1e-12;
  - each layer's absorption against the electric and magnetic loss density
    `½(E†·ε_a·E + H†·μ_a·H)` integrated over its depth, with
    `ε_a = (ε − ε†)/2i`. This check is independent of the interface fluxes,
    whose differences alone would only telescope.

## Method

Each layer's scattering matrix is built without eigendecomposition. The field
equation `dF/dz = iΔF` is integrated over a slice `k₀d / 2ᴺ` thin by a
fourth-order Taylor series, and the slice is doubled `N` times (default 20) by
star products. Only matrix products and linear solves are used, so there are no
eigenvector derivatives to diverge where modes are degenerate. This is the
bisection-doubling scheme of the boundary-value cascade method of Yang, Song &
Cai ([arXiv:2608.06185](https://arxiv.org/abs/2608.06185)).

One choice here differs from that paper. The paper writes a layer as a relation
between E at one face and H at the other, and that relation is singular whenever
a lossless layer is an odd number of quarter-waves thick. Here the doubling
works on wave amplitudes in a reference medium whose waves always propagate. Its
`kz = √(1 + kx²)` grows with the in-plane wavevector, and each wave is scaled to
unit power. In that basis a passive layer's scattering matrix is
power-normalised, so no step can meet a singular inverse. And because the
reference impedance tracks that of evanescent and high-k waves, it doesn't
reflect them almost totally, which is what a fixed `kz = 1` reference did, at
the cost of up to 1e-3 of rounding error on high-order gratings. The start-up series is derived independently of the
paper's supplement, so this is a related implementation rather than a
reproduction of that method.

## Validation

`tests/test_bcm.py` checks:

- **Closed forms.** Fresnel for a bare interface, including total internal
  reflection. The Airy formula for a film swept exactly onto every quarter-wave
  thickness.
- **The NumPy engine.** Complex coefficients, all eight, for dispersive
  hyperbolic calcite, an Otto gap over biaxial α-MoO₃ swept in azimuth, a
  gyrotropic-permeability layer, and a 150 µm evanescent gap that sends the
  transfer backend to `NaN`. Each is fed the engine's own tensors and agrees to
  about 1e-14.
- **Gradients.** Against central finite differences, for thickness, tilt,
  azimuth, wavelength, incidence and the real and imaginary parts of a tensor
  component.
- **Invariants.** Energy conservation (to 1e-12) and passivity, and convergence
  in the number of doublings.

## Limitations

- Incident and exit half-spaces must be isotropic and lossless. A
  `Semi Infinite Anisotropic Layer` exit is not supported yet.
- For wavelength derivatives that include the material's dispersion, take the
  tensor from `hyperbolic_optics.dispersion` (the built-in materials in JAX), not
  from the NumPy material classes, which carry no gradient.
- Exact grazing cutoff in the exit medium (`kx` equal to its index) is singular,
  as in any plane-wave basis.
- Field profiles and layer-resolved absorption are available from
  `FieldProfile` only, not yet here.
