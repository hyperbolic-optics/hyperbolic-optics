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

Enable 64-bit JAX first (the setting is global, so the package leaves it to you):

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without it the solver runs in complex64 and warns.

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

Fields at each plane come from the scattering matrices above and below it,
never from transfer matrices, so thick, absorbing and evanescent layers stay
finite. Samples sit at fixed fractions of each layer, so gradients include the
thicknesses. Each layer's absorption is checked against the loss density
integrated over its depth (`tests/test_stack_fields.py`).

### Stacks on bulk crystals

```python
from hyperbolic_optics import dispersion

calcite = bcm.rotate_tensor(dispersion.permittivity("Calcite", 1470.0),
                            bcm.rotation_matrix(0.0, jnp.radians(65.0), jnp.radians(30.0)))
result = bcm.solve_stack_on_crystal(
    eps=jnp.eye(3, dtype=complex)[None], thickness_um=jnp.array([1.0]),   # an air gap
    wavelength_um=1e4 / 1470.0, incidence_rad=0.6, eps_incident=5.5, eps_substrate=calcite,
)
result.r, result.R, result.T      # T: power carried into the crystal, per input
```

The NumPy engine's `Semi Infinite Anisotropic Layer`, differentiable: any
crystal in any orientation, lossy, lossless or hyperbolic. The outgoing waves
come from a matrix sign function rather than eigenvectors, so degenerate modes
(an optic axis along z at normal incidence) have finite gradients.

## Method

Each layer's scattering matrix is built without eigendecomposition. The field
equation `dF/dz = iΔF` is integrated over a slice `k₀d / 2ᴺ` thin by a
fourth-order Taylor series, and the slice is doubled `N` times (default 20) by
star products. Only matrix products and linear solves are used, so there are no
eigenvector derivatives to diverge where modes are degenerate. This is the
bisection-doubling scheme of the boundary-value cascade method of Yang, Song &
Cai ([arXiv:2608.06185](https://arxiv.org/abs/2608.06185)).

Unlike the paper's E/H form, which is singular when a lossless layer is an odd
number of quarter-waves thick, the doubling works on power-normalised wave
amplitudes in a reference medium with `kz = √(1 + kx²)`, so no step meets a
singular inverse. The start-up series is derived independently of the paper's
supplement: a related implementation, not a reproduction.

Tests: `tests/test_bcm.py` (closed forms, the NumPy engine to ~1e-14,
gradients, energy) and `tests/test_crystal_substrates.py`.

## Limitations

- `solve_stack` and `stack_smatrix` need isotropic, lossless half-spaces;
  use `solve_stack_on_crystal` for a crystal substrate.
- For wavelength derivatives that include the material's dispersion, take the
  tensor from `hyperbolic_optics.dispersion` (the built-in materials in JAX), not
  from the NumPy material classes, which carry no gradient.
- Exact grazing cutoff in the exit medium (`kx` equal to its index) is singular,
  as in any plane-wave basis.
