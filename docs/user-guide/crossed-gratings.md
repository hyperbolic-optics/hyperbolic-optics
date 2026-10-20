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

- **Shapes.** `Shape(vertices, eps, mu=None)` takes any simple polygon in µm,
  either winding (`rectangle` and `ellipse` build common ones). Shapes may
  cross the cell edge and are periodised; they must not overlap each other or
  their images.
- **Orders.** Order `(m, n)` has in-plane wavevector
  `(kx₀ + m·λ/Λx, ky₀ + n·λ/Λy)`, resolved into its own p and s.
  `max_order=(Mx, My)` keeps `(2Mx + 1)(2My + 1)` harmonics; cost grows as
  their cube, so a few hundred is a practical ceiling on a CPU.
- **Batching.** Tensors, thicknesses and vertices may carry batch dimensions,
  and the solve can be `jax.jit`-compiled (`max_order`, `factorization` static).
- **Four-port S-matrix.** `crossed_grating_smatrix` gives the zeroth-order
  `PlanarSMatrix` for SAX circuits ([Scattering Matrices and SAX](sax.md)), on
  `bcm.stack_smatrix`'s normalisation; `kx=` replaces the angle, allowing
  evanescent ports.
- **Crystal substrates.** `eps_exit` may be a 3×3 tensor (anisotropic,
  hyperbolic or absorbing). Each order's two transmitted modes, power-normalised,
  replace p and s.

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

`truncation="circular"` keeps every `G = m·b₁ + n·b₂` with
`|G| ≤ max(M₁, M₂)·g_min`: it respects the lattice's symmetry (hexagonal
included) and does not depend on the primitive vectors chosen, but needs a
concrete lattice. To optimise a lattice constant, use the default
`"parallelogram"` (`|m| ≤ M₁, |n| ≤ M₂`).

## Fourier factorisation

`factorization="normal"` (default) is the normal-vector method generalised to
full tensors: Li's anisotropic rule in a local frame normal to the pattern's
edges, symmetrised so that lossless cells conserve energy at every truncation
(details in the [API reference](../api/rcwa2d.md)). `"laurent"` uses Toeplitz
matrices of ε and converges more slowly where E's normal component jumps.

## Convergence

The normal-vector rule converges faster than Laurent's: on a birefringent disc
(Λ = 0.6 µm, h = 0.25 µm, λ = 1 µm), R_pp is 0.00533 / 0.00539 at 121 / 529
harmonics, against FMMAX's 0.00543 at 1605, and Laurent's 0.00305 / 0.00415.
On the tilted-axis LiNbO₃ nanofins of arXiv:2608.06185, which FMMAX cannot
represent, T_xx at 550 nm is 0.9520 at 289 harmonics, against the paper's
0.9496 (BCM) and 0.9483 (FEM). Metal and hyperbolic inclusions converge slowly
under any rule: check a design at two truncations.

Tests: `tests/test_rcwa2d.py`.

## Limitations

- The incident half-space must be isotropic and lossless.
- Plane-wave results for an infinitely periodic structure.
