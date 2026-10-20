# Independent anisotropic-cell reference

`anisotropic_cell.py` compares RCWA with a separate 3D NGSolve Maxwell FEM
implementation. The FEM path imports no hyperbolic-optics code. It uses
[periodic boundaries](https://docu.ngsolve.org/latest/i-tutorials/unit-2.12-periodicity/periodicity.html),
quadratic curl-conforming elements and outgoing impedance boundaries beyond
vacuum spacers. The latter are exact for the propagating zeroth order; the
spacer-distance check measures sensitivity to reflected evanescent orders.

The cell is a 300 × 200 × 300 nm rectangular pillar on a 600 nm square lattice,
in air, at 1550 nm and normal incidence. Its permittivity is
`Rz(23°) Ry(31°) Rx(17°) diag(2.25, 3.24, 4) Rᵀ`; permeability is identity.
Both incident polarizations are solved. Complex reflection and transmission
matrices use p/s conventions and reference planes at the pillar faces.

The stored `anisotropic_cell_reference.json` contains the numerical results,
solver versions and controls. NGSolve 6.2.2607 gives:

| Check | Maximum absolute discrepancy |
|---|---:|
| Empty domain versus analytic propagation | 7.2e-5 |
| Isotropic film versus analytic Airy amplitudes | 4.8e-4 |
| FEM mesh size 90 → 80 nm, complex amplitudes | 3.9e-5 |
| Vacuum spacer 1.5 → 1.0 µm, including remeshing | 7.9e-5 |
| Finest FEM, `r†r + t†t − I` | 1.1e-5 |

Against the finest FEM result, JAX 0.11.2 RCWA gives:

| Harmonics | Maximum complex-amplitude difference |
|---:|---:|
| 25 | 0.00316 |
| 81 | 0.00291 |
| 169 | 0.00116 |
| 289 | 0.00113 |
| 441 | 0.00098 |
| 625 | 0.00064 |

At 25–289 harmonics the width derivative of converted power agrees with
centred finite differences (step 1e-5 µm) within 3e-8 relative. The derivative
still changes by 0.62% between 169 and 289 harmonics: an accurate derivative
of a truncated solve is not proof of convergence.

Reproduce from the repository root (FEM needs a separate optional dependency):

```bash
uv run --no-project --with ngsolve==6.2.2607 --with numpy python benchmarks/validation/anisotropic_cell.py fem --output /tmp/fem.json
uv run python benchmarks/validation/anisotropic_cell.py rcwa --output /tmp/rcwa.json
uv run python benchmarks/validation/anisotropic_cell.py rcwa --orders 10 12 --forward-only --output /tmp/rcwa-fine.json
```

`tests/test_external_reference.py` checks RCWA against the stored FEM data
without requiring NGSolve. Its 0.002 amplitude tolerance allows the measured
Fourier truncation error. These are refinement measurements for one positive
biaxial dielectric cell, not rigorous error bounds or independent validation
of other media, magnetic tensors or a finite lens.

The script also has an untested `--case hbn` (lossy hyperbolic) option with
oblique incidence (`--angle`, `--azimuth`); no results for it are stored.

The FEM `--meshes`, `--crystal-meshes` and spacer sizes are expressed in
coordinate units of `wavelength / 1.55`; each record includes this scale in µm.
Local crystal refinement leaves the vacuum mesh fixed. The spacer comparison
also remeshes the domain, so its discrepancy includes mesh sensitivity.
