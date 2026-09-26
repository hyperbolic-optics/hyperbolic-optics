# Cross-check against DiffTMM

[DiffTMM](https://github.com/AI4Optics/DiffTMM) (Apache-2.0) is an independent
PyTorch implementation of the 4×4 Berreman transfer-matrix method. This compares
it with hyperbolic-optics -- both its NumPy engine and its differentiable JAX
solver (`hyperbolic_optics.bcm`) -- on five planar stacks. DiffTMM is not a dependency of
the package, so the command pins it for this one run:

```bash
uv sync --extra jax
uv run --with torch \
    --with "difftmm @ git+https://github.com/AI4Optics/DiffTMM@7671270bf6c08ba377c380d1c512def80e21659e" \
    python benchmarks/difftmm/compare.py
```

The script exits non-zero if any case is out of tolerance, and writes one figure
per case to `benchmarks/difftmm/output/` (not committed).

## What is compared, and how

- **Power, not amplitudes.** DiffTMM normalises the p amplitude to the
  tangential `Ex` rather than the total field, and the two codes name the
  cross-polarized coefficients differently. R and T resolved by incident and
  outgoing polarization (8 channels) are convention-free, so they are what is
  compared. DiffTMM's amplitudes are converted with the normal-flux weights of
  its basis (`n/cos θ` for p, `n cos θ` for s).
- **The same tensor.** hyperbolic-optics receives each layer's lab-frame
  permittivity directly, as an arbitrary material. DiffTMM's rotation of the
  principal indices is first checked to reproduce that tensor, so an Euler-angle
  convention mismatch cannot hide or fake agreement.
- **Isotropic exit media only.** DiffTMM supports nothing else, and the
  package's per-channel transmitted split is exact only there.

## Results

Maximum absolute difference in power over all resolved channels, and each
solver's energy-conservation error `|R + T − 1|` on the lossless cases:

| case | points | NumPy engine vs DiffTMM | JAX solver vs NumPy engine | DiffTMM \|R+T−1\| | NumPy engine \|R+T−1\| |
|---|---|---|---|---|---|
| `isotropic_film`: 200 nm TiO₂ on BK7 | 221 | 3.5e-6 | 6.9e-15 | 3.6e-6 | 6.7e-16 |
| `paper_fig3`: arXiv:2608.06185 Fig. 3 | 221 | 4.2e-5 | 5.3e-14 | 4.1e-5 | 1.3e-15 |
| `tilted_axis`: LiNbO₃ axis out of plane, full tensor, 0–85° | 171 | 3.8e-5 | 9.0e-14 | 3.9e-5 | 1.6e-14 |
| `hyperbolic_ir`: lossy hyperbolic film, tilted axis | 201 | 1.0e-6 | 1.9e-15 | — (lossy) | — (lossy) |

- Against the closed-form Airy result for the isotropic film, hyperbolic-optics
  agrees to 1.4e-15 and DiffTMM to 3.5e-6.
- In every case the cross-solver difference equals DiffTMM's own energy error,
  so it measures DiffTMM's complex64 arithmetic, not a disagreement in physics.
- The transfer and scattering backends of hyperbolic-optics agree to about 2e-15
  in total R and T, and the JAX solver matches the NumPy engine to about 1e-13.
  It is given the tensor the engine actually simulates: a `Crystal Layer` adds
  a deliberate 1e-8 rad offset to `rotationY`, which otherwise shows up as a
  1e-8 difference in the input, not the solver.

**Prism-coupled evanescent gap** (`prism_gap`, prism ε = 5.5 at 45°, 1000 cm⁻¹).
The power coupled through the gap, `1 − R`, should fall exponentially with gap
thickness. Both NumPy backends and the JAX solver follow it to about 1e-14 at
20 µm.
DiffTMM follows it until about 9 µm, then flattens at 1.6e-7, its float32 floor
on `R`. At these thicknesses neither method overflows; the package's transfer
backend only fails for much thicker gaps (see `tests/test_scattering.py`).

**Timing** (paper_fig3, 221 wavelengths, all channels, CPU, median):

| solver | time |
|---|---|
| hyperbolic-optics NumPy engine (complex128) | ~2 ms |
| DiffTMM (PyTorch, complex64) | ~1 ms |
| hyperbolic-optics JAX solver (complex128, jitted, compile excluded) | ~27 ms |

The JAX solver is slower for flat stacks: each layer runs 20 doublings of small
linear solves, where an eigendecomposition of a 4×4 matrix is cheap. Its purpose
here is exact gradients. The speed advantage the paper reports appears only at
large Fourier orders. This is not a performance benchmark.

## Assumptions

- **Fig. 3 LiNbO₃ indices.** The paper's Fig. 3 caption gives only the isotropic
  layer indices, so LiNbO₃ uses the non-dispersive εₒ = 5.46, εₑ = 5.06 of its
  Sec. 3A. The comparison between the two solvers doesn't depend on this, but
  overlaying the published curves would need the paper's actual values.
- All materials are non-dispersive, which is enough to compare solvers, not
  materials.

## Bugs this found in hyperbolic-optics (fixed)

- An angle sweep starting at normal incidence with an isotropic exit was
  all-NaN.
- `FieldProfile` under `backend="scattering"` raised on any frequency sweep
  whose first layer is non-dispersive.
