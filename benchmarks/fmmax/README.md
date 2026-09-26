# Cross-check against FMMAX

[FMMAX](https://github.com/mfschubert/fmmax) (MIT) is an independent JAX
implementation of RCWA (the Fourier modal method). This compares it with the
1D-grating solver `hyperbolic_optics.rcwa`. FMMAX is not a dependency of the
package, so the command adds it for this one run:

```bash
uv run --with fmmax python benchmarks/fmmax/compare.py
```

(With a uv config that points at a private index, prefix `UV_NO_CONFIG=1`.) It
takes about 5 minutes on a laptop CPU and writes two figures to `output/` (not
committed):

- `convergence.png`: for each case, how far each code moves as orders are
  added, and how far apart the two codes are.
- `spectra.png`: zeroth-order R and T and the first diffracted order against
  wavelength, from both codes at 41 orders. The MoO₃ panel uses MoO₃'s real
  dispersion.

The script exits non-zero if the two codes disagree by more than 5e-4 once
converged.

## What is compared

- **Efficiency of every diffraction order**, summed over outgoing polarization,
  for two incident states. Summing makes the comparison independent of either
  code's polarization basis.
- **The same incident states in both codes.** FMMAX's uniform-layer amplitudes
  are the tangential H components, Hx block first; the script checks that
  against Fresnel. hyperbolic-optics forms the same Hx-only and Hy-only states
  from its p and s waves.
- **Isotropic media and anisotropy in the plane of the layers**, which is what
  FMMAX handles. That includes in-plane hyperbolic and gyrotropic
  (non-symmetric) tensors. Full tensors with xz / yz terms are covered by
  `benchmarks/hyperbolic_grating/` and `tests/test_rcwa.py`.
- **Geometry.** hyperbolic-optics uses the exact Fourier coefficients of the
  step profile; FMMAX takes an FFT of a 2000-sample grid. FMMAX runs its
  `NORMAL` vector formulation.

## Results

"Change to 161" is each code's difference from its own 161-order result. The
last column compares the two codes once both are converged.

| case | orders | vs FMMAX | hyperbolic-optics change to 161 | FMMAX change to 161 | converged comparison |
|---|---|---|---|---|---|
| silicon ridge, 20° | 21 / 41 / 81 | 7e-7 / 6e-7 / 6e-7 | 9e-4 / 6e-5 / 4e-5 | same | 5e-7 at 161 |
| silicon ridge, conical | 21 / 41 / 81 | 4e-7 / 4e-7 / 4e-7 | 7e-4 / 1e-4 / 3e-5 | same | 4e-7 at 161 |
| gold ridge (ε = −25 + 1.5i), TM | 21 / 41 / 81 | 2e-6 / 2e-5 / 2e-5 | 1e-2 / 1e-2 / 2e-3 | same | 7e-6 at 161 |
| LiNbO₃, axis in the plane, conical | 21 / 41 / 81 | 3e-4 / 2e-4 / 8e-5 | 5e-4 / 5e-5 / 9e-6 | 2e-4 / 6e-5 / 3e-5 | **1e-5** vs FMMAX at ∞ |
| α-MoO₃ 900 cm⁻¹, in-plane hyperbolic | 21 / 41 / 81 | 5e-3 / 3e-3 / 2e-3 | 3e-3 / 7e-4 / 2e-4 | 5e-3 / 2e-3 / 8e-4 | **1e-4** vs FMMAX at ∞ |
| gyrotropic, ε_xy = −ε_yx = 0.8i | 21 / 41 / 81 | 2e-3 / 9e-4 / 5e-4 | 2e-4 / 4e-5 / 9e-6 | 2e-3 / 6e-4 / 2e-4 | **1e-5** vs FMMAX at ∞ |

- **Isotropic gratings.** Both codes implement the same factorisation, so they
  produce the same truncated answer: 4e-7–2e-5 apart at every truncation, far
  inside either one's truncation error.
- **Anisotropic, hyperbolic and gyrotropic gratings.** The factorisations differ
  here: this package uses Li's anisotropic rule, FMMAX a normal-vector
  formulation.
  - At a fixed truncation the codes differ by about FMMAX's truncation error,
    which is 3–25 times larger than this package's.
  - Extrapolating FMMAX to infinitely many orders (Richardson, 1/N) brings it to
    within 1e-5 of this package's converged result for LiNbO₃ and the gyrotropic
    ridge, and within 1e-4 for hyperbolic MoO₃.
- **Spectra** (`spectra.png`, 41 orders):
  - isotropic cases agree to 4e-6–5e-5
  - anisotropic ones to 1e-3
  - hyperbolic MoO₃ to 1.5e-2, worst at 850 cm⁻¹, where it is most strongly
    hyperbolic. Between 41 and 161 orders FMMAX's own value there moves by
    1.3e-2, this package's by 6e-4.
- **Energy conservation** in the lossless cases: 1e-14–1e-13 at 21–41 orders,
  1e-12–3e-11 at 81.

Timing, for 41 orders, one wavelength and one grating layer on a CPU, both run
without `jax.jit`: about 85 ms here and about 250 ms for FMMAX. FMMAX's figure
includes the FFT of its 2000-sample grid. Neither has been tuned, so this is not
a performance comparison.
