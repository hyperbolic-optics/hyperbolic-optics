# Full-tensor hyperbolic grating (tilted-axis hBN)

Checks `hyperbolic_optics.rcwa` on a case no independent grating code here can
model:

- **Material:** hBN in its upper Reststrahlen band (1360–1620 cm⁻¹, ε⊥ < 0 < ε∥),
  using the package's own dispersion.
- **Orientation:** optic axis tilted 40° out of the plane of the layers and
  turned 25°, so every component of ε is nonzero.
- **Structure:** ridges with an 8 µm period, fill 0.5, 0.5 µm thick, on an
  ε = 2 substrate, at 20° incidence.

```bash
uv run python benchmarks/hyperbolic_grating/hbn_grating.py
```

It takes about 1 minute and writes `output/hbn_grating.png` (not committed).

## What the figure shows

| panel | check | result |
|---|---|---|
| (a) | the unpatterned film against the NumPy engine, an independent eigendecomposition solver, given the identical tensor | R and T for p and s input agree to 8e-15 |
| (b) | zeroth-order R and T, first diffracted order and absorption, at 41 and 81 orders | the two truncations differ visibly near 1450 and 1500 cm⁻¹ |
| (c) | convergence in Fourier order at three wavenumbers | 6e-3 at 81 orders at 1450 cm⁻¹: hyperbolic gratings converge slowly |
| (d) | first-order efficiency against fill factor at 41, 81 and 161 orders | a truncation ripple that shrinks with more orders |
| (e) | the exact gradient d/d(fill) against central finite differences | agrees to a median 1e-6 relative, but carries the ripple, which does not shrink |
| (f) | energy conservation of the same grating made lossless | 1e-15–1e-12 at 21–41 orders, rising to 3e-8 at 161 |

## What it means

- **Full-tensor hyperbolic media are handled correctly.** Panel (a) shows it
  against an independent solver. `tests/test_rcwa.py` adds agreement with an
  eigenmode S-matrix on the patterned layer (1e-10), plus mirror symmetries that
  flip the signs of the off-diagonal terms.
- **Convergence is slow**, because hyperbolic ridges carry propagating waves of
  very high in-plane wavevector. Use 81 or more orders.
- **Geometry gradients are rippled.** They are exact for the truncated model,
  but that model's dependence on edge position ripples, and the ripple in the
  slope does not fade with more orders. A silicon grating shows no such ripple.
  When optimising hyperbolic ridges, smooth the edges or filter the design.
