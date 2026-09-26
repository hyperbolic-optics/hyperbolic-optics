# Liquid-crystal polarization grating (arXiv:2608.06185, Fig. 4)

A full-tensor case beyond FMMAX: the liquid-crystal director tilts out of the
plane of the layers, so the permittivity has xz and yz components, and it
rotates along both x and z. This is the transmissive grating of Yang, Song & Cai,
Fig. 4(c):

- n_o = 1.525, n_e = 1.775
- Λx = 400 nm, thickness 1 µm, twist Ω = 217°/µm
- 20 z-slices, as in the paper
- substrate index 1.7

```bash
uv run python benchmarks/paper_fig4/lc_polarization_grating.py
```

It writes `output/lc_grating.png` (not committed): first-order efficiency
against wavelength at normal incidence, and against angle of incidence at
530 nm, for tilts Θ = 0°, 25° and 50°. The −1 order switches off at 680 nm,
where it becomes evanescent in the substrate (λ = n_sub·Λx). That's a physical
check independent of the paper.

## Assumptions

The paper does not state the incident medium or the incident polarization.
This assumes normal incidence from air with circular polarization, and reports
the power diffracted into the ±1 orders. The director follows the paper's
Eqs. 13–14 exactly.

## Result, compared with the paper's FEM curves

Values here are from the script. The paper's are read by eye off Fig. 4(c),
right column.

| Θ (out-of-plane tilt) | here: peak RCP → −1 | here: LCP into either first order | paper, peak |
|---|---|---|---|
| 0° | 0.95 at 500 nm | ≤ 0.05 | ~0.95–1.0 |
| 25° | 0.81 at 530–550 nm | ≤ 0.04 | ~0.6 |
| 50° | 0.27 at 500 nm | ≤ 0.02 | ~0.2 |

- The circular-handedness selectivity and the fall in efficiency with tilt
  agree, and so do the Θ = 0° and 50° values.
- **Θ = 25° does not agree** (0.81 here, about 0.6 in the paper). The value here
  is converged in Fourier order and x-sampling, moves by only 0.006 from 20 to
  80 slices, and shifts by at most 0.04 if the light comes from the substrate
  side instead of from air. The discrepancy therefore likely lies in a
  definition the paper does not give: the incident polarization, how Θ is
  measured, or the twist handedness.
- **This is not a quantitative reproduction.** Resolving it needs the paper's
  code or supplement.
