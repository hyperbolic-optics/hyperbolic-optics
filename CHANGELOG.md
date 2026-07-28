# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Correctness work. Several of these change numbers that 0.3.0 reported, in
regimes 0.3.0 got wrong; the affected regimes are named in each entry.

### Fixed

- **Mode direction sorting** (`Wave.wave_sorting`) for layers carrying both
  propagating and evanescent modes. Two independent defects: the
  forward/backward ordering was built by splicing two sort permutations with
  `np.where`, which is not itself a permutation in the mixed case and silently
  duplicated and dropped modes; and propagating modes were classified by
  `sign(Re kz)` rather than `sign(S_z)`, which is the wrong test in a hyperbolic
  medium, where phase and energy velocity are antiparallel. Together these
  produced unphysical reflectance (`R > 1`) or `NaN` for lossless, hyperbolic
  and anisotropic-permeability media. Every built-in material is lossy and
  therefore unaffected.
- **p/s labelling** (`Wave.sort_poynting_indices`) ranked the two modes by their
  in-plane Poynting fraction descending *or* their field fraction ascending,
  choosing per batch point. The two orders are opposite, so which slot held the
  p-like mode — and hence whether a number was reported as `r_pp` or `r_ss` —
  depended on which branch a point took. The field ratio is now the primary
  criterion, with the flux ratio a tie-break sorted the same way.
- **Amplitude transmission coefficients** were reported in whatever
  normalisation LAPACK returned, scaled over the tangential 4-vector rather than
  over `E` and carrying an arbitrary phase that could flip sign along a sweep.
  Exit modes are now on a fixed basis (`E · E = 1`, dominant transverse
  component real positive), which reproduces the closed-form isotropic exit
  exactly, including past the critical angle. Reflection is unaffected.
- **α-MoO₃ had its [001] and [010] Reststrahlen bands on the wrong axes**, ε∞ on
  [010] was 2.47 rather than 4.47, and two of the three [100] oscillators were
  missing. Parameters are now Álvarez-Pérez et al., *Adv. Mater.* **32**, 1908176
  (2020), Table 1 in full. Because the mis-assignment put [010] — the van der
  Waals stacking axis — in-plane, `examples/layer_resolved_absorption.py` no
  longer needs its compensating `rotationX=90`.
- **hBN** parameters were a mixture of the two references cited; they are now the
  single self-consistent set from Caldwell et al., *Nat. Commun.* **5**, 5221 (2014).
- **Reflection coefficients are no longer invented by cancellation.** Every
  coefficient is a 2×2 minor over a denominator, and a layer carrying a growing
  exponential drives the assembled matrix towards rank 1, where all such minors
  vanish analytically. `r_pp` and `r_ss` survive that — numerator and denominator
  lose precision together and the error cancels in the ratio — but a coefficient
  that is zero by symmetry has no such partner, so its numerator cancels to the
  rounding floor and the noise is returned as a confident value. On the
  `multilayer_incident` battery payload `r_sp` came back as 9e-3 where symmetry
  forbids any p–s conversion, and the stack reflected 1.000076 of the incident
  power. Each subtraction now reports how much of it survived, and the affected
  batch points — 1.5% of that sweep — are recomputed with the Redheffer cascade,
  which reads the same eigenmodes but never forms a growing exponential.
  Well-conditioned points are untouched and bit-identical; the repair is
  best-effort and reports `Structure.repaired_fraction`. Pass
  `calculate_reflectivity(stabilize=False)` for the literal transfer product.
- Omitting `permittivity` on an isotropic exit layer silently produced an
  all-NaN spectrum: `np.float64(None)` is `nan` on NumPy 2, so the guard meant
  to catch it was unreachable.

### Added

- **Fully general ε/μ tensors**: `ArbitraryMaterial` accepts all nine components
  of each tensor (`eps_yx`, `eps_zx`, `eps_zy` and the `mu_` counterparts), so
  non-reciprocal media — gyrotropic permeability, magneto-optic permittivity —
  are expressible. Naming only the upper triangle keeps the previous symmetric
  behaviour, so existing payloads are unaffected.
- **Swept angular axes are controllable.** `polar_points` / `azimuthal_points`
  set the resolution (the design is taken from the Rust engine in the companion
  desktop app, which already exposed both), and `incidentAngle` /
  `azimuthal_angle` accept a list of degrees or a `{"min", "max", "points"}`
  dict. Defaults are unchanged. A bare scalar on a swept axis used to be
  accepted and then silently discarded -- a payload that looked like it fixed
  the angle actually swept the full default range -- and now warns.
- Structure **warns when a material is evaluated outside its fitted frequency
  range**. All layers share one grid, so a stack of two dispersive materials
  extrapolates at least one of them, where the factorized form can return
  `Im(ε) < 0` — gain from a passive crystal, seen as negative layer absorptance.
- Structure **warns when the transfer product has lost conditioning** — non-finite
  coefficients, or a passive stack reflecting more than it receives — naming
  `backend="scattering"`.
- `FieldProfile` now raises `NotImplementedError` when handed a
  `backend="scattering"` structure, instead of claiming it "has not been executed".
- Test batteries for regimes that had no coverage: lossless, hyperbolic and
  gyrotropic media (`test_general_tensors.py`), passivity and cross-validation
  against the closed-form isotropic path (`test_physical_invariants.py`), and
  shipped materials against their published band edges
  (`test_material_literature.py`) — the first test in the suite to compare a
  material with a literature number.

### Changed

- The scattering-vs-transfer cross-check now compares elements. It masked on
  `cond(transfer_matrix) < 1e10`, but a semi-infinite exit makes that matrix
  structurally rank-deficient, so `cond` is `inf` and the mask excused every
  point on 10 of 11 payloads. Enabling it exposed a transfer-backend precision
  defect, recorded as a strict `xfail`.
- CI: `ruff` now gates instead of `continue-on-error`, and the golden battery
  runs on a pinned platform rather than being deselected everywhere.
- Release: `publish.yml` gates on the test suite and verifies the tag matches
  `__version__`. It no longer rewrites the version from `__init__.py` back into
  `__init__.py`, nor commits and pushes to `main` mid-release.

## [0.3.0] - 2026-06-07

A large feature release: power transport, field profiles, polarization analysis,
a stable solver, and more materials. All additions are backwards compatible
(reflection coefficients from `Structure.execute` are unchanged).

### Added

- **Numerical transmission, layer-resolved absorption, and field profiles**
  (`FieldProfile`): power transmittance `T`, per-layer absorptance, and
  `E/H/Sₙ(z)` reconstructed from the propagated field, energy-conserving
  `R + T + ΣA = 1`. Includes amplitude transmission coefficients.
- **Numerically-stable scattering-matrix backend**:
  `Structure.execute(payload, backend="scattering")` — a Redheffer star-product
  solver that stays correct for thick / lossy / evanescent stacks where the
  transfer-matrix product overflows.
- **Layer-thickness as a swept axis**: a list-valued layer `thickness` sweeps a
  fourth canonical batch axis; plus a `ThicknessSweep` helper.
- **Polarization toolkit**: a `Jones` amplitude-level calculus (eigenpolarizations,
  exceptional points, ellipsometric Ψ/Δ), co- vs cross-polarized power, a
  transmission Mueller matrix, Stokes-vs-depth, Lu–Chipman Mueller decomposition,
  `compose_jones` for optical systems in series, and a Poincaré-sphere plot.
- **Materials**: α-MoO₃ (biaxial), AlN, SiC, hBN, GaN; a `BiaxialMaterial` base
  class and a `list_materials()` catalogue.

### Changed

- Canonical batch convention extended to `[A, B, F, T]` via a single parametric
  `axes.canonicalize` adapter.
- Corrected α-MoO₃ high-frequency permittivities to the Álvarez-Pérez values.

### Removed

- Unused `scipy` runtime dependency and the dead `main.py` stub.

## [0.2.3] - 2025-09-30

Baseline release: 4×4 transfer-matrix reflection coefficients for anisotropic
multilayers, Mueller-matrix analysis, the built-in materials library
(Quartz, Calcite, Sapphire, Ga₂O₃), and the Incident / Azimuthal / Dispersion /
Simple / FullSweep scenarios.

[Unreleased]: https://github.com/hyperbolic-optics/hyperbolic-optics/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/hyperbolic-optics/hyperbolic-optics/releases/tag/v0.3.0
[0.2.3]: https://github.com/hyperbolic-optics/hyperbolic-optics/releases
