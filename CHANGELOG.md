# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] - unreleased

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

[0.3.0]: https://github.com/hyperbolic-optics/hyperbolic-optics/releases
[0.2.3]: https://github.com/hyperbolic-optics/hyperbolic-optics/releases
