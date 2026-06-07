# Hyperbolic Optics Simulation Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/hyperbolic-optics.svg)](https://badge.fury.io/py/hyperbolic-optics)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/hyperbolic-optics/hyperbolic-optics/actions/workflows/tests.yml/badge.svg)](https://github.com/hyperbolic-optics/hyperbolic-optics/actions/workflows/tests.yml)
[![Issues](https://img.shields.io/github/issues/hyperbolic-optics/hyperbolic-optics)](https://github.com/hyperbolic-optics/hyperbolic-optics/issues)

This package provides a comprehensive suite of tools to study the reflective properties of hyperbolic materials and anisotropic structures using the 4×4 transfer matrix method. It enables easy configuration of multilayer systems, calculation of reflection coefficients, and analysis using Mueller matrices.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [Citation](#citation)
- [Known Issues / Limitations](#known-issues--limitations)
- [Papers & Further Reading](#papers--further-reading)
- [License](#license)
- [Getting Help](#getting-help)

---

## Features

- **4×4 transfer-matrix engine** for reflection coefficients of arbitrary anisotropic multilayers, plus an opt-in **numerically-stable scattering-matrix backend** for thick / lossy / evanescent stacks
- **Transmission, layer-resolved absorption, and field profiles** (E, H, Sₙ vs depth) computed numerically from the Poynting flux — energy-conserving `R + T + ΣA = 1`
- **Polarization toolkit:** Mueller *and* Jones calculus, ellipsometry (Ψ/Δ), eigenpolarizations / exceptional points, co- vs cross-polarized power, and Poincaré-sphere trajectories
- **Materials library:** Quartz, Calcite, Sapphire, Ga₂O₃ (monoclinic), α-MoO₃ (biaxial), AlN, SiC, hBN, GaN — plus arbitrary permittivity/permeability tensors
- **Scenarios:** incident-angle, azimuthal, k-space dispersion (kₓ–k_y), full 3-D sweep, single-point — and a swept **layer-thickness** axis
- **Fully vectorized** over angle, azimuth, frequency, and thickness via a canonical batch convention
- **Publication-quality plotting** for spectra, k-space maps, and polarization

---

## Installation

The package is now available on PyPI and can be installed using pip or uv:

### Using pip

```bash
pip install hyperbolic-optics
```

### Using uv (recommended for modern Python development)

```bash
uv add hyperbolic-optics
```

### Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/hyperbolic-optics/hyperbolic-optics.git
cd hyperbolic_optics
pip install -e .
```

---

## Quick Start

Here's a simple example to get you started:

```python
import json
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller

# Define a simple multilayer structure
payload = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 45.0,  # degrees
        "azimuthal_angle": 0.0,  # degrees
        "frequency": 1460.0      # cm^-1
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 50.0
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0
        }
    ]
}

# Create and execute the simulation
structure = Structure()
structure.execute(payload)

# Calculate reflectivities
R_pp = abs(structure.r_pp)**2
R_ss = abs(structure.r_ss)**2
print(f"p-polarized reflectivity: {R_pp}")
print(f"s-polarized reflectivity: {R_ss}")

# Perform Mueller matrix analysis
mueller = Mueller(structure)
mueller.set_incident_polarization('linear', angle=45)
mueller.add_optical_component('anisotropic_sample')

# Get Stokes parameters and polarization properties
all_params = mueller.get_all_parameters()
print(f"Reflectance (S0): {all_params['S0']}")
print(f"Degree of polarization: {all_params['DOP']}")
```

---

## Usage Examples

The `examples/` folder contains simple scripts demonstrating various capabilities.

Run any example with:
```bash
python examples/calcite.py
```

---

## Advanced Features

### Custom Materials

Define materials with arbitrary permittivity and permeability tensors:

```python
custom_material = {
    "eps_xx": {"real": 2.27, "imag": 0.001},
    "eps_yy": {"real": -4.84, "imag": 0.755}, 
    "eps_zz": {"real": -4.84, "imag": 0.755},
    "eps_xy": {"real": 0.0, "imag": 0.0},
    "eps_xz": {"real": 0.0, "imag": 0.0},
    "eps_yz": {"real": 0.0, "imag": 0.0},
}
```

### Multiple Scenario Types

- **Incident:** Frequency vs incident angle analysis
- **Azimuthal:** Frequency vs azimuthal rotation analysis  
- **Dispersion:** k-space dispersion at fixed frequency
- **Simple:** Single-point calculation for specific conditions

### Built-in Visualization

```python
from hyperbolic_optics.plots import plot_mueller_dispersion, plot_kx_frequency

# Generate publication-quality plots
plot_kx_frequency(structure, reflectivity, save_name="my_plot")
```

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs and requesting features
- Setting up a development environment
- Code style and testing requirements
- Submitting pull requests

---

## Citation

If you use this package in your research, please cite:

### Software Citation
```bibtex
@software{cunningham2025hyperbolic,
  title={Hyperbolic Optics Simulation Package},
  author={Mark Cunningham},
  year={2026},
  version={0.3.0},
  doi={10.5281/zenodo.14946556},
  url={https://pypi.org/project/hyperbolic-optics/},
  howpublished={PyPI},
  note={Python package for 4×4 transfer matrix method simulations}
}
```

### Related Publications
This package was used to generate results in:

**M. Cunningham et al.**, "Optical footprint of ghost and leaky hyperbolic polaritons," *Photonics Research*, vol. 13, no. 8, pp. 2291-2305 (2025). DOI: [10.1364/PRJ.558334](https://doi.org/10.1364/PRJ.558334)

---

## Transmission, Absorption & Field Profiles

Reflection is computed automatically by `Structure.execute`. Power transmittance,
layer-resolved absorption, and field profiles are computed **numerically** from
the propagated fields (energy-conserving `R + T + ΣA = 1`) via `FieldProfile`:

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.fields import FieldProfile

structure = Structure()
structure.execute(payload)

fp = FieldProfile(structure)
print(fp.summary("p"))            # R, T, per-layer absorption, conservation residual
T = fp.transmittance("p")          # power transmittance (same shape as r_pp)
A = fp.layer_absorption("p")       # per-interior-layer absorptance
prof = fp.field_profile("p")       # z, Ex..Hz, S_z(z), cumulative absorption
```

For a single semi-infinite anisotropic layer there are no interior layers to
resolve, so `T = 1 − R` is the power delivered into the bulk, and `field_profile`
shows it being absorbed with depth.

### Polarization conversion

Cross-polarization (the `r_ps`/`t_ps` channels) can be analysed several ways:

- `FieldProfile.polarization_resolved("p")` *(experimental)* — splits `R` and `T`
  into co- and cross-polarized power and reports the conversion fractions. The
  transmitted s/p split is rigorous for an isotropic exit.
- `FieldProfile.stokes_from_field_profile("p")` — the Stokes vector and
  polarization ellipse (azimuth ψ, ellipticity χ) of the transverse field *versus
  depth*, so you can watch the state evolve through a birefringent layer.
- `Mueller.calculate_transmission_mueller_matrix()` — the transmission Mueller
  matrix from the `t` coefficients. Its `|t|²` intensity equals true power
  transmittance only for a **symmetric prism = substrate** system; into a
  lower-index medium it breaks in the evanescent regime (use
  `FieldProfile.transmittance` for power there).

`examples/layer_resolved_absorption.py` reproduces the azimuthal layer-resolved
absorption of a MoO₃/AlN/SiC heterostructure in the Otto geometry
(Passler, Jeannin & Paarmann, *J. Opt. Soc. Am. B* **37**, 1060 (2020)).
The amplitude transmission coefficients are also available via
`FieldProfile.transmission_coefficients()` (and `Structure.calculate_transmissivity()`).

## Numerical backends

`execute` defaults to the 4×4 **transfer-matrix** method, which is fast but
numerically unstable for thick / lossy / strongly-evanescent layers (the
propagation terms grow exponentially and the matrix product overflows to `NaN`).
For those cases, opt into the numerically-stable **scattering-matrix** backend:

```python
structure.execute(payload, backend="scattering")
```

It cascades per-layer scattering matrices with the Redheffer star product so only
*decaying* exponentials ever appear. It returns the same reflection/transmission
coefficients as the transfer method where that is well-conditioned, and correct
ones where the transfer matrix is (near-)singular — e.g. a thick evanescent Otto
gap, where `backend="transfer"` gives `NaN` but `backend="scattering"` correctly
gives total reflection. Method/conventions follow PyLlama (Bay *et al.*,
*J. Opt. Soc. Am. A* **39**, 1431 (2022)).

## Sweeping Layer Thickness

A layer's `thickness` may be a **list** instead of a scalar, which sweeps it as a
fourth canonical axis (`T`) alongside incident angle, azimuth, and frequency —
computed in a single `execute` (the eigendecomposition is thickness-independent;
only the propagation phase broadcasts over `T`). It composes with any scenario and
is absent (size 1) by default:

```python
{"type": "Crystal Layer", "material": "Calcite",
 "thickness": [0.5, 1.0, 1.5, 2.0], "rotationY": 90}   # -> appends a length-4 T axis
```

Outputs append the trailing `T` axis (e.g. `Simple` → `r_pp.shape == (4,)`,
`Incident` → `(F, angle, 4)`). At most one layer may carry a list thickness. For a
2-D thickness × thickness grid, combine a list thickness on one layer with the
`ThicknessSweep` helper (`from hyperbolic_optics import ThicknessSweep`), which
re-runs the stack and stacks results along a leading index.

## Known Issues / Limitations

- **Multiple Optical Components:** While you can place multiple Mueller matrix components in series, matching incident angles between them isn't yet implemented
- **Field profiles over full sweeps:** `field_profile` is batched but intended for `Simple` / single-point use — a full angle/frequency sweep times the depth axis is memory-heavy.

## Testing

This package includes a comprehensive test suite covering all major functionality. Run tests locally with:

```bash
pytest
pytest --cov=hyperbolic_optics --cov-report=html  # with coverage report
```

*Please open an [issue](https://github.com/hyperbolic-optics/hyperbolic-optics/issues) if you encounter any bugs or have suggestions for improvements.*

---

## Papers & Further Reading

For background on the physics and mathematical methods, see:

**Key References:**
- N. C. Passler and A. Paarmann, "Generalized 4 × 4 matrix formalism for light propagation in anisotropic stratified media," *J. Opt. Soc. Am. B* **34**, 2128-2139 (2017)
- P. Yeh, "Electromagnetic propagation in birefringent layered media," *J. Opt. Soc. Am.* **69**, 742-756 (1979)
- N. C. Passler et al., "Layer-resolved resonance intensity of evanescent polariton modes in anisotropic multilayers," *Phys. Rev. B* **107**, 235426 (2023)
- M. Cunningham et al., "Optical footprint of ghost and leaky hyperbolic polaritons," *Photonics Research* **13**, 2291-2305 (2025)

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Getting Help

- **Documentation:** Check the [examples/](examples/) folder and docstrings
- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/hyperbolic-optics/hyperbolic-optics/issues)
- **Discussions:** Start a discussion for usage questions or feature ideas

---

**Thank you for your interest in the Hyperbolic Optics Simulation Package!** 

*Happy simulating! 🔬✨*