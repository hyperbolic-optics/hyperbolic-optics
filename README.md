# Hyperbolic Optics Simulation Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/hyperbolic-optics.svg)](https://badge.fury.io/py/hyperbolic-optics)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/MarkCunningham0410/hyperbolic_optics/actions/workflows/tests.yml/badge.svg)](https://github.com/MarkCunningham0410/hyperbolic_optics/actions/workflows/tests.yml)
[![Issues](https://img.shields.io/github/issues/MarkCunningham0410/hyperbolic_optics)](https://github.com/MarkCunningham0410/hyperbolic_optics/issues)

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

- **Simulation of Reflective Properties:** Analyze how hyperbolic materials and anisotropic structures reflect light
- **Multilayer Configuration:** Configure multilayer systems with customizable materials and layer properties
- **4×4 Transfer Matrix Method:** Compute reflection coefficients accurately for anisotropic media
- **Mueller Matrix Analysis:** Convert reflection coefficients into Mueller matrices and simulate optical component interactions
- **Built-in Materials Library:** Pre-configured materials including Quartz, Calcite, Sapphire, Gallium Oxide
- **Arbitrary Material Support:** Define custom materials with arbitrary permittivity tensor
- **Multiple Scenario Types:** Support for incident angle sweeps, azimuthal rotations, dispersion analysis, and single-point calculations
- **Visualization:** Publication-quality plotting functionality for results analysis
- **Extensible Architecture:** Modular design for easy extension with new materials and optical components

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
git clone https://github.com/MarkCunningham0410/hyperbolic_optics.git
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
python examples/basic_calcite_example.py
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
  year={2025},
  version={0.2.3},
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

## Known Issues / Limitations

- **Transmission Coefficients:** Currently, transmission coefficients are not fully supported
- **Multiple Optical Components:** While you can place multiple Mueller matrix components in series, matching incident angles between them isn't yet implemented

## Testing

This package includes a comprehensive test suite with 93+ tests covering all major functionality. Run tests locally with:
```bash
pytest
pytest --cov=hyperbolic_optics --cov-report=html  # With coverage report

*Please open an [issue](https://github.com/MarkCunningham0410/hyperbolic_optics/issues) if you encounter any bugs or have suggestions for improvements.*

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
- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/MarkCunningham0410/hyperbolic_optics/issues)
- **Discussions:** Start a discussion for usage questions or feature ideas

---

**Thank you for your interest in the Hyperbolic Optics Simulation Package!** 

*Happy simulating! 🔬✨*