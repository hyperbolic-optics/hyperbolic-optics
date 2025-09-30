# Hyperbolic Optics Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MarkCunningham0410/hyperbolic_optics/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/hyperbolic-optics.svg)](https://badge.fury.io/py/hyperbolic-optics)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python package for simulating the reflective properties of hyperbolic materials and anisotropic structures using the 4×4 transfer matrix method.

## Features

- **Transfer Matrix Method**: Accurate 4×4 transfer matrix implementation for anisotropic media
- **Multiple Scenarios**: Support for incident angle sweeps, azimuthal rotations, dispersion analysis, and single-point calculations
- **Materials Library**: Pre-configured materials including Quartz, Calcite, Sapphire, and Gallium Oxide
- **Mueller Matrix Analysis**: Complete Stokes parameter and polarization analysis capabilities
- **Custom Materials**: Define arbitrary materials with custom permittivity and permeability tensors
- **Visualization**: Publication-quality plotting functions

## Quick Example

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller

# Define a simple structure
payload = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 45.0,
        "azimuthal_angle": 0.0,
        "frequency": 1460.0
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 50.0},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0
        }
    ]
}

# Run simulation
structure = Structure()
structure.execute(payload)

# Calculate reflectivities
R_pp = abs(structure.r_pp)**2
R_ss = abs(structure.r_ss)**2

print(f"p-polarized reflectivity: {R_pp:.4f}")
print(f"s-polarized reflectivity: {R_ss:.4f}")
```

## Installation

Install via pip:

```bash
pip install hyperbolic-optics
```

Or using uv (recommended):

```bash
uv add hyperbolic-optics
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running quickly
- [User Guide](user-guide/concepts.md) - Learn about core concepts
- [API Reference](api/structure.md) - Detailed API documentation
- [Examples](examples/basic.md) - See more examples

## Citation

If you use this package in your research, please cite:

```bibtex
@software{cunningham2025hyperbolic,
  title={Hyperbolic Optics Simulation Package},
  author={Mark Cunningham},
  year={2025},
  version={0.2.3},
  doi={10.5281/zenodo.14946556},
  url={https://pypi.org/project/hyperbolic-optics/}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/MarkCunningham0410/hyperbolic_optics/issues)
- **Repository**: [GitHub Repository](https://github.com/MarkCunningham0410/hyperbolic_optics)
