# Hyperbolic Optics Simulation Package

This Python package provides a set of tools for studying the reflective properties of hyperbolic materials and anisotropic structures using the 4x4 transfer matrix method. It allows for easy configuration of multilayer systems, calculation of reflection coefficients, and analysis of different polarization configurations.

## Features

- Simulate the reflective properties of hyperbolic materials and anisotropic structures
- Configure multilayer systems with different materials and layer properties
- Calculate reflection coefficients using the 4x4 transfer matrix method
- Analyze different polarization configurations using Mueller matrices
- Visualize results with basic plotting functionality

## Installation

To install the package directly from your private GitHub repository, run the following command:
TODO: need to figure out how other people will be able to use this.

## Usage

To use the hyperbolic optics simulation package, follow these steps:

1. Create a payload in JSON format that describes your multilayer structure and scenario. For example:
```python
import json

def mock_incident_payload():
    payload = json.dumps({
        "ScenarioData": {
            "type": "Incident",
        },
        "Layers": [
            {
                "type": "Ambient Incident Layer",
                "permittivity": 5.5
            },
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 1.5
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Quartz",
                "rotationX": 0,
                "rotationY": 70,
                "rotationZ": 45,
            }
        ],
    })
    return payload
```

2. In your main script, import the necessary modules and load the payload

```python
import json
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from payloads import mock_incident_payload
from hyperbolic_optics.plots import contour_plot_mueller_incidence

def main():
    payload = json.loads(mock_incident_payload())
    # ...
```

3. Create a `Structure` object and execute it with the loaded payload:

```python
structure = Structure()
structure.execute(payload)
```

4. Create a `Mueller` object and add optical components, such as a linear polariser and the anisotropic sample. Then calculate the overall reflectivity:

```python
mueller = Mueller()
mueller.add_optical_component("linear_polarizer", 45)  # First polarizer
mueller.add_optical_component(
    "anisotropic_sample",
    structure.r_pp,
    structure.r_ps,
    structure.r_sp,
    structure.r_ss,
)

reflectivity = mueller.calculate_reflectivity() #### DEPRECATED - NEEDS CHANGED
contour_plot_mueller_incidence(structure, reflectivity)
```

For more detailed usage and examples, please refer to the documentation (which I do plan on making eventually).
Also, I plan on adding more functionality for optical components and manipulating incident polarisation better (rather than using optical components before the structure).


## Package Structure

- `hyperbolic_optics/`
  - `anisotropy_utils.py`: Utilities for applying rotations on anisotropic dielectric tensor matrices
  - `device_config.py`: Configuration for TensorFlow (work in progress)
  - `layers.py`: Handles the assembly of each layer in a multilayer structure for the 4x4 transfer matrix method
  - `material_params.py`: Stores phonon resonance parameters for each anisotropic material
  - `mueller.py`: Converts reflection coefficients into Mueller matrices and adds optical components
  - `plots.py`: Basic visualization functionality (to be moved out of the package)
  - `structure.py`: Binds layers together, calculates reflection coefficients
  - `waves.py`: Handles wave solutions, sorting, and Poynting vector calculations for each layer
  - `scenario.py`: Chooses between different scenarios (incident angle, azimuthal angle, or specific frequency polar plots)
- `script.py`: Main script
- `setup.py`: Package setup script
- `payloads.py`: Constructs the multilayer system configuration


## Known Issues/Limitations
1. Does not handle transmission coefficients well yet.
2. Mueller Matrix anisotropic structures can be placed in series, but not matching up incident angles etc. I.e. an incident angle for structure 1 cannot yet be paired with a different incident angle for structure 2.
3. There are no unit tests or anything yet. General best practices need to be adopted.

## Papers to read more about this
Will update this more - but Nikolai Christian Passler is the best port of call.

