# Quick Start

## Your First Simulation

Let's run a simple simulation to calculate reflection coefficients for a Calcite crystal.

```python
from hyperbolic_optics.structure import Structure

# Define the structure
payload = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 45.0,      # degrees
        "azimuthal_angle": 0.0,      # degrees
        "frequency": 1460.0          # cm^-1
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 50.0
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.1,  # mm
            "permittivity": 1.0
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

# Run simulation
structure = Structure()
structure.execute(payload)

# Get reflection coefficients
r_pp = structure.r_pp  # p-to-p
r_ss = structure.r_ss  # s-to-s
r_ps = structure.r_ps  # p-to-s
r_sp = structure.r_sp  # s-to-p

# Calculate reflectivities
R_pp = abs(r_pp)**2
R_ss = abs(r_ss)**2

print(f"p-polarized reflectivity: {R_pp:.4f}")
print(f"s-polarized reflectivity: {R_ss:.4f}")
```

## Adding Mueller Matrix Analysis

```python
from hyperbolic_optics.mueller import Mueller

# Use the structure from above
mueller = Mueller(structure)

# Set incident polarization
mueller.set_incident_polarization('linear', angle=45)

# Add the sample
mueller.add_optical_component('anisotropic_sample')

# Get all parameters
params = mueller.get_all_parameters()

print(f"Reflectance (S0): {params['S0']:.4f}")
print(f"Degree of polarization: {params['DOP']:.4f}")
```

## Running Different Scenarios

### Incident Angle Sweep

```python
payload = {
    "ScenarioData": {
        "type": "Incident",  # Sweeps through angles
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 12.5},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0
        }
    ]
}

structure = Structure()
structure.execute(payload)

# Results are now arrays: shape (410, 360)
# 410 frequency points x 360 angle points
```

### Plotting Results

```python
from hyperbolic_optics.plots import plot_kx_frequency

# Calculate total reflectivity
R_total = abs(structure.r_pp)**2 + abs(structure.r_ps)**2

# Generate plot
plot_kx_frequency(structure, R_total, save_name="my_simulation")
```

## Next Steps

- Learn about [different scenarios](../user-guide/scenarios.md)
- Explore [available materials](../user-guide/materials.md)
- See more [examples](../examples/basic.md)
- Check the [API reference](../api/structure.md)