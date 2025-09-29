# Mueller Matrices

Mueller matrices describe how light's polarization state changes upon reflection from a surface.

## Basic Usage

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller

# Create structure
structure = Structure()
structure.execute(payload)

# Create Mueller analyzer
mueller = Mueller(structure)

# Set incident polarization
mueller.set_incident_polarization('linear', angle=45)

# Add the sample
mueller.add_optical_component('anisotropic_sample')

# Calculate results
params = mueller.get_all_parameters()
```

## Incident Polarization States

### Linear Polarization

```python
# p-polarized (0°)
mueller.set_incident_polarization('linear', angle=0)

# s-polarized (90°)
mueller.set_incident_polarization('linear', angle=90)

# 45° linear
mueller.set_incident_polarization('linear', angle=45)
```

### Circular Polarization

```python
# Right-handed circular
mueller.set_incident_polarization('circular', handedness='right')

# Left-handed circular
mueller.set_incident_polarization('circular', handedness='left')
```

### Elliptical Polarization

```python
# Elliptical with azimuth 30° and ellipticity 20°
mueller.set_incident_polarization('elliptical', alpha=30, ellipticity=20)
```

## Optical Components

### Anisotropic Sample

The main sample being analyzed:

```python
mueller.add_optical_component('anisotropic_sample')
```

### Linear Polarizer

```python
# Horizontal polarizer (0°)
mueller.add_optical_component('linear_polarizer', 0)

# Vertical polarizer (90°)
mueller.add_optical_component('linear_polarizer', 90)

# 45° polarizer
mueller.add_optical_component('linear_polarizer', 45)
```

### Quarter-Wave Plate

```python
# QWP with fast axis at 45°
mueller.add_optical_component('quarter_wave_plate', 45)
```

### Half-Wave Plate

```python
# HWP with fast axis at 22.5°
mueller.add_optical_component('half_wave_plate', 22.5)
```

## Multiple Components

Place components in series:

```python
# Polarizer → Sample → Analyzer
mueller.set_incident_polarization('linear', angle=0)
mueller.add_optical_component('linear_polarizer', 0)  # Input polarizer
mueller.add_optical_component('anisotropic_sample')    # Sample
mueller.add_optical_component('linear_polarizer', 90)  # Analyzer (crossed)

reflectivity = mueller.get_reflectivity()
```

## Stokes Parameters

The Stokes parameters describe the polarization state:

- **S0**: Total intensity (reflectance)
- **S1**: Horizontal vs vertical linear polarization
- **S2**: +45° vs -45° linear polarization
- **S3**: Right vs left circular polarization

```python
stokes = mueller.get_stokes_parameters()

print(f"Total intensity: {stokes['S0']}")
print(f"Linear H/V: {stokes['S1']}")
print(f"Linear ±45°: {stokes['S2']}")
print(f"Circular R/L: {stokes['S3']}")
```

## Polarization Properties

### Degree of Polarization (DOP)

```python
dop = mueller.get_degree_of_polarisation()
# DOP = 0: unpolarized
# DOP = 1: fully polarized
```

### Ellipticity

```python
ellipticity = mueller.get_ellipticity()
# Ellipticity angle in radians
# 0: linear
# ±π/4: circular
```

### Azimuth

```python
azimuth = mueller.get_azimuth()
# Orientation of polarization ellipse major axis
```

## All Parameters at Once

```python
params = mueller.get_all_parameters()

# Contains: S0, S1, S2, S3, DOP, Ellipticity, Azimuth
for key, value in params.items():
    print(f"{key}: {value}")
```

## Complete Example

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller

# Define structure
payload = {
    "ScenarioData": {
        "type": "Incident"
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 50.0},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 0,
            "rotationY": 70,
            "rotationZ": 0
        }
    ]
}

# Run simulation
structure = Structure()
structure.execute(payload)

# Mueller analysis
mueller = Mueller(structure)

# Try different incident polarizations
for angle in [0, 45, 90]:
    mueller.set_incident_polarization('linear', angle=angle)
    mueller.add_optical_component('anisotropic_sample')
    
    params = mueller.get_all_parameters()
    print(f"\nIncident angle: {angle}°")
    print(f"Average reflectance: {params['S0'].mean():.4f}")
    print(f"Average DOP: {params['DOP'].mean():.4f}")
    
    mueller.reset()  # Reset for next calculation
```

## Resetting the Mueller Object

```python
# Clear all settings and start fresh
mueller.reset()
```

This clears:
- Mueller matrix
- Stokes parameters
- Optical components
- Resets incident polarization to unpolarized

## Physical Interpretation

### Reflectance (S0)

Total reflected power, sum of all polarization components.

### Degree of Polarization

- **DOP = 1**: Fully polarized (pure state)
- **0 < DOP < 1**: Partially polarized
- **DOP = 0**: Unpolarized (random)

### Understanding S1, S2, S3

- **S1 > 0**: More horizontal than vertical
- **S1 < 0**: More vertical than horizontal
- **S2 > 0**: More +45° than -45°
- **S2 < 0**: More -45° than +45°
- **S3 > 0**: More right-circular
- **S3 < 0**: More left-circular