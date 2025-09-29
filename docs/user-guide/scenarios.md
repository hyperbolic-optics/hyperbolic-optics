# Scenarios

The package supports four different scenario types for different analysis needs.

## Simple Scenario

Single point calculation at a specific angle, frequency, and orientation.

```python
payload = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 45.0,      # degrees
        "azimuthal_angle": 0.0,      # degrees  
        "frequency": 1460.0          # cm^-1
    },
    "Layers": [...]
}
```

**Output shape**: Scalar values for reflection coefficients

**Use case**: Quick calculations, debugging, single point analysis

## Incident Scenario

Sweeps through incident angles at multiple frequencies.

```python
payload = {
    "ScenarioData": {
        "type": "Incident"
    },
    "Layers": [...]
}
```

**Output shape**: `[410, 360]` - 410 frequency points × 360 angle points

**Use case**: Analyzing angle-dependent behavior, finding resonances

**Plotting**: Use `plot_kx_frequency()` for frequency vs kx plots

## Azimuthal Scenario  

Rotates the sample azimuthally at a fixed incident angle.

```python
payload = {
    "ScenarioData": {
        "type": "Azimuthal",
        "incidentAngle": 40.0  # degrees
    },
    "Layers": [...]
}
```

**Output shape**: `[410, 360]` - 410 frequency points × 360 azimuthal angles

**Use case**: Studying rotational symmetry, in-plane anisotropy

**Plotting**: Use `plot_mueller_azimuthal()` for frequency vs β plots

## Dispersion Scenario

k-space dispersion at a fixed frequency.

```python
payload = {
    "ScenarioData": {
        "type": "Dispersion",
        "frequency": 1460.0  # cm^-1
    },
    "Layers": [...]
}
```

**Output shape**: `[180, 480]` - 180 incident angles × 480 azimuthal angles

**Use case**: Visualizing polariton dispersion, identifying wave modes

**Plotting**: Use `plot_mueller_dispersion()` for kx vs ky plots

## Choosing a Scenario

| Goal | Scenario Type |
|------|---------------|
| Quick single calculation | Simple |
| Angle-dependent analysis | Incident |
| Rotational behavior | Azimuthal |
| k-space dispersion relations | Dispersion |

## Frequency Ranges

For `Incident` and `Azimuthal` scenarios, frequency ranges are automatically determined by the material:

- **Calcite**: 1300-1600 cm⁻¹ (upper) or 860-920 cm⁻¹ (lower)
- **Quartz**: 410-600 cm⁻¹
- **Sapphire**: 210-1000 cm⁻¹
- **Gallium Oxide**: 350-800 cm⁻¹

## Example: Comparing Scenarios

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.plots import plot_kx_frequency, plot_mueller_dispersion

# Same material, different scenarios
base_layers = [
    {"type": "Ambient Incident Layer", "permittivity": 25.0},
    {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
    {
        "type": "Semi Infinite Anisotropic Layer",
        "material": "Calcite",
        "rotationX": 0,
        "rotationY": 70,
        "rotationZ": 0
    }
]

# Incident scenario
payload_incident = {
    "ScenarioData": {"type": "Incident"},
    "Layers": base_layers
}
structure_inc = Structure()
structure_inc.execute(payload_incident)
R_inc = abs(structure_inc.r_pp)**2

# Dispersion scenario
payload_disp = {
    "ScenarioData": {"type": "Dispersion", "frequency": 1460.0},
    "Layers": base_layers
}
structure_disp = Structure()
structure_disp.execute(payload_disp)
R_disp = abs(structure_disp.r_pp)**2

# Plot both
plot_kx_frequency(structure_inc, R_inc, save_name="incident")
plot_mueller_dispersion(structure_disp, R_disp, save_name="dispersion")
```