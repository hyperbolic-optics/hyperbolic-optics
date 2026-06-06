# Materials

## Built-in Materials

### Uniaxial Materials

Materials with a single optical axis (extraordinary axis differs from ordinary axes).

#### Quartz


```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": "Quartz",
    "rotationX": 0,
    "rotationY": 90,  # Optical axis orientation
    "rotationZ": 0
}
```


- **Frequency range**: 410-600 cm⁻¹
- **Type**: Uniaxial positive
- **Applications**: Mid-infrared phonon polaritons

#### Calcite

Two variants available:

**Upper Reststrahlen Band**:

```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": "Calcite",  # or "CalciteUpper"
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0
}
```

- **Frequency range**: 1300-1600 cm⁻¹
- **Type**: Uniaxial negative (hyperbolic)

**Lower Reststrahlen Band**:

```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": "CalciteLower",
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0
}
```

- **Frequency range**: 860-920 cm⁻¹
- **Type**: Uniaxial positive

#### Sapphire

```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": "Sapphire",
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0
}
```

- **Frequency range**: 210-1000 cm⁻¹
- **Type**: Uniaxial negative
- **Applications**: Wide-band phonon polaritons

#### AlN and SiC

Polar uniaxial crystals used (with MoO₃) in the layer-resolved absorption example.
Both are modeled c-cut; SiC is treated as effectively isotropic in its
reststrahlen band.

```python
{"type": "Semi Infinite Anisotropic Layer", "material": "SiC", "rotationY": 0}
{"type": "Crystal Layer", "material": "AlN", "thickness": 1.0, "rotationY": 0}
```

- **Type**: Uniaxial (AlN), near-isotropic (SiC)
- **Note**: phonon parameters in `material_params.json` are literature values
  flagged for verification.

### Monoclinic Materials

#### Gallium Oxide (β-Ga₂O₃)

```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": "GalliumOxide",
    "rotationX": 0,
    "rotationY": 90,
    "rotationZ": 0
}
```

- **Frequency range**: 350-800 cm⁻¹
- **Type**: Monoclinic (non-zero off-diagonal elements)
- **Special**: Has xy coupling (ε_xy ≠ 0)

### Biaxial Materials

#### α-MoO₃ (molybdenum trioxide)

```python
{
    "type": "Crystal Layer",
    "material": "MoO3",
    "thickness": 1.2,
    "rotationY": 0,
    "rotationZ": 0  # azimuthal rotation exposes the in-plane anisotropy
}
```

- **Type**: Orthorhombic biaxial — a **diagonal** tensor with three distinct
  principal permittivities (no off-diagonal coupling, unlike monoclinic Ga₂O₃).
- **Special**: strong in-plane anisotropy supports azimuth-dependent hyperbolic
  phonon polaritons (see `examples/layer_resolved_absorption.py`).
- **Note**: parameters in `material_params.json` are literature values flagged
  for verification.

## Isotropic Materials

### Air

Used for air gaps and isotropic layers:

```python
{
    "type": "Isotropic Middle-Stack Layer",
    "thickness": 0.5,
    "permittivity": 1.0  # Air
}
```

### Custom Isotropic


```python
{
    "type": "Isotropic Middle-Stack Layer",
    "thickness": 0.5,
    "permittivity": {"real": 2.5, "imag": 0.1}
}
```

## Arbitrary Materials

Define custom materials with full tensors:


```python
{
    "type": "Semi Infinite Anisotropic Layer",
    "material": {
        "eps_xx": {"real": 2.27, "imag": 0.001},
        "eps_yy": {"real": -4.84, "imag": 0.755},
        "eps_zz": {"real": -4.84, "imag": 0.755},
        "eps_xy": {"real": 0.0, "imag": 0.0},
        "eps_xz": {"real": 0.0, "imag": 0.0},
        "eps_yz": {"real": 0.0, "imag": 0.0}
    },
    "rotationX": 0,
    "rotationY": 0,
    "rotationZ": 0
}
```

## Material Rotations

### Rotation Types

Materials can be rotated using Euler angles (in degrees):


- **rotationX**: Rotation around x-axis (Not really used)
- **rotationY**: Rotation around y-axis (typically optical axis tilt)
- **rotationZ**: Rotation around z-axis (azimuthal)

### Rotation Examples

**Optical axis perpendicular to interface**:

```python
"rotationX": 0,
"rotationY": 0,
"rotationZ": 0
```

**Optical axis in the plane (70° tilt)**:

```python
"rotationX": 0,
"rotationY": 70,
"rotationZ": 0
```

**45° azimuthal rotation**:

```python
"rotationX": 0,
"rotationY": 90,
"rotationZ": 45
```

### Rotation Types for Z-rotation


```python
"rotationZ": 45,
"rotationZType": "relative"  # Default: rotates with azimuthal angle
```


or


```python
"rotationZ": 45,
"rotationZType": "static"  # Fixed: doesn't rotate with azimuthal angle
```

## Plotting Material Properties

```python
from hyperbolic_optics.materials import CalciteUpper
from hyperbolic_optics.plots import plot_permittivity

# Create material
calcite = CalciteUpper()

# Get permittivity
eps_ext, eps_ord = calcite.permittivity_fetch()

# Plot
plot_permittivity(calcite, eps_ext, eps_ord, save_name="calcite_permittivity")
```

## Material Properties Summary

| Material | Type | Frequency Range | Hyperbolic? |
|----------|------|-----------------|-------------|
| Quartz | Uniaxial | 410-600 cm⁻¹ | Yes |
| Calcite Upper | Uniaxial | 1300-1600 cm⁻¹ | Yes |
| Calcite Lower | Uniaxial | 860-920 cm⁻¹ | Yes |
| Sapphire | Uniaxial | 210-1000 cm⁻¹ | Yes |
| Gallium Oxide | Monoclinic | 350-800 cm⁻¹ | Yes |
| α-MoO₃ | Biaxial | 500-1050 cm⁻¹ | Yes |
| AlN | Uniaxial | 500-1050 cm⁻¹ | Yes |
| SiC | Uniaxial (≈isotropic) | 500-1050 cm⁻¹ | Yes |

**Note**: A material is "hyperbolic" when the real parts of permittivity components have opposite signs.