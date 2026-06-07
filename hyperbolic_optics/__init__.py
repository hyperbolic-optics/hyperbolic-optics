"""Hyperbolic Optics Simulation Package.

A comprehensive Python package for simulating the reflective properties of
hyperbolic materials and anisotropic structures using the 4×4 transfer
matrix method.

The package enables:
- Calculation of reflection coefficients for multilayer anisotropic structures
- Mueller matrix analysis for polarization transformations
- Support for various crystal materials (Quartz, Calcite, Sapphire, etc.)
- Multiple simulation scenarios (angle sweeps, dispersion, etc.)

Example:
    >>> from hyperbolic_optics import Structure
    >>> payload = {
    ...     "ScenarioData": {"type": "Simple", "incidentAngle": 45.0,
    ...                      "azimuthal_angle": 0.0, "frequency": 1460.0},
    ...     "Layers": [
    ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
    ...         {"type": "Semi Infinite Anisotropic Layer",
    ...          "material": "Calcite", "rotationY": 90}
    ...     ]
    ... }
    >>> structure = Structure()
    >>> structure.execute(payload)
    >>> reflectivity = abs(structure.r_pp)**2

References:
    - Passler & Paarmann, JOSA B 34, 2128-2139 (2017)
    - Yeh, JOSA 69, 742-756 (1979)
    - Berreman, JOSA 62, 502-510 (1972)
"""

__version__ = "0.3.0"

# Field profiles, transmission & absorption
from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.jones import Jones, compose_jones

# Layers
from hyperbolic_optics.layers import LayerFactory

# Materials
from hyperbolic_optics.materials import (
    Air,
    AluminiumNitride,
    ArbitraryMaterial,
    CalciteLower,
    CalciteUpper,
    GalliumNitride,
    GalliumOxide,
    HexagonalBoronNitride,
    MolybdenumTrioxide,
    Quartz,
    Sapphire,
    SiliconCarbide,
    list_materials,
)
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.scenario import ScenarioSetup

# Core classes
from hyperbolic_optics.structure import Structure

# Thickness sweep helper
from hyperbolic_optics.sweep import ThicknessSweep

# Type hints (for users who want them)
from hyperbolic_optics.types import (
    ComplexArray,
    FloatArray,
    LayerConfig,
    MuellerMatrix,
    PayloadConfig,
    PermeabilityTensor,
    PermittivityTensor,
    ScenarioConfig,
    StokesVector,
    TransferMatrix,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Structure",
    "Mueller",
    "Jones",
    "compose_jones",
    "FieldProfile",
    "ThicknessSweep",
    "ScenarioSetup",
    # Materials
    "Quartz",
    "Sapphire",
    "CalciteUpper",
    "CalciteLower",
    "GalliumOxide",
    "MolybdenumTrioxide",
    "AluminiumNitride",
    "SiliconCarbide",
    "HexagonalBoronNitride",
    "GalliumNitride",
    "list_materials",
    "ArbitraryMaterial",
    "Air",
    # Layers
    "LayerFactory",
    # Types
    "ComplexArray",
    "FloatArray",
    "PermittivityTensor",
    "PermeabilityTensor",
    "TransferMatrix",
    "MuellerMatrix",
    "StokesVector",
    "LayerConfig",
    "ScenarioConfig",
    "PayloadConfig",
]
