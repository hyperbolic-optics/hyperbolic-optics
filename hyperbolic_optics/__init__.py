"""Hyperbolic Optics Simulation Package.

A comprehensive Python package for simulating the reflective properties of
hyperbolic materials and anisotropic structures using the 4×4 transfer
matrix method.
"""

__version__ = "0.1.8"

# Core classes
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.scenario import ScenarioSetup

# Materials
from hyperbolic_optics.materials import (
    Quartz,
    Sapphire,
    CalciteUpper,
    CalciteLower,
    GalliumOxide,
    ArbitraryMaterial,
    Air,
)

# Layers
from hyperbolic_optics.layers import LayerFactory

# Type hints (for users who want them)
from hyperbolic_optics.types import (
    ComplexArray,
    FloatArray,
    PermittivityTensor,
    PermeabilityTensor,
    TransferMatrix,
    MuellerMatrix,
    StokesVector,
    LayerConfig,
    ScenarioConfig,
    PayloadConfig,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Structure",
    "Mueller",
    "ScenarioSetup",
    # Materials
    "Quartz",
    "Sapphire",
    "CalciteUpper",
    "CalciteLower",
    "GalliumOxide",
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