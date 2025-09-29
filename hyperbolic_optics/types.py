"""Type aliases for hyperbolic_optics package.

This module defines common type aliases used throughout the package
for improved type hinting and documentation. These types help users
and developers understand expected data structures.

Note:
    Python 3.12+ syntax using the `type` statement is available, but
    we use TypeAlias for backward compatibility and explicit exports.
"""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

# ============================================================================
# Array Types
# ============================================================================

ComplexArray: TypeAlias = npt.NDArray[np.complex128]
"""Complex-valued NumPy array with 128-bit precision.

Used for electromagnetic field components, reflection coefficients,
permittivity tensors, and other complex-valued quantities.
"""

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""Real-valued NumPy array with 64-bit precision.

Used for frequencies, angles, real-valued parameters, and Mueller matrices.
"""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Boolean NumPy array.

Used for masks and conditional selections.
"""

# ============================================================================
# Scalar Types
# ============================================================================

ComplexScalar: TypeAlias = complex | np.complex128
"""Complex scalar value (Python complex or NumPy complex128)."""

FloatScalar: TypeAlias = float | np.float64
"""Real scalar value (Python float or NumPy float64)."""

# ============================================================================
# Physical Tensor Types
# ============================================================================

PermittivityTensor: TypeAlias = ComplexArray
"""Permittivity tensor (ε).

Shape:
    - [3, 3] for single frequency/angle
    - [N, 3, 3] for frequency-dependent materials
    - [N, M, 3, 3] for scenario-dependent calculations

Note:
    The tensor is symmetric (ε_ij = ε_ji) for reciprocal media.
"""

PermeabilityTensor: TypeAlias = ComplexArray
"""Magnetic permeability tensor (μ).

Shape:
    - [3, 3] for single frequency/angle
    - [N, 3, 3] for frequency-dependent materials
    - [N, M, 3, 3] for scenario-dependent calculations

Note:
    Usually diagonal with μ = I for non-magnetic materials.
"""

TransferMatrix: TypeAlias = ComplexArray
"""4×4 transfer matrix relating field components at boundaries.

Shape:
    - [4, 4] for Simple scenario
    - [N, 4, 4] for single-axis sweep (Incident at one frequency)
    - [N, M, 4, 4] for two-axis sweep (Incident, Azimuthal, Dispersion)

Note:
    Relates [Ex, Ey, Hx, Hy] at one boundary to fields at another.
"""

MuellerMatrix: TypeAlias = FloatArray
"""4×4 Mueller matrix describing polarization transformations.

Shape:
    - [4, 4] for Simple scenario
    - [N, M, 4, 4] for angle/frequency sweeps

Note:
    Real-valued matrix relating input and output Stokes parameters.
    M_ij represents the coupling from input component j to output i.
"""

StokesVector: TypeAlias = FloatArray
"""Stokes parameters [S0, S1, S2, S3] describing polarization state.

Shape:
    - [4] for Simple scenario
    - [N, M, 4] for angle/frequency sweeps

Components:
    - S0: Total intensity
    - S1: Horizontal vs vertical linear polarization
    - S2: +45° vs -45° linear polarization
    - S3: Right vs left circular polarization
"""

# ============================================================================
# Configuration Types
# ============================================================================

LayerConfig: TypeAlias = dict[str, Any]
"""Configuration dictionary for a single layer.

Keys depend on layer type, but commonly include:
    - type (str): Layer type identifier
    - material (str | dict): Material name or custom parameters
    - thickness (float): Thickness in mm (for finite layers)
    - permittivity (float | dict): For isotropic layers
    - rotationX, rotationY, rotationZ (float): Euler angles in degrees

Example:
    >>> layer: LayerConfig = {
    ...     "type": "Semi Infinite Anisotropic Layer",
    ...     "material": "Calcite",
    ...     "rotationX": 0,
    ...     "rotationY": 90,
    ...     "rotationZ": 0
    ... }
"""

ScenarioConfig: TypeAlias = dict[str, str | float]
"""Configuration dictionary for simulation scenario.

Keys depend on scenario type:
    - type (str): 'Simple', 'Incident', 'Azimuthal', or 'Dispersion'

For 'Simple':
    - incidentAngle (float): Incident angle in degrees
    - azimuthal_angle (float): Azimuthal angle in degrees
    - frequency (float): Frequency in cm⁻¹

For 'Azimuthal':
    - incidentAngle (float): Fixed incident angle in degrees

For 'Dispersion':
    - frequency (float): Fixed frequency in cm⁻¹

For 'Incident':
    - No additional parameters needed

Example:
    >>> scenario: ScenarioConfig = {
    ...     "type": "Simple",
    ...     "incidentAngle": 45.0,
    ...     "azimuthal_angle": 0.0,
    ...     "frequency": 1460.0
    ... }
"""

PayloadConfig: TypeAlias = dict[str, ScenarioConfig | list[LayerConfig]]
"""Complete simulation configuration payload.

Keys:
    - ScenarioData (ScenarioConfig): Scenario configuration
    - Layers (list[LayerConfig]): List of layer configurations

Example:
    >>> payload: PayloadConfig = {
    ...     "ScenarioData": {
    ...         "type": "Simple",
    ...         "incidentAngle": 45.0,
    ...         "azimuthal_angle": 0.0,
    ...         "frequency": 1460.0
    ...     },
    ...     "Layers": [
    ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
    ...         {
    ...             "type": "Semi Infinite Anisotropic Layer",
    ...             "material": "Calcite",
    ...             "rotationY": 90
    ...         }
    ...     ]
    ... }
"""

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Array types
    "ComplexArray",
    "FloatArray",
    "BoolArray",
    # Scalar types
    "ComplexScalar",
    "FloatScalar",
    # Physical tensor types
    "PermittivityTensor",
    "PermeabilityTensor",
    "TransferMatrix",
    "MuellerMatrix",
    "StokesVector",
    # Configuration types
    "LayerConfig",
    "ScenarioConfig",
    "PayloadConfig",
]
