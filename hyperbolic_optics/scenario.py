"""Simulation scenario configuration and setup.

This module defines four scenario types for different analysis needs:

1. Simple: Single-point calculation (scalar angles and frequency)
2. Incident: Angle sweep at multiple frequencies (kx-ω plots)
3. Azimuthal: Sample rotation at fixed incident angle (β-ω plots)
4. Dispersion: k-space map at fixed frequency (kx-ky plots)

Each scenario type automatically sets up appropriate angle and frequency
arrays with correct dimensions for batch processing.
"""

import math as m
from abc import ABC
from typing import Any

import numpy as np


class ScenarioSetup(ABC):
    """
    Abstract class for a scenario setup
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize scenario configuration for simulation.

        Args:
            data: Dictionary with 'type' and scenario-specific parameters

        Raises:
            NotImplementedError: If scenario type is not recognized

        Example:
            >>> data = {"type": "Simple", "incidentAngle": 45.0,
            ...         "azimuthal_angle": 0.0, "frequency": 1460.0}
            >>> scenario = ScenarioSetup(data)
        """
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle", None)
        self.azimuthal_angle = data.get("azimuthal_angle", None)
        self.frequency = data.get("frequency", None)
        self.create_scenario()

    def create_scenario(self) -> None:
        """Create scenario-specific angle and frequency arrays.

        Dispatches to appropriate scenario creation method based on type.

        Raises:
            NotImplementedError: If scenario type is not implemented
        """
        if self.type == "Incident":
            self.create_incident_scenario()
        elif self.type == "Azimuthal":
            self.create_azimuthal_scenario()
        elif self.type == "Dispersion":
            self.create_dispersion_scenario()
        elif self.type == "Simple":
            self.create_simple_scenario()
        elif self.type == "FullSweep":
            self.create_full_sweep_scenario()
        else:
            raise NotImplementedError(f"Scenario type {self.type} not implemented")

    def create_incident_scenario(self) -> None:
        """Create incident angle sweep scenario.

        Sets up 360 incident angles from -π/2 to π/2 for analyzing angle-dependent
        reflectivity across the material's frequency range.

        Note:
            Frequency range is determined by the material in the final layer.
            Creates arrays suitable for generating kx vs frequency plots.
        """

        self.incident_angle = np.linspace(
            -m.pi / 2.0 + 1.0e-9, m.pi / 2.0 - 1.0e-9, 360, dtype=np.float64
        )

    def create_azimuthal_scenario(self) -> None:
        """Create azimuthal rotation scenario at fixed incident angle.

        Rotates the sample through 360 azimuthal angles (0 to 2π) while
        maintaining constant incident angle.

        Note:
            Useful for studying in-plane anisotropy and rotational symmetry.
            incidentAngle must be provided in input data.
        """
        self.incident_angle = np.float64(m.radians(self.incident_angle))
        self.azimuthal_angle = np.linspace(
            0.0 + 1.0e-15, 2.0 * m.pi - 1.0e-15, 360, dtype=np.float64
        )

    def create_dispersion_scenario(self) -> None:
        """Create k-space dispersion scenario at fixed frequency.

        Sets up grid of incident angles (180 points) and azimuthal angles
        (480 points) for mapping isofrequency contours in kx-ky space.

        Note:
            Requires 'frequency' to be specified in input data.
            Generates data for kx vs ky momentum-space plots.
        """
        self.incident_angle = np.linspace(0.0 + 1.0e-8, m.pi / 2.0 - 1.0e-8, 180, dtype=np.float64)

        self.azimuthal_angle = np.linspace(1.0e-5, 2.0 * m.pi - 1.0e-5, 480, dtype=np.float64)

        self.frequency = float(self.frequency)

    def create_simple_scenario(self) -> None:
        """Create single-point scenario with scalar values.

        Converts incident angle, azimuthal angle, and frequency to scalar
        values for quick single-point calculations.

        Note:
            All three parameters (incidentAngle, azimuthal_angle, frequency)
            must be provided in input data.
        """
        # Convert to scalar values for consistency
        self.incident_angle = np.float64(m.radians(self.incident_angle) + 1.0e-15)
        self.azimuthal_angle = np.float64(m.radians(self.azimuthal_angle) + 1.0e-15)
        self.frequency = float(self.frequency)

    def create_full_sweep_scenario(self) -> None:
        """Create full 3D parameter sweep: frequency × incident_angle × azimuthal_angle.

        Sets up a 3D grid sweeping all three parameters simultaneously for complete
        visualization of the optical response space.

        Note:
            Frequency range is determined by the material in the final layer.
            Output will have shape [N_freq, N_incident, N_azimuthal]
        """
        # Incident angles - sweep from 0 to +90 degrees (90 points)
        self.incident_angle = np.linspace(0.0 + 1.0e-9, m.pi / 2.0 - 1.0e-9, 180, dtype=np.float64)

        # Azimuthal angles - full rotation 0 to 360 degrees (120 points)
        self.azimuthal_angle = np.linspace(
            0.0 + 1.0e-15, 2.0 * m.pi - 1.0e-15, 120, dtype=np.float64
        )
