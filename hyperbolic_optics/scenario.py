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
import warnings
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
        self.data = data
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle", None)
        self.azimuthal_angle = data.get("azimuthal_angle", None)
        self.frequency = data.get("frequency", None)
        self.create_scenario()

    def _angle_axis(
        self,
        value: Any,
        name: str,
        default_min: float,
        default_max: float,
        default_points: int,
        points_key: str,
    ) -> np.ndarray:
        """Resolve one swept angular axis, in radians.

        Precedence mirrors :meth:`~hyperbolic_optics.structure.Structure.resolve_frequency`:
        an explicit specification wins, otherwise the scenario's default grid is
        used. Accepts

        * a sequence of angles in degrees -- used verbatim,
        * a ``{"min": .., "max": .., "points": ..}`` dict in degrees,
        * nothing, giving the default grid at ``points_key`` resolution.

        A bare scalar cannot define a swept axis. It used to be accepted and then
        silently discarded, so a payload that looked like it fixed the angle
        actually swept the full default range; that now warns rather than
        quietly returning something else's answer.
        """
        points = int(self.data.get(points_key, default_points))
        if points < 2:
            raise ValueError(f"{points_key} must be at least 2, got {points}")

        if value is None:
            return np.linspace(default_min, default_max, points, dtype=np.float64)

        if isinstance(value, dict):
            low = m.radians(float(value["min"]))
            high = m.radians(float(value["max"]))
            return np.linspace(low, high, int(value.get("points", points)), dtype=np.float64)

        if np.ndim(value) > 0:
            return np.radians(np.asarray(value, dtype=np.float64))

        warnings.warn(
            f"{self.type} sweeps {name}, so the scalar {name}={value} cannot be "
            f"honoured and the default grid is used instead. Pass a list of "
            f"angles or {{'min': .., 'max': .., 'points': ..}} to control the "
            f"axis, set '{points_key}' to change only its resolution, or use the "
            f"'Simple' scenario for a single point.",
            UserWarning,
            stacklevel=4,
        )
        return np.linspace(default_min, default_max, points, dtype=np.float64)

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

        self.incident_angle = self._angle_axis(
            self.incident_angle,
            "incidentAngle",
            -m.pi / 2.0 + 1.0e-9,
            m.pi / 2.0 - 1.0e-9,
            360,
            "polar_points",
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
        self.azimuthal_angle = self._angle_axis(
            self.azimuthal_angle,
            "azimuthal_angle",
            0.0 + 1.0e-15,
            2.0 * m.pi - 1.0e-15,
            360,
            "azimuthal_points",
        )

    def create_dispersion_scenario(self) -> None:
        """Create k-space dispersion scenario at fixed frequency.

        Sets up grid of incident angles (180 points) and azimuthal angles
        (480 points) for mapping isofrequency contours in kx-ky space.

        Note:
            Requires 'frequency' to be specified in input data.
            Generates data for kx vs ky momentum-space plots.
        """
        self.incident_angle = self._angle_axis(
            self.incident_angle, "incidentAngle", 1.0e-8, m.pi / 2.0 - 1.0e-8, 180, "polar_points"
        )
        self.azimuthal_angle = self._angle_axis(
            self.azimuthal_angle,
            "azimuthal_angle",
            1.0e-5,
            2.0 * m.pi - 1.0e-5,
            480,
            "azimuthal_points",
        )

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
        self.incident_angle = self._angle_axis(
            self.incident_angle, "incidentAngle", 1.0e-9, m.pi / 2.0 - 1.0e-9, 180, "polar_points"
        )
        self.azimuthal_angle = self._angle_axis(
            self.azimuthal_angle,
            "azimuthal_angle",
            1.0e-15,
            2.0 * m.pi - 1.0e-15,
            120,
            "azimuthal_points",
        )
