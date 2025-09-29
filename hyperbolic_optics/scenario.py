"""
Scenario module - NumPy implementation
Used for construction of four scenarios:

1. Frequency vs. Incident Angle
2. Frequency vs. Azimuthal Rotation
3. Dispersion at a given frequency
4. Simple - Single incident angle, orientation, and frequency
"""

import math as m
from abc import ABC

import numpy as np


class ScenarioSetup(ABC):
    """
    Abstract class for a scenario setup
    """

    def __init__(self, data):
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle", None)
        self.azimuthal_angle = data.get("azimuthal_angle", None)
        self.frequency = data.get("frequency", None)
        self.create_scenario()

    def create_scenario(self):
        """
        Creates the scenario based on the type
        """
        if self.type == "Incident":
            self.create_incident_scenario()
        elif self.type == "Azimuthal":
            self.create_azimuthal_scenario()
        elif self.type == "Dispersion":
            self.create_dispersion_scenario()
        elif self.type == "Simple":
            self.create_simple_scenario()
        else:
            raise NotImplementedError(f"Scenario type {self.type} not implemented")

    def create_incident_scenario(self):
        """
        Creates the incident scenario
        """
        self.incident_angle = np.linspace(
            -m.pi / 2.0 + 1.0e-9, m.pi / 2.0 - 1.0e-9, 360, dtype=np.float64
        )

    def create_azimuthal_scenario(self):
        """
        Creates the azimuthal scenario
        """
        self.incident_angle = np.float64(m.radians(self.incident_angle))
        self.azimuthal_angle = np.linspace(
            0.0 + 1.0e-15, 2.0 * m.pi - 1.0e-15, 360, dtype=np.float64
        )

    def create_dispersion_scenario(self):
        """
        Creates the dispersion scenario
        """
        self.incident_angle = np.linspace(0.0 + 1.0e-8, m.pi / 2.0 - 1.0e-8, 180, dtype=np.float64)

        self.azimuthal_angle = np.linspace(1.0e-5, 2.0 * m.pi - 1.0e-5, 480, dtype=np.float64)

        self.frequency = float(self.frequency)

    def create_simple_scenario(self):
        """
        Creates the simple scenario - single values for all parameters
        """
        # Convert to scalar values for consistency
        self.incident_angle = np.float64(m.radians(self.incident_angle) + 1.0e-15)
        self.azimuthal_angle = np.float64(m.radians(self.azimuthal_angle) + 1.0e-15)
        self.frequency = float(self.frequency)
