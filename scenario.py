"""
Scenario module
Used for construction of three scenarios:

1. Frequency vs. Incident Angle
2. Frequency vs. Azimuthal Rotation
3. Dispersion at a given frequency
"""

from abc import ABC
import math as m
import tensorflow as tf


class ScenarioSetup(ABC):
    """
    Abstract class for a scenario setup
    """
    def __init__(self, data):
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle")
        self.frequency = data.get("frequency", None)
        self.azimuthal_angle = data.get("azimuthalAngle", None)
        self.create_scenario()


    def create_scenario(self):
        """
        Creates the scenario based on the type
        """
        if self.type == 'Incident':
            self.create_incident_scenario()
        elif self.type == 'Azimuthal':
            self.create_azimuthal_scenario()
        elif self.type == 'Dispersion':
            self.create_dispersion_scenario()
        else:
            raise NotImplementedError(f"Scenario type {self.type} not implemented")


    def create_incident_scenario(self):
        """
        Creates the incident scenario
        """
        incident_min = m.radians(float(self.incident_angle.get("min")))
        incident_max = m.radians(float(self.incident_angle.get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)))
            )


    def create_azimuthal_scenario(self):
        """
        Creates the azimuthal scenario
        """
        self.incident_angle = m.radians(float(self.incident_angle))
        z_min = m.radians(float(self.azimuthal_angle.get("min")))
        z_max = m.radians(float(self.azimuthal_angle.get("max")))
        self.azimuthal_angle = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )


    def create_dispersion_scenario(self):
        """
        Creates the dispersion scenario
        """
        incident_min = m.radians(float(self.incident_angle.get("min")))
        incident_max = m.radians(float(self.incident_angle.get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )

        z_min = m.radians(float(self.azimuthal_angle.get("min")))
        z_max = m.radians(float(self.azimuthal_angle.get("max")))
        self.azimuthal_angle = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )

        self.frequency = float(self.frequency)
