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
        self.incident_angle = data.get("incidentAngle", None)
        self.azimuthal_angle = data.get("azimuthal_angle", None)
        self.frequency = data.get("frequency", None)
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
        self.incident_angle = tf.linspace(
            tf.constant(-m.pi/2., dtype=tf.float64),
            tf.constant(m.pi/2., dtype=tf.float64),
            360)


    def create_azimuthal_scenario(self):
        """
        Creates the azimuthal scenario
        """
        self.incident_angle = tf.cast(m.radians((self.incident_angle)), dtype = tf.float64)
        self.azimuthal_angle = tf.linspace(
            tf.constant(0. + 1.e-15, dtype=tf.float64),
            tf.constant(2. * m.pi, dtype=tf.float64),
            360)


    def create_dispersion_scenario(self):
        """
        Creates the dispersion scenario
        """
        self.incident_angle = tf.linspace(
            tf.constant(0. + 1.e-5, dtype=tf.float64),
            tf.constant(m.pi/2., dtype=tf.float64),
            180)

        self.azimuthal_angle = tf.linspace(
            tf.constant(1.e-5, dtype=tf.float64),
            tf.constant(2. * m.pi, dtype=tf.float64),
            480)

        self.frequency = float(self.frequency)
