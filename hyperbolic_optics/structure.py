"""
Structure class for the optical system
"""

import math as m
import functools
import operator
import tensorflow as tf

from hyperbolic_optics.materials import CalciteUpper, Quartz, Sapphire, GalliumOxide

from hyperbolic_optics.layers import LayerFactory
from hyperbolic_optics.scenario import ScenarioSetup


class Structure:
    """Class for the structure of the optical system."""

    def __init__(self):
        self.scenario = None
        self.factory = LayerFactory()
        self.layers = []
        self.incident_angle = None
        self.azimuthal_angle = None
        self.frequency = None
        self.eps_prism = None
        self.k_x = None
        self.k_0 = None
        self.r_pp = None
        self.r_ss = None
        self.r_ps = None
        self.r_sp = None
        self.transfer_matrix = None

    def get_scenario(self, scenario_data):
        """Get the scenario from the scenario_data."""
        self.scenario = ScenarioSetup(scenario_data)
        self.setup_attributes()

    def setup_attributes(self):
        """Set up the attributes for the structure based on the scenario."""
        self.incident_angle = self.scenario.incident_angle
        self.azimuthal_angle = self.scenario.azimuthal_angle
        self.frequency = self.scenario.frequency

    def get_frequency_range(self, last_layer):
        """Get the frequency range based on the material of the last layer."""
        material = last_layer["material"]

        if material == "Quartz":
            self.frequency = Quartz().frequency
        elif material == "Sapphire":
            self.frequency = Sapphire().frequency
        elif material == "Calcite":
            self.frequency = CalciteUpper().frequency
        elif material == "GalliumOxide":
            self.frequency = GalliumOxide().frequency
        else:
            raise NotImplementedError("Material not implemented")

    def calculate_kx_k0(self):
        """Calculate the k_x and k_0 values for the structure."""
        self.k_x = tf.cast(
            tf.sqrt(tf.cast(self.eps_prism, dtype=tf.float64))
            * tf.sin(tf.cast(self.incident_angle, dtype=tf.float64)),
            dtype=tf.float64,
        )
        self.k_0 = self.frequency * 2.0 * m.pi

    def get_layers(self, layer_data_list):
        """Create the layers from the layer_data_list."""
        # First Layer is prism, so we parse it
        self.eps_prism = layer_data_list[0].get("permittivity", None)
        if not self.frequency:
            last_layer = layer_data_list[-1]
            if last_layer.get("type") != "Semi Infinite Isotropic Layer":
                self.get_frequency_range(last_layer)
            else:
                self.get_frequency_range(layer_data_list[-2])
        self.calculate_kx_k0()

        # Create prism layer and add it to layers list
        self.layers.append(
            self.factory.create_layer(
                layer_data_list[0],
                self.scenario,
                self.k_x,
                self.k_0,
            )
        )

        # Create the rest of the layers and add them to layers list
        for layer_data in layer_data_list[1:]:
            self.layers.append(
                self.factory.create_layer(
                    layer_data,
                    self.scenario,
                    self.k_x,
                    self.k_0,
                )
            )

    def calculate(self):
        """Calculate the transfer matrix for the given layers."""
        self.transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, self.transfer_matrices)

    def calculate_reflectivity(self):
        """Calculate the reflectivity for the given transfer matrix."""
        bottom_line = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        )
        self.r_pp = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 3, 2]
            - self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 0, 2]
        ) / bottom_line
        self.r_ps = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 1, 2]
            - (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 0, 2])
        ) / bottom_line
        self.r_sp = (
            self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 3, 2] * self.transfer_matrix[..., 2, 0]
        ) / bottom_line
        self.r_ss = (
            self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 1, 2] * self.transfer_matrix[..., 2, 0]
        ) / bottom_line

    def calculate_transmissivity(self):
        """Calculate the transmissivity for the given transfer matrix."""
        bottom_line = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        )
        self.t_pp = (self.transfer_matrix[..., 0, 0]) / bottom_line
        self.t_ps = (-self.transfer_matrix[..., 0, 2]) / bottom_line
        self.t_sp = (-self.transfer_matrix[..., 2, 0]) / bottom_line
        self.t_ss = (self.transfer_matrix[..., 2, 2]) / bottom_line


    def display_layer_info(self):
        """Display the information for each layer in the structure."""
        for layer in self.layers:
            print(layer)
            print(layer)

        

    def execute(self, payload):
        """
        Execute the calculation of reflectivity for the given scenario and layers.

        Args:
            payload (dict): A dictionary containing the scenario data and layers.
        """
        # Get the scenario data
        self.get_scenario(payload.get("ScenarioData"))

        # Get the layers
        self.get_layers(payload.get("Layers", None))

        # Calculate the transfer matrix
        self.calculate()

        # Calculate the reflectivity
        self.calculate_reflectivity()

    # def plot(self):
    #     """Plot the reflectivity for the given scenario."""
    #     if self.scenario.type == "Incident":
    #         contour_plot_simple_incidence(self)
    #     elif self.scenario.type == "Azimuthal":
    #         contour_plot_simple_azimuthal(self)
    #     elif self.scenario.type == "Dispersion":
    #         contour_plot_simple_dispersion(self)