"""
Structure class for the optical system
"""

import math as m
import functools
import operator
import tensorflow as tf

from material_params import (CalciteUpper, Quartz, Sapphire)
from plots import (contour_plot_simple_incidence, contour_plot_simple_azimuthal,
                   contour_plot_simple_dispersion)

from layers import LayerFactory
from scenario import ScenarioSetup

class Structure:
    """
    Class for the structure of the optical system
    """

    def __init__(self):
        print("Initializing Structure object")  # Add this line
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
        """
        Gets the scenario from the scenario_data
        """
        print("Getting scenario")  # Add this line
        print(f"Scenario data: {scenario_data}")  # Add this line
        self.scenario = ScenarioSetup(scenario_data)
        print("Created ScenarioSetup object")  # Add this line
        self.setup_attributes()
        print("Finished setting up attributes")  # Add this line

    def setup_attributes(self):
        """
        Sets up the attributes for the structure depending on the scenario
        """
        print("Setting up attributes")  # Add this line
        self.incident_angle = self.scenario.incident_angle
        self.azimuthal_angle = self.scenario.azimuthal_angle
        self.frequency = self.scenario.frequency

    def get_frequency_range(self, last_layer):
        """
        Gets the frequency range for the structure depending on the material of the last layer
        """
        print("Getting frequency range")  # Add this line
        material = last_layer["material"]

        if material == 'Quartz':
            self.frequency = Quartz().frequency
        elif material == 'Sapphire':
            self.frequency = Sapphire().frequency
        elif material == 'Calcite':
            self.frequency = CalciteUpper().frequency
        else:
            raise NotImplementedError("Material not implemented")

    def calculate_kx_k0(self):
        """
        Calculates the k_x and k_0 values for the structure
        """
        print("Calculating k_x and k_0")  # Add this line
        self.k_x = tf.cast(tf.sqrt(float(self.eps_prism)) * tf.sin(self.incident_angle), dtype=tf.complex64)
        self.k_0 = self.frequency * 2.0 * m.pi

    def get_layers(self, layer_data_list):
        """
        Creates the layers from the layer_data_list
        """
        print("Getting layers")  # Add this line
        ## First Layer is prism, so we parse it
        self.eps_prism = layer_data_list[0].get('permittivity', None)
        if not self.frequency:
            last_layer = layer_data_list[-1]
            if last_layer.get('type') != 'Semi Infinite Isotropic Layer':
                self.get_frequency_range(last_layer)
            else:
                self.get_frequency_range(layer_data_list[-2])
        self.calculate_kx_k0()
        
        ## Create prism layer and add it to layers list
        print("Creating prism layer")  # Add this line
        self.layers.append(self.factory.create_layer(layer_data_list[0],
                                                     self.scenario,
                                                     self.k_x,
                                                     self.k_0))
        
        ## Create the rest of the layers and add them to layers list
        print("Creating remaining layers")  # Add this line
        for layer_data in layer_data_list[1:]:
            self.layers.append(self.factory.create_layer(layer_data,
                                                         self.scenario,
                                                         self.k_x,
                                                         self.k_0))

    def calculate(self):
        """
        Calculates the transfer matrix for the given layers.
        """
        print("Calculating transfer matrix")  # Add this line
        self.transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, self.transfer_matrices)
    
    def calculate_reflectivity(self):
        """
        Calculates the reflectivity for the given transfer matrix.
        """
        print("Calculating reflectivity")  # Add this line
        bottom_line = self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        self.r_pp = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 3, 2] - self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 0, 2]) / bottom_line
        self.r_ps = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 1, 2] - (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 0, 2])) / bottom_line
        self.r_sp = (self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 3, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line
        self.r_ss = (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 1, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line

    def execute(self, payload):
        """
        Executes the calculation of the reflectivity for the given scenario data and layers.

        Args:
            payload (dict): A dictionary containing the scenario data and layers.

        Returns:
            None
        """
        try:
            print("Executing structure")  # Add this line
            
            # Get the scenario data
            print("Getting scenario data")  # Add this line
            self.get_scenario(payload.get("ScenarioData"))
            print("Got scenario data")  # Add this line
            
            # Get the layers
            print("Getting layers")  # Add this line
            self.get_layers(payload.get("Layers", None))
            print("Got layers")  # Add this line
            
            # Calculate the transfer matrix
            print("Calculating transfer matrix")  # Add this line
            self.calculate()
            print("Calculated transfer matrix")  # Add this line

            # Calculate the reflectivity
            print("Calculating reflectivity")  # Add this line
            self.calculate_reflectivity()
            print("Calculated reflectivity")  # Add this line

        except Exception as e:
            print(f"An exception occurred: {str(e)}")

    def plot(self):
        """
        Plots the reflectivity for the given scenario."""

        if self.scenario.type == "Incident":
            contour_plot_simple_incidence(self)
        elif self.scenario.type == "Azimuthal":
            contour_plot_simple_azimuthal(self)
        elif self.scenario.type == "Dispersion":
            contour_plot_simple_dispersion(self)