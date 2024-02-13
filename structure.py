"""
Structure class for the optical system
"""

import math as m
import functools
import operator
import tensorflow as tf

from material_params import (CalciteUpper, Quartz, Sapphire, CalciteLower)
from plots import (contour_plot_simple_incidence, contour_plot_simple_azimuthal,
                   contour_plot_simple_dispersion)

from layers import LayerFactory
from scenario import ScenarioSetup

class Structure:
    """
    Class for the structure of the optical system
    """

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
        """
        Gets the scenario from the scenario_data
        """
        self.scenario = ScenarioSetup(scenario_data)
        self.setup_attributes()


    def setup_attributes(self):
        """
        Sets up the attributes for the structure depending on the scenario
        """
        self.incident_angle = self.scenario.incident_angle
        self.azimuthal_angle = self.scenario.azimuthal_angle
        self.frequency = self.scenario.frequency


    def get_frequency_range(self, last_layer):
        """
        Gets the frequency range for the structure depending on the material of the last layer
        """
        material = last_layer["material"]

        if material == 'Quartz':
            self.frequency = Quartz().frequency
        elif material == 'Sapphire':
            self.frequency = Sapphire().frequency
        elif material == 'Calcite':
            self.frequency = CalciteUpper().frequency
        elif material =='CalciteLower':
            self.frequency = CalciteLower().frequency
        else:
            raise NotImplementedError("Material not implemented")


    def calculate_kx_k0(self):
        """
        Calculates the k_x and k_0 values for the structure
        """
        self.k_x = tf.sqrt(self.eps_prism) * tf.sin(self.incident_angle)
        self.k_0 = self.frequency * 2.0 * m.pi


    def get_layers(self, layer_data_list):
        """
        Creates the layers from the layer_data_list
        """
        ## First Layer is prism, so we parse it
        self.eps_prism = tf.cast(layer_data_list[0].get('permittivity', None), dtype=tf.float64)
        if not self.frequency:
            last_layer = layer_data_list[-1]
            if last_layer.get('type') != 'Semi Infinite Isotropic Layer':
                self.get_frequency_range(last_layer)
            else:
                self.get_frequency_range(layer_data_list[-2])
        self.calculate_kx_k0()
        
        ## Create prism layer and add it to layers list
        self.layers.append(self.factory.create_layer(layer_data_list[0],
                                                     self.scenario,
                                                     self.k_x,
                                                     self.k_0))
        
        ## Create the rest of the layers and add them to layers list
        for layer_data in layer_data_list[1:]:
            self.layers.append(self.factory.create_layer(layer_data,
                                                         self.scenario,
                                                         self.k_x,
                                                         self.k_0))
    
    
    def analyse_profiles(self):
        """
        Creates the propagation matrix for the given layers.
        """
        profiles = [layer for layer in self.layers[1:]]
        print(profiles)
        exit()

    def calculate(self):
        """
        Calculates the transfer matrix for the given layers.
        """
        transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, transfer_matrices)
    
    def calculate_reflectivity(self):
        """
        Calculates the reflectivity for the given transfer matrix.
        """
        bottom_line = self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        self.r_pp = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 3, 2] - self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 0, 2]) / bottom_line
        self.r_ps = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 1, 2] - (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 0, 2])) / bottom_line
        self.r_sp = (self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 3, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line
        self.r_ss = (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 1, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line

    def calculate_transmission(self):
        bottom_line = self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]

        self.t_pp = self.transfer_matrix[..., 0,0] / bottom_line
        self.t_ps = - self.transfer_matrix[..., 0,2] / bottom_line
        self.t_sp = - self.transfer_matrix[..., 2,0] / bottom_line
        self.t_ss = self.transfer_matrix[..., 2,2] / bottom_line


    def execute(self, payload):
        """
        Executes the calculation of the reflectivity for the given scenario data and layers.

        Args:
            payload (dict): A dictionary containing the scenario data and layers.

        Returns:
            None
        """
        # Get the scenario data
        self.get_scenario(payload.get("ScenarioData"))

        # Get the layers
        self.get_layers(payload.get("Layers", None))

        ### TODO: Now handle multiplication of transfer matrices from wave profiles
        ### Rather than handled during the creation of the layers

        # # Calculate the transfer matrix
        # self.calculate()

        # # Calculate the reflectivity
        # self.calculate_reflectivity()
        # self.calculate_transmission()

    def plot(self):
        """
        Plots the reflectivity for the given scenario."""

        if self.scenario.type == "Incident":
            contour_plot_simple_incidence(self)
        elif self.scenario.type == "Azimuthal":
            contour_plot_simple_azimuthal(self)
        elif self.scenario.type == "Dispersion":
            contour_plot_simple_dispersion(self)