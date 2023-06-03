import math as m
import tensorflow as tf
import json
import functools
import operator
from abc import abstractmethod

from anisotropy_utils import anisotropy_rotation_one_value, anisotropy_rotation_one_axis
from berreman import transfer_matrix_wrapper, reflection_coefficients
from device_config import run_on_device
from material_params import (Air, AmbientIncidentMedium, CalciteUpper, Quartz, Sapphire, Ambient_Exit_Medium)
from plots import contour_plot_simple_incidence, contour_plot_simple_azimuthal, contour_plot_simple_dispersion

from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload, mock_dispersion_payload_full
from scenarios import IncidentScenarioSetup, AzimuthalScenarioSetup, DispersionScenarioSetup



class Situation:
    def __init__(self, params):

        self.eps_prism = params.get("eps_prism")
        self.air_gap_thickness = params.get("air_gap_thickness")
        self.material = params.get("material")
        self.x_rotation = params.get("x_rotation")
        self.y_rotation = params.get("y_rotation")
        self.z_rotation = params.get("z_rotation")
        self.incident_angle = params.get("incident_angle")
        self.requested_frequency = params.get("frequency", None)

        self.k_x = tf.cast(tf.sqrt(self.eps_prism) * tf.sin(self.incident_angle), dtype=tf.complex64)

        # Adjust frequency if available
        if self.requested_frequency:
            self.eps_tensor = self.material.fetch_permittivity_tensor_for_freq(self.requested_frequency)
            self.k_0 = self.requested_frequency * 2.0 * m.pi
        else:
            self.eps_tensor = self.material.fetch_permittivity_tensor()
            self.k_0 = self.material.frequency * 2.0 * m.pi

        # Construct the non-magnetic tensor.
        self.non_magnetic_tensor = Air(
            run_on_device_decorator=run_on_device
        ).construct_tensor_singular()


    def prepare_common(self, rotation_func, prism_construct, airgap_mode):

        self.eps_tensor = rotation_func(
            self.eps_tensor, self.x_rotation, self.y_rotation, self.z_rotation
        )

        incident_prism = AmbientIncidentMedium(
            self.eps_prism, self.incident_angle
        )
        if prism_construct == "tensor":
            self.prism_layer = incident_prism.construct_tensor()
        else:
            self.prism_layer = incident_prism.construct_tensor_singular()

        # Construct the air layer.
        self.air_layer = (
                transfer_matrix_wrapper(
                    self.k_x,
                    self.non_magnetic_tensor,
                    self.non_magnetic_tensor,
                    self.k_0,
                    thickness=self.air_gap_thickness,
                    mode = airgap_mode
                )
            )
        

    def produce_layers(self, mode):

        self.semi_infinite_layer = transfer_matrix_wrapper(
        self.k_x,
        self.eps_tensor,
        self.non_magnetic_tensor,
        semi_infinite=True,
        mode = mode
        )

        if mode == 'simple_azimuthal':
            self.prism_layer = self.prism_layer[tf.newaxis, tf.newaxis, ...]
            self.air_layer = self.air_layer[:, tf.newaxis, ...]

        elif mode == 'simple_dispersion':
            self.prism_layer = self.prism_layer[:, tf.newaxis, ...]
            self.air_layer = self.air_layer[:, tf.newaxis, ...]

        self.layers = [self.prism_layer, self.air_layer, self.semi_infinite_layer]


    def calculate_reflectivity(self):
        transfer_matrix = functools.reduce(operator.matmul, self.layers)
        self.reflectivity = reflection_coefficients(transfer_matrix)
    

    def prepare_output(self):
        self.data_output = {
            "reflectivity": self.reflectivity,
            "material": self.material,
            "incident_angle": self.incident_angle,
            "x_rotation": self.x_rotation,
            "y_rotation": self.y_rotation,
            "z_rotation": self.z_rotation,
            "eps_prism": self.eps_prism,
            "air_gap_thickness": self.air_gap_thickness,
        }
        # include frequency if exists in params
        if hasattr(self, "requested_frequency"):
            self.data_output["frequency"] = self.requested_frequency
    
    def execute(self):
        self.calculate()
        self.prepare_output()
        self.plot()


class IncidentSituation(Situation):
    
    def prepare(self):
        self.prepare_common(anisotropy_rotation_one_value, "tensor", 'airgap')
        
    def calculate(self):
        self.prepare()
        self.produce_layers('single_rotation')
        self.calculate_reflectivity()

    def plot(self):
        contour_plot_simple_incidence(self.data_output)


class AzimuthalSituation(Situation):

    def prepare(self):
        super().prepare_common(anisotropy_rotation_one_axis, "singular", "simple_airgap")
        
    def calculate(self):
        self.prepare()
        self.produce_layers('simple_azimuthal')
        self.calculate_reflectivity()

    def plot(self):
        contour_plot_simple_azimuthal(self.data_output)


class DispersionSituation(Situation):
    
        def prepare(self):
            self.prepare_common(anisotropy_rotation_one_value, "tensor", "simple_airgap")
            
        def calculate(self):
            self.prepare()
            self.produce_layers('simple_dispersion')
            self.calculate_reflectivity()

        def plot(self):
            contour_plot_simple_dispersion(self.data_output)
    


def perform_calculation(payload):
    scenario = parse_json_initial(json.loads(payload))

    if isinstance(scenario, IncidentScenarioSetup):
        IncidentSituation(scenario.__dict__).execute()
    elif isinstance(scenario, AzimuthalScenarioSetup):
        AzimuthalSituation(scenario.__dict__).execute()
    elif isinstance(scenario, DispersionScenarioSetup):
        DispersionSituation(scenario.__dict__).execute()


def parse_json_initial(data):
    scenario_classes = {"Incident": IncidentScenarioSetup, 
                        "Azimuthal": AzimuthalScenarioSetup,
                        "Dispersion": DispersionScenarioSetup}
    return scenario_classes[data.get("scenario")](data)


def main():
    perform_calculation(mock_azimuthal_payload())

    
main()

