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


from abc import ABC, abstractmethod

class Scenario(ABC):
    def __init__(self, data):
        self.x_rotation = m.radians(float(data.get("rotationX")))
        self.y_rotation = m.radians(float(data.get("rotationY")))
        self.air_gap_thickness = float(data.get("airGapThickness")) * 1.e-4
        self.eps_prism = float(data.get("dielectricConstant"))
        self.material = self.create_material(data.get("material"))

    def create_material(self, material_name):
        material_classes = {"Quartz": Quartz, "Sapphire": Sapphire, "Calcite": CalciteUpper}
        return material_classes[material_name]()

    def prepare_data(self, reflectivity_values):
        data = {
            "reflectivity": reflectivity_values,
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
            data["frequency"] = self.requested_frequency
        return data

    @abstractmethod
    def execute(self):
        pass

class IncidentScenario(Scenario):
    def __init__(self, data):
        super().__init__(data)
        self.z_rotation = m.radians(float(data.get("azimuthalAngle")))
        incident_min = m.radians(float(data.get("incidentAngle").get("min")))
        incident_max = m.radians(float(data.get("incidentAngle").get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )

    def execute(self):
        reflectivity_values = IncidentSituation(self.__dict__)
        return self.prepare_data(reflectivity_values)


class AzimuthalScenario(Scenario):
    def __init__(self, data):
        super().__init__(data)
        self.incident_angle = m.radians(float(data.get("incidentAngle")))
        z_min = m.radians(float(data.get("azimuthalAngle").get("min")))
        z_max = m.radians(float(data.get("azimuthalAngle").get("max")))
        self.z_rotation = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )
    
    def execute(self):
        reflectivity_values = AzimuthalSituation(self.__dict__)
        return self.prepare_data(reflectivity_values)


class DispersionScenario(Scenario):
    def __init__(self, data):
        super().__init__(data)
        incident_min = m.radians(float(data.get("incidentAngle").get("min")))
        incident_max = m.radians(float(data.get("incidentAngle").get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )
        
        z_min = m.radians(float(data.get("azimuthalAngle").get("min")))
        z_max = m.radians(float(data.get("azimuthalAngle").get("max")))
        self.z_rotation = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )
        
        self.requested_frequency = float(data.get("frequency"))

    def execute(self):
        reflectivity_values = DispersionSituation(self.__dict__)
        return self.prepare_data(reflectivity_values)


def parse_json_initial(data):
    scenario_classes = {"Incident": IncidentScenario, 
                        "Azimuthal": AzimuthalScenario,
                        "Dispersion": DispersionScenario}
    return scenario_classes[data.get("scenario")](data)


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


    def prepare_common(self, rotation_func, prism_construct, airgap_mode):

        # Adjust frequency if available
        if self.requested_frequency:
            eps_tensor = self.material.fetch_permittivity_tensor_for_freq(self.requested_frequency)
            self.k_0 = self.requested_frequency * 2.0 * m.pi
        else:
            eps_tensor = self.material.fetch_permittivity_tensor()
            self.k_0 = self.material.frequency * 2.0 * m.pi

        self.eps_tensor = rotation_func(
            eps_tensor, self.x_rotation, self.y_rotation, self.z_rotation
        )

        self.k_x = tf.cast(tf.sqrt(self.eps_prism) * tf.sin(self.incident_angle), dtype=tf.complex64)

        # Construct the non-magnetic tensor.
        self.non_magnetic_tensor = Air(
            run_on_device_decorator=run_on_device
        ).construct_tensor_singular()

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


    def calculate_reflectivity(self, layers):
        transfer_matrix = functools.reduce(operator.matmul, layers)
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
        super().prepare_common(anisotropy_rotation_one_value, "tensor", 'airgap')
        
    def calculate(self):
        self.prepare()
        
        self.semi_infinite_layer = transfer_matrix_wrapper(
        self.k_x,
        self.eps_tensor,
        self.non_magnetic_tensor,
        semi_infinite=True,
        mode = "single_rotation"
        )

        layers = [self.prism_layer, self.air_layer, self.semi_infinite_layer]
        self.calculate_reflectivity(layers)

    def plot(self):
        contour_plot_simple_incidence(self.data_output)


class AzimuthalSituation(Situation):

    def prepare(self):
        super().prepare_common(anisotropy_rotation_one_axis, "singular", "simple_airgap")
        
    def calculate(self):
        self.prepare()
        
        self.semi_infinite_layer = transfer_matrix_wrapper(
        self.k_x,
        self.eps_tensor,
        self.non_magnetic_tensor,
        semi_infinite=True,
        mode = "simple_azimuthal"
        )

        self.prism_layer = self.prism_layer[tf.newaxis, tf.newaxis, ...]
        self.air_layer = self.air_layer[:, tf.newaxis, ...]

        layers = [self.prism_layer, self.air_layer, self.semi_infinite_layer]

        self.calculate_reflectivity(layers)

    def plot(self):
        contour_plot_simple_azimuthal(self.data_output)


class DispersionSituation(Situation):
    
        def prepare(self):
            super().prepare_common(anisotropy_rotation_one_value, "tensor", "simple_airgap")
            
        def calculate(self):
            self.prepare()
            
            self.semi_infinite_layer = transfer_matrix_wrapper(
            self.k_x,
            self.eps_tensor,
            self.non_magnetic_tensor,
            semi_infinite=True,
            mode = "simple_dispersion"
            )

            self.prism_layer = self.prism_layer[:, tf.newaxis, ...]
            self.air_layer = self.air_layer[:, tf.newaxis, ...]
    
            layers = [self.prism_layer, self.air_layer, self.semi_infinite_layer]
            self.calculate_reflectivity(layers)

        def plot(self):
            contour_plot_simple_dispersion(self.data_output)
    





def perform_calculation(payload):
    scenario = parse_json_initial(json.loads(payload))
    returned_data = scenario.execute()
    
    if isinstance(scenario, IncidentScenario):
        IncidentSituation(returned_data).execute()
    elif isinstance(scenario, AzimuthalScenario):
        AzimuthalSituation(returned_data).execute()
    elif isinstance(scenario, DispersionScenario):
        DispersionSituation(returned_data).execute()


def main():
    perform_calculation(mock_dispersion_payload())

    
main()

