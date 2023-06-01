import math as m
import tensorflow as tf
import json
import functools
import operator
from abc import abstractmethod

from anisotropy_utils import anisotropy_rotation_one_value, anisotropy_rotation_one_axis
from berreman import transfer_matrix_wrapper, reflection_coefficients
from device_config import run_on_device
from material_params import (Air, Ambient_Incident_Prism, CalciteUpper, Quartz, Sapphire, Ambient_Exit_Medium)
from plots import contour_plot_simple_incidence, contour_plot_simple_azimuthal, contour_plot_simple_dispersion



class Scenario:
    def __init__(self, data):
        self.x_rotation = m.radians(float(data.get("rotationX")))
        self.y_rotation = m.radians(float(data.get("rotationY")))
        self.air_gap_thickness = float(data.get("airGapThickness")) * 1.e-4
        self.eps_prism = float(data.get("dielectricConstant"))
        self.material = self.create_material(data.get("material"))

    def create_material(self, material_name):
        material_classes = {"Quartz": Quartz, "Sapphire": Sapphire, "Calcite": CalciteUpper}
        return material_classes[material_name]()

    @abstractmethod
    def preprocess_data(self):
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
        return incident_situation(self.__dict__)


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
        return azimuthal_situation(self.__dict__)


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
        return dispersion_situation(self.__dict__)


def parse_json_initial(data):
    scenario_classes = {"Incident": IncidentScenario, 
                        "Azimuthal": AzimuthalScenario,
                        "Dispersion": DispersionScenario}
    return scenario_classes[data.get("scenario")](data)


def prepare_data(reflectivity_values, params):
    data = {
        "reflectivity": reflectivity_values,
        "material": params.get("material"),
        "incident_angle": params.get("incident_angle"),
        "x_rotation": params.get("x_rotation"),
        "y_rotation": params.get("y_rotation"),
        "z_rotation": params.get("z_rotation"),
        "eps_prism": params.get("eps_prism"),
        "air_gap_thickness": params.get("air_gap_thickness"),
    }
    # include frequency if exists in params
    if "requested_frequency" in params:
        data["frequency"] = params.get("requested_frequency")
    return data



def prepare_situation_common(params, rotation_func = anisotropy_rotation_one_value, prism_construct = "tensor", airgap_mode = 'simple_airgap'):
    eps_prism = params.get("eps_prism")
    air_gap_thickness = params.get("air_gap_thickness")
    material = params.get("material")
    x_rotation = params.get("x_rotation")
    y_rotation = params.get("y_rotation")
    z_rotation = params.get("z_rotation")
    incident_angle = params.get("incident_angle")
    requested_frequency = params.get("requested_frequency", None)

    # Adjust frequency if available
    if requested_frequency:
        eps_tensor = material.fetch_permittivity_tensor_for_freq(requested_frequency)
        k_0 = requested_frequency * 2.0 * m.pi
    else:
        eps_tensor = material.fetch_permittivity_tensor()
        k_0 = material.frequency * 2.0 * m.pi

    eps_tensor = rotation_func(
        eps_tensor, x_rotation, y_rotation, z_rotation
    )

    k_x = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype=tf.complex64)

    # Construct the non-magnetic tensor.
    non_magnetic_tensor = Air(
        run_on_device_decorator=run_on_device
    ).construct_tensor_singular()

    incident_prism = Ambient_Incident_Prism(
        eps_prism, incident_angle, run_on_device_decorator=run_on_device
    )
    if prism_construct == "tensor":
        prism_layer = incident_prism.construct_tensor()
    else:
        prism_layer = incident_prism.construct_tensor_singular()

    # Construct the air layer.
    air_layer = (
            transfer_matrix_wrapper(
                k_x,
                non_magnetic_tensor,
                non_magnetic_tensor,
                k_0,
                thickness=air_gap_thickness,
                mode = airgap_mode
            )
        )

    return prism_layer, air_layer, k_x, eps_tensor, non_magnetic_tensor, k_0, incident_prism, requested_frequency


def incident_situation(params):
    prism_layer, air_layer, k_x, eps_tensor, non_magnetic_tensor, k_0, incident_prism, _ = prepare_situation_common(params, airgap_mode = 'airgap')

    ambient_exit_layer = Ambient_Exit_Medium(
        incident_prism, 1.,
    ).construct_tensor()

    thin_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        k_0,
        thickness=0.5e-5,
        mode = "single_rotation"
    )

    #semi_infinite end layer
    semi_infinite_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        semi_infinite=True,
        mode = "single_rotation"
    )

    ### Reshaping
    prism_layer = prism_layer[tf.newaxis, ...]
    ambient_exit_layer = ambient_exit_layer[tf.newaxis, ...] 

    # Defining our layers in order that we want
    layers = [prism_layer, air_layer, semi_infinite_layer]

    # Apply matrix multiplication successively using functools.reduce and operator.matmul
    transfer_matrix = functools.reduce(operator.matmul, layers)

    ### Reflection Coefficient
    reflectivity_values = reflection_coefficients(transfer_matrix)

    return prepare_data(reflectivity_values, params)


def azimuthal_situation(params):
    prism_layer, air_layer, k_x, eps_tensor, non_magnetic_tensor, k_0, incident_prism, _ = prepare_situation_common(params,prism_construct = "singular", rotation_func = anisotropy_rotation_one_axis)
    

    thin_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        k_0,
        thickness=0.1e-4,
        mode = "simple_azimuthal"
    )

    #semi_infinite end layer
    semi_infinite_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        semi_infinite=True,
        mode = "simple_azimuthal"
    )


    ### Reshaping
    prism_layer = prism_layer[tf.newaxis, tf.newaxis, ...]
    air_layer = air_layer[:, tf.newaxis, ...]

    print(air_layer.shape)


    layers = [prism_layer, air_layer, semi_infinite_layer]

    # Apply matrix multiplication successively using functools.reduce and operator.matmul
    transfer_matrix = functools.reduce(operator.matmul, layers)

    ### Reflection Coefficient
    reflectivity_values = reflection_coefficients(transfer_matrix)
    

    return prepare_data(reflectivity_values, params)


def dispersion_situation(params):

    prism_layer, air_layer, k_x, eps_tensor, non_magnetic_tensor, k_0, incident_prism, requested_frequency = prepare_situation_common(params)
    
    thin_layer_1 = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        k_0,
        thickness=1.e-4,
        mode = "simple_dispersion"
    )

    ambient_exit_layer = Ambient_Exit_Medium(
        incident_prism, 1.
    ).construct_tensor()

    #semi_infinite end layer
    semi_infinite_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor,
        semi_infinite=True,
        mode = "simple_dispersion"
    )

    ### Reshaping
    prism_layer = prism_layer[:, tf.newaxis, ...]
    air_layer = air_layer[:, tf.newaxis, ...]
    ambient_exit_layer = ambient_exit_layer[:, tf.newaxis, ...]

    layers = [prism_layer, air_layer, semi_infinite_layer]

    # Apply matrix multiplication successively using functools.reduce and operator.matmul
    transfer_matrix = functools.reduce(operator.matmul, layers)

    ### Reflection Coefficient
    reflectivity_values = reflection_coefficients(transfer_matrix)

    return prepare_data(reflectivity_values, params)



def perform_calculation(payload):
    scenario = parse_json_initial(payload)
    returned_data = scenario.execute()
    # reflectivity, material, angle = scenario.execute()
    if isinstance(scenario, IncidentScenario):
        contour_plot_simple_incidence(returned_data)
    elif isinstance(scenario, AzimuthalScenario):
        contour_plot_simple_azimuthal(returned_data)
    elif isinstance(scenario, DispersionScenario):
        contour_plot_simple_dispersion(returned_data)


def mock_incident_payload():
    payload = json.dumps({
        "scenario":"Incident",
        "rotationX":45,
        "rotationY":45,
        "airGapThickness":"1.5",
        "dielectricConstant":5.5,
        "material":"Quartz",
        "incidentAngle":{"min":-90,"max":90},
        "azimuthalAngle":45
        })
    
    return payload


def mock_azimuthal_payload():
    payload = json.dumps({
        "scenario":"Azimuthal",
        "rotationX":"0",
        "rotationY": 45,
        "airGapThickness":1.5,
        "dielectricConstant": 5.5,
        "material":"Quartz",
        "azimuthalAngle":{"min":0,"max":360},
        "incidentAngle":"45"
        })
    
    return payload

def mock_dispersion_payload():
    payload = json.dumps({
        "scenario":"Dispersion",
        "rotationX": 0,
        "rotationY": 90,
        "airGapThickness": 0.,
        "dielectricConstant": 5.5,
        "material":"Quartz",
        "azimuthalAngle": {"min":0, "max":360},
        "incidentAngle": {"min":0,"max":90},
        "frequency": 468
        })
    
    return payload

def main():
    payload = json.loads(mock_incident_payload())
    perform_calculation(payload)

    
main()

