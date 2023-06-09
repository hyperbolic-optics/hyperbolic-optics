import math as m
import tensorflow as tf
import json 
import functools
import operator

from abc import abstractmethod, ABC
from payloads import mock_azimuthal_payload, mock_incident_payload, mock_dispersion_payload
from material_params import (Air, AmbientIncidentMedium, CalciteUpper, Quartz, Sapphire, CalciteLower, AmbientExitMedium)
from berreman import transfer_matrix_wrapper, reflection_coefficients
from anisotropy_utils import anisotropy_rotation_one_axis, anisotropy_rotation_one_value
from plots import contour_plot_simple_dispersion, contour_plot_simple_incidence, contour_plot_simple_azimuthal


class ScenarioSetup(ABC):
    def __init__(self, data):
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle")
        self.frequency = data.get("frequency", None)
        self.azimuthal_angle = data.get("azimuthalAngle", None)
        self.create_scenario()
        

    def create_scenario(self):
        if self.type == 'Incident':
            self.create_incident_scenario()
        elif self.type == 'Azimuthal':
            self.create_azimuthal_scenario()
        elif self.type == 'Dispersion':
            self.create_dispersion_scenario()
        else:
            raise NotImplementedError(f"Scenario type {self.type} not implemented")
    
    def create_incident_scenario(self):
        incident_min = m.radians(float(self.incident_angle.get("min")))
        incident_max = m.radians(float(self.incident_angle.get("max")))
        self.incident_angle = tf.linspace(
            tf.constant(incident_min, dtype=tf.float32),
            tf.constant(incident_max, dtype=tf.float32),
            abs(int(m.degrees(incident_max - incident_min)//2))
            )
    
    def create_azimuthal_scenario(self):
        self.incident_angle = m.radians(float(self.incident_angle))
        z_min = m.radians(float(self.azimuthal_angle.get("min")))
        z_max = m.radians(float(self.azimuthal_angle.get("max")))
        self.azimuthal_angle = tf.linspace(
            tf.constant(z_min, dtype=tf.float32),
            tf.constant(z_max, dtype=tf.float32),
            abs(int(m.degrees(z_max - z_min)//2))
            )
        
    def create_dispersion_scenario(self):
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



class Layer(ABC):
    def __init__(self, data, scenario, kx, k0):
        self.type = data.get('type')
        self.material = data.get('material', None)
        self.rotationX = m.radians(float(data.get('rotationX', 0)))
        self.rotationY = m.radians(float(data.get('rotationY', 0)))
        self.rotationZ = m.radians(float(data.get('rotationZ', 0)))
        self.rotationZ_type = data.get('rotationZType', 'relative')
        self.kx = kx
        self.k0 = k0
        self.frequency = scenario.frequency
        self.scenario = scenario.type
        self.incident_angle = scenario.incident_angle
        self.azimuthal_angle = scenario.azimuthal_angle

        self.non_magnetic_tensor = Air().construct_tensor_singular()

        self.thickness = data.get('thickness', None)
        if self.thickness:
            self.thickness*=1e-4

    def material_factory(self):
        if self.material == 'Quartz':
            self.material = Quartz()
        elif self.material == 'Sapphire':
            self.material = Sapphire()
        elif self.material == 'Calcite':
            self.material = CalciteUpper()
        elif self.material == "CalciteLower":
            self.material == CalciteLower()
        else:
            raise NotImplementedError(f"Material {self.material} not implemented")

    def calculate_z_rotation(self):
        if self.scenario == 'Dispersion' or self.scenario == "Azimuthal":
            if self.rotationZ_type == 'relative':
                self.rotationZ = self.azimuthal_angle + self.rotationZ
            elif self.rotationZ_type == 'static':
                self.rotationZ = self.rotationZ * tf.ones_like(self.azimuthal_angle)
        elif self.scenario == 'Incident':
            pass
    
    def calculate_eps_tensor(self):
        self.material_factory()
        if self.scenario == 'Incident' or self.scenario == 'Azimuthal':
            self.eps_tensor = self.material.fetch_permittivity_tensor()
        if self.scenario == 'Dispersion':
            self.eps_tensor = self.material.fetch_permittivity_tensor_for_freq(self.frequency)

    def rotate_tensor(self):
        if self.scenario == 'Incident' or self.scenario == 'Dispersion':
            rotation_func = anisotropy_rotation_one_value
        elif self.scenario == 'Azimuthal':
            rotation_func = anisotropy_rotation_one_axis
        
        self.eps_tensor = rotation_func(self.eps_tensor, self.rotationX, self.rotationY, self.rotationZ)

    @abstractmethod
    def create(self):
        pass


class PrismLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_prism = float(data.get('permittivity', None))
        self.create()
            
    def create(self):
        prism = AmbientIncidentMedium(self.eps_prism, self.kx)

        if self.scenario == 'Incident':
            self.matrix = prism.construct_tensor()
        elif self.scenario == 'Azimuthal':
            self.matrix = prism.construct_tensor_singular()[tf.newaxis, tf.newaxis, ...]
        elif self.scenario == 'Dispersion':
            self.matrix = prism.construct_tensor()[:, tf.newaxis, ...]


class AirGapLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.permittivity = data.get('permittivity', 1.0)
        self.non_magnetic_tensor = self.non_magnetic_tensor * self.permittivity
        self.calculate_mode()
        self.create()
        self.reshape_for_multiplication()
    
    def calculate_mode(self):
        if self.scenario == 'Incident':
            self.mode = 'airgap'
        elif self.scenario == 'Azimuthal':
            self.mode = 'simple_airgap'
        elif self.scenario == 'Dispersion':
            self.mode = 'simple_airgap'
    
    def create(self):
        self.matrix = transfer_matrix_wrapper(
            self.kx,
            self.non_magnetic_tensor,
            self.non_magnetic_tensor,
            self.mode,
            k0 = self.k0,
            thickness=self.thickness,
        )
    
    def reshape_for_multiplication(self):
        if self.scenario == 'Azimuthal' or self.scenario == 'Dispersion':
            self.matrix = self.matrix[:, tf.newaxis, ...]


class CrystalLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.calculate_eps_tensor()
        self.calculate_z_rotation()
        self.rotate_tensor()
        self.calculate_mode()
        self.create()

    def calculate_mode(self):
        if self.scenario == 'Incident':
            self.mode = 'incidence'
        elif self.scenario == 'Azimuthal':
            self.mode = 'azimuthal'
        elif self.scenario == 'Dispersion':
            self.mode = 'dispersion'

    def create(self):
        self.matrix = transfer_matrix_wrapper(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        self.mode,
        k0=self.k0,
        thickness=self.thickness,
        semi_infinite=False,
        )


class SemiInfiniteCrystalLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.calculate_z_rotation()
        self.calculate_eps_tensor()
        self.rotate_tensor()
        self.calculate_mode()
        self.create()

    def calculate_mode(self):
        if self.scenario == 'Incident':
            self.mode = 'incidence'
        elif self.scenario == 'Azimuthal':
            self.mode = 'azimuthal'
        elif self.scenario == 'Dispersion':
            self.mode = 'dispersion'

    def create(self):
        self.matrix = transfer_matrix_wrapper(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        self.mode,
        semi_infinite=True,
        )

class IsotropicSemiInfiniteLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_incident = (tf.cast(kx, dtype=tf.float32)/ tf.sin(self.incident_angle))**2.
        self.eps_exit = data.get('permittivity', None)

        if not self.eps_exit:
            raise ValueError("No exit permittivity provided for isotropic semi-infinite layer")
        
        self.create()

    def create(self):
        exit_medium = AmbientExitMedium(self.incident_angle, self.eps_incident, self.eps_exit)

        if self.scenario == 'Incident':
            self.matrix = exit_medium.construct_tensor()
        elif self.scenario == 'Azimuthal':
            self.matrix = exit_medium.construct_tensor()[tf.newaxis, tf.newaxis, ...]
        elif self.scenario == 'Dispersion':
            self.matrix = exit_medium.construct_tensor()[:, tf.newaxis, ...]


class LayerFactory:
    def __init__(self):
        self.layer_classes = {
            "Ambient Incident Layer": PrismLayer,
            "Isotropic Middle-Stack Layer": AirGapLayer,
            "Crystal Layer": CrystalLayer,
            "Semi Infinite Anisotropic Layer": SemiInfiniteCrystalLayer,
            "Semi Infinite Isotropic Layer" : IsotropicSemiInfiniteLayer,
        }

    def create_layer(self, layer_data, scenario, kx, k0):
        layer_class = self.layer_classes.get(layer_data['type'])
        if layer_class is not None:
            return layer_class(layer_data, scenario, kx, k0)
        else:
            raise ValueError(f"Invalid layer type {layer_data['type']}")



class Structure:

    def __init__(self):
        self.scenario = None
        self.factory = LayerFactory()
        self.layers = []


    def get_scenario(self, scenario_data):
        self.scenario = ScenarioSetup(scenario_data)
        self.setup_attributes()


    def setup_attributes(self):
        self.incident_angle = self.scenario.incident_angle
        self.azimuthal_angle = self.scenario.azimuthal_angle
        self.frequency = self.scenario.frequency


    def get_frequency_range(self, last_layer):

        if last_layer["material"] == 'Quartz':
            self.frequency = Quartz().frequency
        elif last_layer["material"] == 'Sapphire':
            self.frequency = Sapphire().frequency
        elif last_layer["material"] == 'Calcite':
            self.frequency = CalciteUpper().frequency
        else:
            raise NotImplementedError(f"Frequency for {self.material} not implemented")


    def calculate_kx_k0(self):
        self.k_x = tf.cast(tf.sqrt(float(self.eps_prism)) * tf.sin(self.incident_angle), dtype=tf.complex64)
        self.k_0 = self.frequency * 2.0 * m.pi


    def get_layers(self, layer_data_list):
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


    def calculate(self):
        transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, transfer_matrices)
    
    def calculate_reflectivity(self):
        bottom_line = self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        self.r_pp = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 3, 2] - self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 0, 2]) / bottom_line
        self.r_ps = (self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 1, 2] - (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 0, 2])) / bottom_line
        self.r_sp = (self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 3, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line
        self.r_ss = (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 2, 2] - self.transfer_matrix[..., 1, 2] * self.transfer_matrix[..., 2, 0]) / bottom_line

    def execute(self, payload):
        self.get_scenario(payload.get("ScenarioData"))
        self.get_layers(payload.get("Layers", None))
        self.calculate()
        self.calculate_reflectivity()