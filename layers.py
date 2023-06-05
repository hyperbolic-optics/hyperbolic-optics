import math as m
import tensorflow as tf
import json 
import functools
import operator

from abc import abstractmethod, ABC
from payloads import mock_dispersion_payload_full as mock_payload
from material_params import (Air, AmbientIncidentMedium, CalciteUpper, Quartz, Sapphire, Ambient_Exit_Medium)
from berreman import transfer_matrix_wrapper, reflection_coefficients
from anisotropy_utils import anisotropy_rotation_one_axis, anisotropy_rotation_one_value
from plots import contour_plot_simple_dispersion


class ScenarioSetup(ABC):
    def __init__(self, data):
        self.type = data.get("type")
        self.incident_angle = data.get("incidentAngle")
        self.frequency = data.get("frequency", None)
        self.azimuthal_angle = data.get("azimuthalAngle")
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
        self.azimuthal_angle = m.radians(float(self.azimuthal_angle))
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
        self.rotationZ = m.radians(float(data.get('rotationZShift', 0)))
        self.rotationZ_type = data.get('rotationZType', None)
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
        else:
            raise NotImplementedError(f"Material {self.material} not implemented")

    def calculate_z_rotation(self):
        if self.rotationZ_type == 'relative':
            self.rotationZ = self.azimuthal_angle + self.rotationZ
        elif self.rotationZ_type == 'static':
            self.rotationZ = self.rotationZ * tf.ones_like(self.azimuthal_angle)
    
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
        self.eps_prism = data.get('eps_prism', None)
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
            self.k0,
            thickness=self.thickness,
            mode = self.mode
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
            self.mode = 'single_rotation'
        elif self.scenario == 'Azimuthal':
            self.mode = 'simple_azimuthal'
        elif self.scenario == 'Dispersion':
            self.mode = 'simple_dispersion'

    def create(self):
        self.matrix = transfer_matrix_wrapper(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        k0=self.k0,
        thickness=self.thickness,
        semi_infinite=False,
        mode = self.mode
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
            self.mode = 'single_rotation'
        elif self.scenario == 'Azimuthal':
            self.mode = 'simple_azimuthal'
        elif self.scenario == 'Dispersion':
            self.mode = 'simple_dispersion'

    def create(self):
        self.matrix = transfer_matrix_wrapper(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        semi_infinite=True,
        mode = self.mode
        )

class IsotropicSemiInfiniteLayer(Layer):

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)

    def create(self):
        # Your implementation
        pass


class LayerFactory:
    def __init__(self):
        self.layer_classes = {
            "prism": PrismLayer,
            "air_gap": AirGapLayer,
            "crystal_layer": CrystalLayer,
            "semi_infinite_crystal": SemiInfiniteCrystalLayer,
            "isotropic_semi_infinite" : IsotropicSemiInfiniteLayer,
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


    def calculate_kx_k0(self):
        self.k_x = tf.cast(tf.sqrt(self.eps_prism) * tf.sin(self.incident_angle), dtype=tf.complex64)
        self.k_0 = self.frequency * 2.0 * m.pi


    def get_layers(self, layer_data_list):
        ## First Layer is prism, so we parse it
        self.eps_prism = layer_data_list[0].get('eps_prism', None)
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
    
    def plot_reflectivity(self):
        contour_plot_simple_dispersion(self.r_pp, self.r_ps, self.r_sp, self.r_ss, 
                                       self.incident_angle, self.azimuthal_angle)


def main():
    payload = json.loads(mock_payload())
    scenario_data = payload.get("ScenarioData", None)

    structure = Structure()
    structure.get_scenario(scenario_data)
    structure.get_layers(payload.get("Layers", None))
    structure.calculate()
    structure.calculate_reflectivity()
    structure.plot_reflectivity()
    

main()