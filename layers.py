"""
Layers

Used for constructing individual layers in the device
"""

from abc import abstractmethod, ABC
import math as m
import tensorflow as tf
from material_params import (Air, AmbientIncidentMedium, CalciteUpper,
                            Quartz, Sapphire, CalciteLower, AmbientExitMedium)
from waves import Wave
from anisotropy_utils import anisotropy_rotation_one_axis, anisotropy_rotation_one_value


class Layer(ABC):
    """
    Abstract class for a layer in the device
    """
    def __init__(self, data, scenario, kx, k0):
        self.type = data.get('type')
        self.material = data.get('material', None)
        self.rotationX = tf.cast(m.radians(data.get('rotationX', 0)), dtype=tf.float64)
        self.rotationY = tf.cast(m.radians(data.get('rotationY', 0)), dtype=tf.float64) + 1.e-8
        self.rotationZ = tf.cast(m.radians(data.get('rotationZ', 0)), dtype=tf.float64)
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
            self.thickness = float(self.thickness) * 1.e-4

    def material_factory(self):
        """
        Creates the material object based on the material name
        """
        if self.material == 'Quartz':
            self.material = Quartz()
        elif self.material == 'Sapphire':
            self.material = Sapphire()
        elif self.material == 'Calcite':
            self.material = CalciteUpper()
        elif self.material == "CalciteLower":
            self.material = CalciteLower()
        else:
            raise NotImplementedError(f"Material {self.material} not implemented")

    def calculate_z_rotation(self):
        """
        Calculates the rotation of the layer in the z direction
        If the scenario is dispersion or azimuthal, the rotation is relative to the azimuthal angle, but
        can be defined to be static while all other layers are rotated.
        If it's relative, the rotation is added to the azimuthal angle as it has been 'shifted'.
        """
        if self.scenario == 'Dispersion' or self.scenario == "Azimuthal":
            if self.rotationZ_type == 'relative':
                self.rotationZ = self.azimuthal_angle + self.rotationZ
            elif self.rotationZ_type == 'static':
                self.rotationZ = self.rotationZ * tf.ones_like(self.azimuthal_angle)
        elif self.scenario == 'Incident':
            pass
    
    def calculate_eps_tensor(self):
        """
        Calculates the permittivity tensor for the layer
        """
        self.material_factory()
        if self.scenario == 'Incident' or self.scenario == 'Azimuthal':
            self.eps_tensor = tf.cast(self.material.fetch_permittivity_tensor(),dtype = tf.complex128)
        if self.scenario == 'Dispersion':
            self.eps_tensor = tf.cast(self.material.fetch_permittivity_tensor_for_freq(self.frequency), dtype= tf.complex128)

    def rotate_tensor(self):
        """
        Rotates the permittivity tensor according to the rotation angles
        """
        if self.scenario == 'Incident' or self.scenario == 'Dispersion':
            rotation_func = anisotropy_rotation_one_value
        elif self.scenario == 'Azimuthal':
            rotation_func = anisotropy_rotation_one_axis
        
        self.eps_tensor = rotation_func(self.eps_tensor, self.rotationX, self.rotationY, self.rotationZ)

    @abstractmethod
    def create(self):
        pass


class PrismLayer(Layer):
    """
    The Incident coupling prism layer
    """

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_prism = tf.cast(data.get('permittivity', 5.5), dtype=tf.float64)
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
    """
    The airgap layer
    """

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.permittivity = data.get('permittivity', 1.0)
        self.non_magnetic_tensor = self.non_magnetic_tensor * self.permittivity
        self.calculate_mode()
        self.create()
    
    def calculate_mode(self):
        """
        Determines the mode of the airgap layer
        """
        if self.scenario == 'Incident':
            self.mode = 'airgap'
        elif self.scenario == 'Azimuthal':
            self.mode = 'azimuthal_airgap'
        elif self.scenario == 'Dispersion':
            self.mode = 'simple_airgap'
    
    def create(self):
        self.profile, self.matrix = Wave(
            self.kx,
            self.non_magnetic_tensor,
            self.non_magnetic_tensor,
            self.mode,
            k_0 = self.k0,
            thickness=self.thickness,
        ).execute()



class CrystalLayer(Layer):
    """
    Anisotropic Crystal of arbritary orientation and thickness
    """

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.calculate_eps_tensor()
        self.calculate_z_rotation()
        self.rotate_tensor()
        self.create()

    def create(self):
        self.profile, self.matrix = Wave(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        self.scenario,
        k_0=self.k0,
        thickness=self.thickness
        ).execute()


class SemiInfiniteCrystalLayer(Layer):
    """
    Anisotropic Semi-infinite crystal layer is a layer that is infinite in the z direction
    """

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.calculate_z_rotation()
        self.calculate_eps_tensor()
        self.rotate_tensor()
        self.create()

    def create(self):
        self.profile, self.matrix = Wave(
        self.kx,
        self.eps_tensor,
        self.non_magnetic_tensor,
        self.scenario,
        semi_infinite=True,
        ).execute()



class IsotropicSemiInfiniteLayer(Layer):
    """
    Isotropic semi-infinite layer with a given permittivity
    """

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_incident = (tf.cast(kx, dtype=tf.float64)/ tf.sin(self.incident_angle))**2.
        self.eps_exit = float(data.get('permittivity', None))

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
    """
    Factory class for creating layers
    """
    def __init__(self):
        self.layer_classes = {
            "Ambient Incident Layer": PrismLayer,
            "Isotropic Middle-Stack Layer": AirGapLayer,
            "Crystal Layer": CrystalLayer,
            "Semi Infinite Anisotropic Layer": SemiInfiniteCrystalLayer,
            "Semi Infinite Isotropic Layer" : IsotropicSemiInfiniteLayer,
        }

    def create_layer(self, layer_data, scenario, kx, k0):
        """
        Creates a layer from the layer data
        """
        layer_class = self.layer_classes.get(layer_data['type'])
        if layer_class is not None:
            return layer_class(layer_data, scenario, kx, k0)
        else:
            raise ValueError(f"Invalid layer type {layer_data['type']}")