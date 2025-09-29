"""
Layers module for constructing individual layers in the device.
NumPy implementation - Stage 2 Refactor: Updated to properly handle both eps and mu tensors from materials
"""

import math as m
from abc import ABC, abstractmethod

import numpy as np

from hyperbolic_optics.anisotropy_utils import (anisotropy_rotation_one_axis,
                                                anisotropy_rotation_one_value)
from hyperbolic_optics.materials import (Air, ArbitraryMaterial, CalciteLower,
                                         CalciteUpper, GalliumOxide, Quartz,
                                         Sapphire)
from hyperbolic_optics.waves import Wave


class AmbientMedium:
    """Base class for ambient mediums (incident and exit)."""

    def __init__(self):
        """Initialize the ambient medium."""
        pass


class AmbientIncidentMedium(AmbientMedium):
    """
    Class representing the ambient incident medium.
    Moved from material_params.py to better organize layer-related functionality.
    """

    def __init__(self, permittivity, kx):
        """
        Initialize the AmbientIncidentMedium class.

        Args:
            permittivity (float): The permittivity of the ambient incident medium.
            kx (float): The x-component of the wavevector.
        """
        super().__init__()
        self.permittivity = permittivity
        self.theta = np.arcsin(kx / np.sqrt(permittivity)).astype(np.float64)

    def construct_tensor(self):
        """
        Construct the tensor for the ambient incident medium.

        Returns:
            np.ndarray: The constructed tensor.
        """
        return self._construct_tensor()

    def _construct_tensor(self):
        n = np.sqrt(self.permittivity)
        cos_theta = np.cos(self.theta)
        n_cos_theta = n * cos_theta

        # Combine updates into a single tensor with shape [180, 4, 4]
        element1 = np.stack(
            [
                np.zeros_like(self.theta),
                np.ones_like(self.theta),
                -1.0 / n_cos_theta,
                np.zeros_like(self.theta),
            ],
            axis=-1,
        )
        element2 = np.stack(
            [
                np.zeros_like(self.theta),
                np.ones_like(self.theta),
                1.0 / n_cos_theta,
                np.zeros_like(self.theta),
            ],
            axis=-1,
        )
        element3 = np.stack(
            [
                1.0 / cos_theta,
                np.zeros_like(self.theta),
                np.zeros_like(self.theta),
                1.0 / n * np.ones_like(self.theta),
            ],
            axis=-1,
        )
        element4 = np.stack(
            [
                -1.0 / cos_theta,
                np.zeros_like(self.theta),
                np.zeros_like(self.theta),
                1.0 / n * np.ones_like(self.theta),
            ],
            axis=-1,
        )

        matrix = np.stack([element1, element2, element3, element4], axis=1)
        return 0.5 * matrix.astype(np.complex128)

    def construct_tensor_singular(self):
        """
        Construct the singular tensor for the ambient incident medium.

        Returns:
            np.ndarray: The constructed singular tensor.
        """
        return self._construct_tensor_singular()

    def _construct_tensor_singular(self):
        n = np.sqrt(self.permittivity)
        cos_theta = np.cos(self.theta)
        n_cos_theta = n * cos_theta

        element1 = np.stack([0.0, 1.0, -1.0 / n_cos_theta, 0.0])
        element2 = np.stack([0.0, 1.0, 1.0 / n_cos_theta, 0.0])
        element3 = np.stack([1.0 / cos_theta, 0.0, 0.0, 1.0 / n])
        element4 = np.stack([-1.0 / cos_theta, 0.0, 0.0, 1.0 / n])

        matrix = np.stack([element1, element2, element3, element4], axis=0)
        return 0.5 * matrix.astype(np.complex128)


class AmbientExitMedium(AmbientMedium):
    """
    Class representing the ambient exit medium.
    Moved from material_params.py to better organize layer-related functionality.
    """

    def __init__(self, incident_angle, permittivity_incident, permittivity_exit):
        """
        Initialize the AmbientExitMedium class.

        Args:
            incident_angle (float): The incident angle.
            permittivity_incident (float): The permittivity of the incident medium.
            permittivity_exit (float): The permittivity of the exit medium.
        """
        super().__init__()
        self.theta_incident = incident_angle
        self.N_exit = np.sqrt(permittivity_exit)
        self.N_incident = np.sqrt(permittivity_incident)

    def construct_tensor(self):
        """
        Construct the tensor for the ambient exit medium.

        Returns:
            np.ndarray: The constructed tensor.
        """
        return self._construct_tensor()

    def _construct_tensor(self):
        sin_theta_incident = np.sin(self.theta_incident)
        expr_inside_sqrt = (
            1.0 - ((self.N_incident / self.N_exit) * sin_theta_incident) ** 2.0
        )
        expr_inside_sqrt_complex = expr_inside_sqrt.astype(np.complex128)
        cos_theta_f = np.sqrt(expr_inside_sqrt_complex)
        N_exit = self.N_exit.astype(np.complex128)
        Nf_cos_theta_f = N_exit * cos_theta_f

        element1 = np.stack(
            [
                np.zeros_like(cos_theta_f),
                np.zeros_like(cos_theta_f),
                cos_theta_f,
                -cos_theta_f,
            ],
            axis=-1,
        )

        element2 = np.stack(
            [
                np.ones_like(cos_theta_f),
                np.ones_like(cos_theta_f),
                np.zeros_like(cos_theta_f),
                np.zeros_like(cos_theta_f),
            ],
            axis=-1,
        )

        element3 = np.stack(
            [
                -Nf_cos_theta_f,
                Nf_cos_theta_f,
                np.zeros_like(cos_theta_f),
                np.zeros_like(cos_theta_f),
            ],
            axis=-1,
        )

        element4 = np.stack(
            [
                np.zeros_like(cos_theta_f),
                np.zeros_like(cos_theta_f),
                N_exit * np.ones_like(cos_theta_f),
                N_exit * np.ones_like(cos_theta_f),
            ],
            axis=-1,
        )

        matrix = np.stack([element1, element2, element3, element4], axis=1)
        return matrix.astype(np.complex128)

    def construct_tensor_singular(self):
        """
        Construct the singular tensor for the ambient exit medium.

        Returns:
            np.ndarray: The constructed singular tensor.
        """
        return self._construct_tensor_singular()

    def _construct_tensor_singular(self):
        sin_theta_incident = np.sin(self.theta_incident)
        expr_inside_sqrt = (
            1.0 - ((self.N_incident / self.N_exit) * sin_theta_incident) ** 2.0
        )
        expr_inside_sqrt_complex = np.complex128(expr_inside_sqrt)
        cos_theta_f = np.sqrt(expr_inside_sqrt_complex)
        N_exit = np.complex128(self.N_exit)
        Nf_cos_theta_f = N_exit * cos_theta_f

        element1 = np.array([0.0, 0.0, cos_theta_f, -cos_theta_f])
        element2 = np.array([1.0, 1.0, 0.0, 0.0])
        element3 = np.array([-Nf_cos_theta_f, Nf_cos_theta_f, 0.0, 0.0])
        element4 = np.array([0.0, 0.0, N_exit, N_exit])

        matrix = np.stack([element1, element2, element3, element4], axis=0)
        return matrix.astype(np.complex128)


class Layer(ABC):
    """Abstract base class for a layer in the device."""

    def __init__(self, data, scenario, kx, k0):
        self.type = data.get("type")
        self.material = data.get("material", None)
        self.rotationX = np.float64(m.radians(data.get("rotationX", 0)))
        self.rotationY = np.float64(m.radians(data.get("rotationY", 0))) + 1e-8
        self.rotationZ = np.float64(m.radians(data.get("rotationZ", 0))) + 1.0e-9
        self.rotationZ_type = data.get("rotationZType", "relative")
        self.kx = kx
        self.k0 = k0
        self.frequency = scenario.frequency
        self.scenario = scenario.type
        self.incident_angle = scenario.incident_angle
        self.azimuthal_angle = scenario.azimuthal_angle

        # CHANGED: Remove the default non_magnetic_tensor assignment
        # We'll get both tensors from materials now
        self.eps_tensor = None
        self.mu_tensor = None

        self.thickness = data.get("thickness", None)
        if self.thickness:
            self.thickness = float(self.thickness) * 1e-4

    def material_factory(self):
        """Create the material object based on the material name or specifications.

        This method handles both predefined materials (like Quartz, Sapphire) and
        arbitrary materials specified via a dictionary of parameters.

        Returns:
            Material object: An instance of a material class with the specified properties
        """
        if isinstance(self.material, dict):
            # Create an ArbitraryMaterial instance instead of returning the dict
            self.material = ArbitraryMaterial(self.material)
        elif self.material == "Quartz":
            self.material = Quartz()
        elif self.material == "Sapphire":
            self.material = Sapphire()
        elif self.material == "Calcite":
            self.material = CalciteUpper()
        elif self.material == "CalciteLower":
            self.material = CalciteLower()
        elif self.material == "GalliumOxide":
            self.material = GalliumOxide()
        else:
            raise NotImplementedError(f"Material {self.material} not implemented")

    def calculate_z_rotation(self):
        """
        Calculate the rotation of the layer in the z direction.

        If the scenario is dispersion, azimuthal, or simple, the rotation is relative to
        the azimuthal angle, but can be defined to be static while all other
        layers are rotated. If it's relative, the rotation is added to the
        azimuthal angle as it has been 'shifted'.
        """
        if self.scenario in ["Dispersion", "Azimuthal", "Simple"]:
            if self.rotationZ_type == "relative":
                self.rotationZ = self.azimuthal_angle + self.rotationZ
            elif self.rotationZ_type == "static":
                if self.scenario == "Simple":
                    # For simple scenario, keep as scalar
                    self.rotationZ = self.rotationZ
                else:
                    self.rotationZ = self.rotationZ * np.ones_like(self.azimuthal_angle)

    def calculate_tensors(self):
        """Calculate both permittivity and magnetic tensors for the layer."""
        self.material_factory()

        if self.scenario in ["Incident", "Azimuthal"]:
            self.eps_tensor = self.material.fetch_permittivity_tensor().astype(
                np.complex128
            )
            self.mu_tensor = self.material.fetch_magnetic_tensor().astype(np.complex128)
        elif self.scenario in ["Dispersion", "Simple"]:
            self.eps_tensor = self.material.fetch_permittivity_tensor_for_freq(
                self.frequency
            ).astype(np.complex128)
            self.mu_tensor = self.material.fetch_magnetic_tensor_for_freq(
                self.frequency
            ).astype(np.complex128)

    def rotate_tensors(self):
        """Rotate both permittivity and magnetic tensors according to the rotation angles."""
        if self.scenario in ["Incident", "Dispersion"]:
            rotation_func = anisotropy_rotation_one_value
        elif self.scenario == "Azimuthal":
            rotation_func = anisotropy_rotation_one_axis
        elif self.scenario == "Simple":
            rotation_func = anisotropy_rotation_one_value

        self.eps_tensor = rotation_func(
            self.eps_tensor, self.rotationX, self.rotationY, self.rotationZ
        )
        self.mu_tensor = rotation_func(
            self.mu_tensor, self.rotationX, self.rotationY, self.rotationZ
        )

    # DEPRECATED: Remove this method in favor of calculate_tensors()
    def calculate_eps_tensor(self):
        """Calculate the permittivity tensor for the layer. DEPRECATED - use calculate_tensors()."""
        import warnings

        warnings.warn(
            "calculate_eps_tensor() is deprecated. Use calculate_tensors() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.calculate_tensors()

    # DEPRECATED: Remove this method in favor of rotate_tensors()
    def rotate_tensor(self):
        """Rotate the permittivity tensor according to the rotation angles. DEPRECATED - use rotate_tensors()."""
        import warnings

        warnings.warn(
            "rotate_tensor() is deprecated. Use rotate_tensors() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rotate_tensors()

    @abstractmethod
    def create(self):
        pass


class PrismLayer(Layer):
    """The incident coupling prism layer."""

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_prism = np.float64(data.get("permittivity", 5.5))
        self.create()

    def create(self):
        prism = AmbientIncidentMedium(self.eps_prism, self.kx)

        if self.scenario == "Incident":
            self.matrix = prism.construct_tensor()
        elif self.scenario == "Azimuthal":
            self.matrix = prism.construct_tensor_singular()[np.newaxis, np.newaxis, ...]
        elif self.scenario == "Dispersion":
            self.matrix = prism.construct_tensor()[:, np.newaxis, ...]
        elif self.scenario == "Simple":
            self.matrix = prism.construct_tensor_singular()


class AirGapLayer(Layer):
    """The airgap/isotropic middle layer."""

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)

        # Handle complex permittivity input
        perm = data.get("permittivity", 1.0)
        if isinstance(perm, dict):
            if "real" in perm or "imag" in perm:
                self.permittivity = complex(perm.get("real", 0), perm.get("imag", 0))
            else:
                # Handle nested permittivity structure if present
                self.permittivity = {
                    k: (
                        complex(v.get("real", 0), v.get("imag", 0))
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in perm.items()
                }
        else:
            self.permittivity = complex(perm, 0)

        # CHANGED: Handle magnetic permeability input
        mu = data.get("permeability", 1.0)
        if isinstance(mu, dict):
            if "real" in mu or "imag" in mu:
                self.permeability = complex(mu.get("real", 0), mu.get("imag", 0))
            else:
                # Handle nested permeability structure if present
                self.permeability = {
                    k: (
                        complex(v.get("real", 0), v.get("imag", 0))
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in mu.items()
                }
        else:
            self.permeability = complex(mu, 0)

        # CHANGED: Create the isotropic material with both eps and mu
        self.isotropic_material = Air(
            permittivity=self.permittivity, permeability=self.permeability
        )

        # CHANGED: Get both tensors from the material
        self.eps_tensor = self.isotropic_material.fetch_permittivity_tensor()
        self.mu_tensor = self.isotropic_material.fetch_magnetic_tensor()

        self.calculate_mode()
        self.create()

    def calculate_mode(self):
        """Determine the mode of the airgap/isotropic layer."""
        if self.scenario == "Incident":
            self.mode = "airgap"
        elif self.scenario == "Azimuthal":
            self.mode = "azimuthal_airgap"
        elif self.scenario == "Dispersion":
            self.mode = "simple_airgap"
        elif self.scenario == "Simple":
            self.mode = "simple_scalar_airgap"

    def create(self):
        # CHANGED: Pass both tensors instead of duplicating the eps tensor
        self.profile, self.matrix = Wave(
            self.kx,
            self.eps_tensor,
            self.mu_tensor,  # Now passing the actual magnetic tensor
            self.mode,
            k_0=self.k0,
            thickness=self.thickness,
        ).execute()


class CrystalLayer(Layer):
    """Anisotropic crystal of arbitrary orientation and thickness."""

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        # CHANGED: Use the new unified tensor calculation methods
        self.calculate_tensors()  # Get both eps and mu tensors
        self.calculate_z_rotation()
        self.rotate_tensors()  # Rotate both tensors
        self.create()

    def create(self):
        # CHANGED: Pass both tensors to Wave
        self.profile, self.matrix = Wave(
            self.kx,
            self.eps_tensor,
            self.mu_tensor,  # Now using the actual magnetic tensor from material
            self.scenario,
            k_0=self.k0,
            thickness=self.thickness,
        ).execute()


class SemiInfiniteCrystalLayer(Layer):
    """Anisotropic semi-infinite crystal layer."""

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.calculate_z_rotation()
        # CHANGED: Use the new unified tensor calculation methods
        self.calculate_tensors()  # Get both eps and mu tensors
        self.rotate_tensors()  # Rotate both tensors
        self.create()

    def create(self):
        # CHANGED: Pass both tensors to Wave
        self.profile, self.matrix = Wave(
            self.kx,
            self.eps_tensor,
            self.mu_tensor,  # Now using the actual magnetic tensor from material
            self.scenario,
            semi_infinite=True,
        ).execute()


class IsotropicSemiInfiniteLayer(Layer):
    """Isotropic semi-infinite layer with a given permittivity."""

    def __init__(self, data, scenario, kx, k0):
        super().__init__(data, scenario, kx, k0)
        self.eps_incident = (kx.astype(np.float64) / np.sin(self.incident_angle)) ** 2
        self.eps_exit = np.float64(data.get("permittivity"))

        if self.eps_exit is None:
            raise ValueError(
                "No exit permittivity provided for isotropic semi-infinite layer"
            )

        self.create()

    def create(self):
        exit_medium = AmbientExitMedium(
            self.incident_angle, self.eps_incident, self.eps_exit
        )

        if self.scenario == "Incident":
            self.matrix = exit_medium.construct_tensor()
        elif self.scenario == "Azimuthal":
            self.matrix = exit_medium.construct_tensor()[np.newaxis, np.newaxis, ...]
        elif self.scenario == "Dispersion":
            self.matrix = exit_medium.construct_tensor()[:, np.newaxis, ...]
        elif self.scenario == "Simple":
            # For simple scenario, just get the scalar tensor without additional dimensions
            self.matrix = exit_medium.construct_tensor_singular()


class LayerFactory:
    """Factory class for creating layers."""

    def __init__(self):
        self.layer_classes = {
            "Ambient Incident Layer": PrismLayer,
            "Isotropic Middle-Stack Layer": AirGapLayer,
            "Crystal Layer": CrystalLayer,
            "Semi Infinite Anisotropic Layer": SemiInfiniteCrystalLayer,
            "Semi Infinite Isotropic Layer": IsotropicSemiInfiniteLayer,
        }

    def create_layer(self, layer_data, scenario, kx, k0):
        """Create a layer from the layer data."""
        layer_class = self.layer_classes.get(layer_data["type"])
        if layer_class is not None:
            return layer_class(layer_data, scenario, kx, k0)
        else:
            raise ValueError(f"Invalid layer type {layer_data['type']}")
