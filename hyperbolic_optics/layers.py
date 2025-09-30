"""Layer construction and transfer matrix calculation.

This module implements various layer types for multilayer optical structures:

- Ambient media (incident prism, exit medium)
- Isotropic layers (air gaps, dielectric layers)
- Anisotropic crystal layers (finite thickness)
- Semi-infinite anisotropic layers

Each layer type calculates its own transfer matrix using the 4×4 formalism,
which relates electromagnetic field components at the layer boundaries.
"""

import math as m
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hyperbolic_optics.anisotropy_utils import (
    anisotropy_rotation_one_axis,
    anisotropy_rotation_one_value,
)
from hyperbolic_optics.materials import (
    Air,
    ArbitraryMaterial,
    CalciteLower,
    CalciteUpper,
    GalliumOxide,
    Quartz,
    Sapphire,
)
from hyperbolic_optics.scenario import ScenarioSetup
from hyperbolic_optics.waves import Wave


class AmbientMedium:
    """Base class for ambient mediums (incident and exit)."""

    def __init__(self):
        """Initialize the ambient medium."""


class AmbientIncidentMedium(AmbientMedium):
    """
    Class representing the ambient incident medium.
    Moved from material_params.py to better organize layer-related functionality.
    """

    def __init__(self, permittivity: float, kx: np.ndarray) -> None:
        """Initialize the ambient incident medium (prism).

        Args:
            permittivity: Relative permittivity of the incident medium
            kx: x-component of the wavevector (parallel to interface)
        """
        super().__init__()
        self.permittivity = permittivity
        self.theta = np.arcsin(kx / np.sqrt(permittivity)).astype(np.float64)

    def construct_tensor(self) -> np.ndarray:
        """Construct the transfer matrix for the ambient incident medium.

        Builds the 4×4 transfer matrix relating incident and reflected field
        components in the incident medium. The matrix accounts for both s and p
        polarizations.

        Returns:
            Transfer matrix with shape [180, 4, 4] for incident scenario or
            [1, 1, 4, 4] for azimuthal scenario

        Note:
            This implements the boundary condition matrix for a semi-infinite
            incident medium (prism) as described in the 4×4 formalism.
        """

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

        matrix = np.stack([element1, element2, element3, element4], axis=-2)
        return 0.5 * matrix.astype(np.complex128)

    def construct_tensor_singular(self) -> np.ndarray:
        """Construct transfer matrix for single-point (simple) scenarios.

        Returns:
            Transfer matrix with shape [4, 4] for scalar incident angle

        Note:
            This is a specialized version of construct_tensor for cases where
            only a single incident angle is calculated.
        """

        n = np.sqrt(self.permittivity)
        cos_theta = np.cos(self.theta)
        n_cos_theta = n * cos_theta

        element1 = np.stack([0.0, 1.0, -1.0 / n_cos_theta, 0.0])
        element2 = np.stack([0.0, 1.0, 1.0 / n_cos_theta, 0.0])
        element3 = np.stack([1.0 / cos_theta, 0.0, 0.0, 1.0 / n])
        element4 = np.stack([-1.0 / cos_theta, 0.0, 0.0, 1.0 / n])

        matrix = np.stack([element1, element2, element3, element4], axis=-2)
        return 0.5 * matrix.astype(np.complex128)


class AmbientExitMedium(AmbientMedium):
    """
    Class representing the ambient exit medium.
    Moved from material_params.py to better organize layer-related functionality.
    """

    def __init__(
        self, incident_angle: float, permittivity_incident: float, permittivity_exit: float
    ) -> None:
        """Initialize the ambient exit medium.

        Args:
            incident_angle: Incident angle at the first interface in radians
            permittivity_incident: Permittivity of the incident medium
            permittivity_exit: Permittivity of the exit medium
        """
        super().__init__()
        self.theta_incident = incident_angle
        self.N_exit = np.sqrt(permittivity_exit)
        self.N_incident = np.sqrt(permittivity_incident)

    def construct_tensor(self) -> np.ndarray:
        """Construct the transfer matrix for the ambient exit medium.

        Builds the 4×4 transfer matrix for the semi-infinite exit layer,
        accounting for refraction at the final interface.

        Returns:
            Transfer matrix accounting for transmitted waves only

        Note:
            The exit medium has no reflected waves (semi-infinite), so only
            forward-propagating modes are included.
        """

        sin_theta_incident = np.sin(self.theta_incident)
        expr_inside_sqrt = 1.0 - ((self.N_incident / self.N_exit) * sin_theta_incident) ** 2.0
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

        matrix = np.stack([element1, element2, element3, element4], axis=-2)
        return matrix.astype(np.complex128)

    def construct_tensor_singular(self) -> np.ndarray:
        """Construct transfer matrix for single-point exit medium.

        Returns:
            Transfer matrix with shape [4, 4] for scalar case
        """
        sin_theta_incident = np.sin(self.theta_incident)
        expr_inside_sqrt = 1.0 - ((self.N_incident / self.N_exit) * sin_theta_incident) ** 2.0
        expr_inside_sqrt_complex = np.complex128(expr_inside_sqrt)
        cos_theta_f = np.sqrt(expr_inside_sqrt_complex)
        N_exit = np.complex128(self.N_exit)
        Nf_cos_theta_f = N_exit * cos_theta_f

        element1 = np.array([0.0, 0.0, cos_theta_f, -cos_theta_f])
        element2 = np.array([1.0, 1.0, 0.0, 0.0])
        element3 = np.array([-Nf_cos_theta_f, Nf_cos_theta_f, 0.0, 0.0])
        element4 = np.array([0.0, 0.0, N_exit, N_exit])

        matrix = np.stack([element1, element2, element3, element4], axis=-2)
        return matrix.astype(np.complex128)


class Layer(ABC):
    """Abstract base class for a layer in the device."""

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize a generic layer.

        Args:
            data: Dictionary containing layer parameters (type, material, rotations, etc.)
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber (2π/λ)

        Note:
            This is an abstract base class. Use specific layer types like
            PrismLayer, AirGapLayer, or CrystalLayer instead.
        """
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

    def material_factory(self) -> None:
        """Create the material object based on material name or specifications.

        Instantiates the appropriate material class (Quartz, Calcite, etc.) or
        creates an ArbitraryMaterial from custom parameters.

        Raises:
            NotImplementedError: If the material name is not recognized

        Note:
            For arbitrary materials, expects a dictionary with permittivity
            and optionally permeability tensor components.
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

    def calculate_z_rotation(self) -> None:
        """Calculate the z-axis rotation based on scenario type and rotation mode.

        Determines whether the z-rotation is relative to the azimuthal angle or
        static (fixed in space). For dispersion and azimuthal scenarios, adds
        the azimuthal angle to relative rotations.

        Note:
            The rotationZType attribute controls whether rotation is 'relative'
            (rotates with sample) or 'static' (fixed in lab frame).
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
        elif self.scenario == "FullSweep":
            # Create 2D grid [N_incident, N_azim]
            if self.rotationZ_type == "relative":
                # Broadcast: [N_incident, 1] + [1, N_azim] -> [N_incident, N_azim]
                self.rotationZ = (
                    self.incident_angle[:, np.newaxis] * 0
                    + self.azimuthal_angle[np.newaxis, :]
                    + self.rotationZ
                )
            else:
                # Static rotation - broadcast scalar to grid
                self.rotationZ = self.rotationZ + np.zeros(
                    (len(self.incident_angle), len(self.azimuthal_angle))
                )

    def calculate_tensors(self) -> None:
        """Calculate both permittivity and permeability tensors for the layer.

        Fetches the material's permittivity (ε) and permeability (μ) tensors
        for the appropriate frequency or frequency range based on scenario type.

        Note:
            For Incident/Azimuthal scenarios, tensors span the full frequency
            range. For Dispersion/Simple scenarios, tensors are calculated for
            a single frequency.
        """
        self.material_factory()

        if self.scenario in ["Incident", "Azimuthal", "FullSweep"]:
            self.eps_tensor = self.material.fetch_permittivity_tensor().astype(np.complex128)
            self.mu_tensor = self.material.fetch_magnetic_tensor().astype(np.complex128)
        elif self.scenario in ["Dispersion", "Simple"]:
            self.eps_tensor = self.material.fetch_permittivity_tensor_for_freq(
                self.frequency
            ).astype(np.complex128)
            self.mu_tensor = self.material.fetch_magnetic_tensor_for_freq(self.frequency).astype(
                np.complex128
            )

    def rotate_tensors(self) -> None:
        """Apply Euler angle rotations to permittivity and permeability tensors.

        Rotates both ε and μ tensors according to the specified Euler angles
        (rotationX, rotationY, rotationZ) to account for crystal orientation.

        Note:
            The rotation function used depends on the scenario type to handle
            proper broadcasting across angle arrays.
        """
        if self.scenario in ["Incident", "Dispersion"]:
            rotation_func = anisotropy_rotation_one_value
        elif self.scenario == "Azimuthal":
            rotation_func = anisotropy_rotation_one_axis
        elif self.scenario == "FullSweep":
            from hyperbolic_optics.anisotropy_utils import anisotropy_rotation_two_axes

            rotation_func = anisotropy_rotation_two_axes
        elif self.scenario == "Simple":
            rotation_func = anisotropy_rotation_one_value

        self.eps_tensor = rotation_func(
            self.eps_tensor, self.rotationX, self.rotationY, self.rotationZ
        )
        self.mu_tensor = rotation_func(
            self.mu_tensor, self.rotationX, self.rotationY, self.rotationZ
        )

        # For Azimuthal, rotation produces [freq, azim, 3, 3], transpose to [azim, freq, 3, 3]
        if self.scenario == "Azimuthal":
            self.eps_tensor = np.swapaxes(self.eps_tensor, 0, 1)
            self.mu_tensor = np.swapaxes(self.mu_tensor, 0, 1)

    @abstractmethod
    def create(self) -> None:
        """Create the layer's transfer matrix and wave profile.

        This method must be implemented by concrete layer classes to construct
        the layer-specific transfer matrix.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass


class PrismLayer(Layer):
    """The incident coupling prism layer."""

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize the incident coupling prism layer.

        Args:
            data: Dictionary containing 'permittivity' key for prism permittivity
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber
        """
        super().__init__(data, scenario, kx, k0)
        self.eps_prism = np.float64(data.get("permittivity", 5.5))
        self.create()

    def create(self) -> None:
        """Create the prism transfer matrix.

        Constructs the appropriate transfer matrix based on scenario type,
        handling different array shapes for Simple, Incident, Azimuthal, and
        Dispersion scenarios.
        """
        # Ensure kx is 1D for prism calculation
        kx_1d = self.kx.flatten() if self.kx.ndim > 1 else self.kx
        prism = AmbientIncidentMedium(self.eps_prism, kx_1d)

        if self.scenario == "Incident":
            # Add frequency dimension: [N_angles, 4, 4] -> [N_angles, 1, 4, 4]
            self.matrix = prism.construct_tensor()[:, np.newaxis, ...]
        elif self.scenario == "Azimuthal":
            # Add dimensions for azimuthal and frequency: [4,4] -> [1, N_freq, 4, 4]
            base_matrix = prism.construct_tensor_singular()
            n_freq = self.k0.shape[0] if hasattr(self.k0, "shape") else 1
            self.matrix = np.broadcast_to(
                base_matrix[np.newaxis, np.newaxis, ...], (1, n_freq, 4, 4)
            )
        elif self.scenario == "Dispersion":
            self.matrix = prism.construct_tensor()[:, np.newaxis, ...]
        elif self.scenario == "FullSweep":
            # Prism varies with incident angle, broadcast over azim and freq
            # [N_incident, 4, 4] -> [N_incident, 1, 1, 4, 4]
            self.matrix = prism.construct_tensor()[:, np.newaxis, np.newaxis, ...]
        elif self.scenario == "Simple":
            self.matrix = prism.construct_tensor_singular()


class AirGapLayer(Layer):
    """The airgap/isotropic middle layer."""

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize an isotropic middle-stack layer (air gap or dielectric).

        Args:
            data: Dictionary with 'thickness', 'permittivity', and optionally 'permeability'
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber

        Note:
            Permittivity and permeability can be specified as scalars or as
            dictionaries with 'real' and 'imag' components.
        """
        super().__init__(data, scenario, kx, k0)

        # Handle complex permittivity input
        perm = data.get("permittivity", 1.0)
        if isinstance(perm, dict):
            if "real" in perm or "imag" in perm:
                self.permittivity = complex(perm.get("real", 0), perm.get("imag", 0))
            else:
                # Handle nested permittivity structure if present
                self.permittivity = {
                    k: (complex(v.get("real", 0), v.get("imag", 0)) if isinstance(v, dict) else v)
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
                    k: (complex(v.get("real", 0), v.get("imag", 0)) if isinstance(v, dict) else v)
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

    def calculate_mode(self) -> None:
        """Determine the calculation mode based on scenario type.

        Sets the internal mode string used by the Wave class to determine
        appropriate tensor shapes and broadcasting patterns.
        """
        if self.scenario == "Incident":
            self.mode = "airgap"
        elif self.scenario == "Azimuthal":
            self.mode = "azimuthal_airgap"
        elif self.scenario == "Dispersion":
            self.mode = "simple_airgap"
        elif self.scenario == "FullSweep":
            self.mode = "full_sweep_airgap"
        elif self.scenario == "Simple":
            self.mode = "simple_scalar_airgap"

    def create(self) -> None:
        """Create the air gap layer transfer matrix and wave profile.

        Constructs the transfer matrix by solving the wave equation for the
        isotropic layer with specified thickness.
        """
        self.profile, self.matrix = Wave(
            self.kx,
            self.eps_tensor,
            self.mu_tensor,  # Now passing the actual magnetic tensor
            self.mode,
            k_0=self.k0,
            thickness=self.thickness,
        ).execute()

        # Add dimensions for broadcasting with crystal layers
        if self.scenario == "Dispersion":
            # Add azimuthal dimension: [180, 4, 4] -> [180, 1, 4, 4]
            self.matrix = self.matrix[:, np.newaxis, ...]
        elif self.scenario == "Azimuthal":
            # Add azimuthal dimension: [410, 4, 4] -> [1, 410, 4, 4]
            self.matrix = self.matrix[np.newaxis, ...]
        elif self.scenario == "FullSweep":
            # Airgap doesn't vary with incident angle, only with frequency
            # Add incident and azimuthal dimensions: [410, 4, 4] -> [1, 1, 410, 4, 4]
            self.matrix = self.matrix[np.newaxis, np.newaxis, ...]


class CrystalLayer(Layer):
    """Anisotropic crystal of arbitrary orientation and thickness."""

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize an anisotropic crystal layer with finite thickness.

        Args:
            data: Dictionary with 'material', 'thickness', and rotation angles
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber

        Note:
            The crystal can be rotated using rotationX, rotationY, and rotationZ
            Euler angles specified in degrees.
        """
        super().__init__(data, scenario, kx, k0)
        self.calculate_tensors()  # Get both eps and mu tensors
        self.calculate_z_rotation()
        self.rotate_tensors()  # Rotate both tensors
        self.create()

    def create(self) -> None:
        """Create the crystal layer transfer matrix and wave profile.

        Solves for the wave modes in the anisotropic layer and constructs the
        transfer matrix accounting for phase accumulation through the layer.
        """
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

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize a semi-infinite anisotropic crystal layer.

        Args:
            data: Dictionary with 'material' and rotation angles
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber

        Note:
            Semi-infinite layers have no thickness parameter as they extend
            infinitely in the +z direction. Only forward-propagating modes exist.
        """
        super().__init__(data, scenario, kx, k0)
        self.calculate_z_rotation()
        # CHANGED: Use the new unified tensor calculation methods
        self.calculate_tensors()  # Get both eps and mu tensors
        self.rotate_tensors()  # Rotate both tensors
        self.create()

    def create(self) -> None:
        """Create the semi-infinite crystal transfer matrix.

        Constructs a transfer matrix that includes only the eigenvectors
        without phase propagation (no thickness dependence).
        """
        self.profile, self.matrix = Wave(
            self.kx,
            self.eps_tensor,
            self.mu_tensor,  # Now using the actual magnetic tensor from material
            self.scenario,
            semi_infinite=True,
        ).execute()


class IsotropicSemiInfiniteLayer(Layer):
    """Isotropic semi-infinite layer with a given permittivity."""

    def __init__(
        self, data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> None:
        """Initialize a semi-infinite isotropic exit layer.

        Args:
            data: Dictionary with 'permittivity' for exit medium
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber

        Raises:
            ValueError: If exit permittivity is not provided
        """
        super().__init__(data, scenario, kx, k0)
        self.eps_incident = (kx.astype(np.float64) / np.sin(self.incident_angle)) ** 2
        self.eps_exit = np.float64(data.get("permittivity"))

        if self.eps_exit is None:
            raise ValueError("No exit permittivity provided for isotropic semi-infinite layer")

        self.create()

    def create(self) -> None:
        """Create the isotropic exit layer transfer matrix.

        Constructs the transfer matrix for the semi-infinite isotropic exit
        medium, accounting for refraction at the final interface.
        """
        exit_medium = AmbientExitMedium(self.incident_angle, self.eps_incident, self.eps_exit)

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

    def __init__(self) -> None:
        """Initialize the layer factory with available layer types.

        Registers all available layer classes for creation based on type string.
        """
        self.layer_classes = {
            "Ambient Incident Layer": PrismLayer,
            "Isotropic Middle-Stack Layer": AirGapLayer,
            "Crystal Layer": CrystalLayer,
            "Semi Infinite Anisotropic Layer": SemiInfiniteCrystalLayer,
            "Semi Infinite Isotropic Layer": IsotropicSemiInfiniteLayer,
        }

    def create_layer(
        self, layer_data: dict[str, Any], scenario: ScenarioSetup, kx: np.ndarray, k0: np.ndarray
    ) -> Layer:
        """Create a layer instance from configuration data.

        Args:
            layer_data: Dictionary containing layer type and parameters
            scenario: The simulation scenario configuration
            kx: x-component of wavevector
            k0: Free-space wavenumber

        Returns:
            An instance of the appropriate Layer subclass

        Raises:
            ValueError: If layer type is not recognized

        Example:
            >>> factory = LayerFactory()
            >>> layer_data = {"type": "Ambient Incident Layer", "permittivity": 50.0}
            >>> layer = factory.create_layer(layer_data, scenario, kx, k0)
        """
        layer_class = self.layer_classes.get(layer_data["type"])
        if layer_class is not None:
            return layer_class(layer_data, scenario, kx, k0)
        else:
            raise ValueError(f"Invalid layer type {layer_data['type']}")
