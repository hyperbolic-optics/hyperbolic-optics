"""Main simulation structure and workflow orchestration.

The Structure class is the primary interface for running optical simulations.
It coordinates the entire calculation workflow:

1. Parse scenario configuration
2. Create multilayer stack
3. Calculate transfer matrices for each layer
4. Multiply matrices to get total system response
5. Extract reflection coefficients

The transfer matrix method is a recursive approach where each layer's
effect is represented by a 4×4 matrix relating field components at its
boundaries. The total response is the product of all layer matrices.

Reference: Passler & Paarmann, JOSA B 34, 2128-2139 (2017)
"""

import functools
import math as m
import operator
from typing import Any

import numpy as np

from hyperbolic_optics.layers import LayerFactory
from hyperbolic_optics.materials import CalciteUpper, GalliumOxide, Quartz, Sapphire
from hyperbolic_optics.scenario import ScenarioSetup


class Structure:
    """Main interface for optical simulations.

    Examples:
        Basic single-point calculation:

        >>> payload = {
        ...     "ScenarioData": {
        ...         "type": "Simple",
        ...         "incidentAngle": 45.0,
        ...         "azimuthal_angle": 0.0,
        ...         "frequency": 1460.0
        ...     },
        ...     "Layers": [
        ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
        ...         {"type": "Semi Infinite Anisotropic Layer",
        ...          "material": "Calcite", "rotationY": 90}
        ...     ]
        ... }
        >>> structure = Structure()
        >>> structure.execute(payload)
        >>> print(f"R_pp = {abs(structure.r_pp)**2:.4f}")

        Multi-layer structure with air gap:

        >>> payload = {
        ...     "ScenarioData": {"type": "Incident"},
        ...     "Layers": [
        ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
        ...         {"type": "Isotropic Middle-Stack Layer",
        ...          "thickness": 0.5, "permittivity": 1.0},
        ...         {"type": "Crystal Layer", "material": "Quartz",
        ...          "thickness": 1.0, "rotationY": 70},
        ...         {"type": "Semi Infinite Anisotropic Layer",
        ...          "material": "Sapphire", "rotationY": 90}
        ...     ]
        ... }
        >>> structure = Structure()
        >>> structure.execute(payload)
        >>> # Results are arrays: structure.r_pp.shape = (410, 360)

        Custom material with complex permittivity:

        >>> custom_material = {
        ...     "eps_xx": {"real": 2.5, "imag": 0.1},
        ...     "eps_yy": {"real": 3.0, "imag": 0.05},
        ...     "eps_zz": {"real": -4.0, "imag": 0.5}
        ... }
        >>> payload = {
        ...     "ScenarioData": {"type": "Simple", "incidentAngle": 45.0,
        ...                      "azimuthal_angle": 0.0, "frequency": 1000.0},
        ...     "Layers": [
        ...         {"type": "Ambient Incident Layer", "permittivity": 25.0},
        ...         {"type": "Semi Infinite Anisotropic Layer",
        ...          "material": custom_material, "rotationY": 45}
        ...     ]
        ... }
        >>> structure = Structure()
        >>> structure.execute(payload)
    """

    def __init__(self) -> None:
        """Initialize empty optical structure for simulation.

        Creates placeholder attributes for scenario, layers, reflection
        coefficients, and intermediate calculation results.
        """
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

    def get_scenario(self, scenario_data: dict[str, Any]) -> None:
        """Parse and initialize scenario from configuration data.

        Args:
            scenario_data: Dictionary with scenario type and parameters

        Note:
            Automatically sets up angle and frequency arrays based on
            scenario type.
        """
        self.scenario = ScenarioSetup(scenario_data)
        self.setup_attributes()

    def setup_attributes(self) -> None:
        """Transfer scenario attributes to structure for easy access.

        Copies incident_angle, azimuthal_angle, and frequency from scenario
        to structure attributes.
        """
        self.incident_angle = self.scenario.incident_angle
        self.azimuthal_angle = self.scenario.azimuthal_angle
        self.frequency = self.scenario.frequency

    def get_frequency_range(self, last_layer: dict[str, Any]) -> None:
        """Determine frequency range from material's default range.

        Args:
            last_layer: Dictionary containing material name

        Raises:
            NotImplementedError: If material is not recognized

        Note:
            Automatically uses material-specific frequency range (e.g.,
            1300-1600 cm⁻¹ for Calcite upper band).
        """
        material = last_layer["material"]

        if material == "Quartz":
            self.frequency = Quartz().frequency
        elif material == "Sapphire":
            self.frequency = Sapphire().frequency
        elif material == "Calcite":
            self.frequency = CalciteUpper().frequency
        elif material == "GalliumOxide":
            self.frequency = GalliumOxide().frequency
        else:
            raise NotImplementedError("Material not implemented")

    def calculate_kx_k0(self) -> None:
        """Calculate parallel wavevector and free-space wavenumber.

        Computes:
            kx = n_prism · sin(θ) where n_prism = √ε_prism
            k0 = ω / c = 2π · frequency

        Note:
            kx is conserved across all interfaces (phase matching condition).
        """
        self.k_x = (
            np.sqrt(np.float64(self.eps_prism)) * np.sin(self.incident_angle.astype(np.float64))
        ).astype(np.float64)
        self.k_0 = self.frequency * 2.0 * m.pi

    def get_layers(self, layer_data_list: list[dict[str, Any]]) -> None:
        """Create all layers in the structure from configuration.

        Args:
            layer_data_list: List of layer configuration dictionaries

        Note:
            First layer must be Ambient Incident Layer (prism).
            Automatically determines frequency range if not specified.
        """
        # First Layer is prism, so we parse it
        self.eps_prism = layer_data_list[0].get("permittivity", None)
        if not self.frequency:
            last_layer = layer_data_list[-1]
            if last_layer.get("type") != "Semi Infinite Isotropic Layer":
                self.get_frequency_range(last_layer)
            else:
                self.get_frequency_range(layer_data_list[-2])
        self.calculate_kx_k0()

        # Create prism layer and add it to layers list
        self.layers.append(
            self.factory.create_layer(
                layer_data_list[0],
                self.scenario,
                self.k_x,
                self.k_0,
            )
        )

        # Create the rest of the layers and add them to layers list
        for layer_data in layer_data_list[1:]:
            self.layers.append(
                self.factory.create_layer(
                    layer_data,
                    self.scenario,
                    self.k_x,
                    self.k_0,
                )
            )

    def calculate(self) -> None:
        """Calculate total transfer matrix by multiplying layer matrices.

        Performs matrix multiplication of all layer transfer matrices from
        incident to exit medium: M_total = M_exit · ... · M_2 · M_1 · M_prism

        Note:
            Uses functools.reduce with operator.matmul for efficient
            sequential multiplication.
        """
        self.transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, self.transfer_matrices)

    def calculate_reflectivity(self) -> None:
        """Extract reflection coefficients from total transfer matrix.

        Solves the system of equations to obtain r_pp, r_ss, r_ps, r_sp
        from the boundary conditions encoded in the transfer matrix.

        Note:
            Reflection coefficients are complex and relate incident field
            amplitudes to reflected field amplitudes:
            E_reflected = r · E_incident
        """
        bottom_line = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        )
        self.r_pp = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 3, 2]
            - self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 0, 2]
        ) / bottom_line
        self.r_ps = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 1, 2]
            - (self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 0, 2])
        ) / bottom_line
        self.r_sp = (
            self.transfer_matrix[..., 3, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 3, 2] * self.transfer_matrix[..., 2, 0]
        ) / bottom_line
        self.r_ss = (
            self.transfer_matrix[..., 1, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 1, 2] * self.transfer_matrix[..., 2, 0]
        ) / bottom_line

        # Transpose for Azimuthal and Incident to get (freq, angle) ordering
        if self.scenario.type in ["Azimuthal", "Incident"]:
            self.r_pp = np.swapaxes(self.r_pp, 0, 1)
            self.r_ps = np.swapaxes(self.r_ps, 0, 1)
            self.r_sp = np.swapaxes(self.r_sp, 0, 1)
            self.r_ss = np.swapaxes(self.r_ss, 0, 1)
        elif self.scenario.type == "FullSweep":
            # Squeeze extra dimensions and reorder to [freq, incident, azim]
            # Current: [1, 1, N_incident, N_azim, N_freq] -> [N_freq, N_incident, N_azim]
            self.r_pp = np.moveaxis(np.squeeze(self.r_pp), -1, 0)
            self.r_ps = np.moveaxis(np.squeeze(self.r_ps), -1, 0)
            self.r_sp = np.moveaxis(np.squeeze(self.r_sp), -1, 0)
            self.r_ss = np.moveaxis(np.squeeze(self.r_ss), -1, 0)

    def calculate_transmissivity(self) -> None:
        """Extract transmission coefficients from total transfer matrix.

        Calculates t_pp, t_ss, t_ps, t_sp representing transmission through
        the entire structure.

        Warning:
            Transmission coefficient support is incomplete and may not be
            fully validated. Use with caution.
        """
        bottom_line = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        )
        self.t_pp = (self.transfer_matrix[..., 0, 0]) / bottom_line
        self.t_ps = (-self.transfer_matrix[..., 0, 2]) / bottom_line
        self.t_sp = (-self.transfer_matrix[..., 2, 0]) / bottom_line
        self.t_ss = (self.transfer_matrix[..., 2, 2]) / bottom_line

    def display_layer_info(self) -> None:
        """Print information about all layers in the structure.

        Debugging utility to display layer configuration and properties.
        """
        for layer in self.layers:
            print(layer)

    def execute(self, payload: dict[str, Any]) -> None:
        """Execute complete simulation from configuration payload.

        Args:
            payload: Dictionary with 'ScenarioData' and 'Layers' keys

        Example:
            >>> payload = {
            ...     "ScenarioData": {"type": "Simple", "incidentAngle": 45.0,
            ...                      "azimuthal_angle": 0.0, "frequency": 1460.0},
            ...     "Layers": [
            ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
            ...         {"type": "Semi Infinite Anisotropic Layer",
            ...          "material": "Calcite", "rotationY": 90}
            ...     ]
            ... }
            >>> structure = Structure()
            >>> structure.execute(payload)
            >>> R_pp = abs(structure.r_pp)**2
        """
        # Get the scenario data
        self.get_scenario(payload.get("ScenarioData"))

        # Get the layers
        self.get_layers(payload.get("Layers", None))

        # Calculate the transfer matrix
        self.calculate()

        # Calculate the reflectivity
        self.calculate_reflectivity()

    # def plot(self):
    #     """Plot the reflectivity for the given scenario."""
    #     if self.scenario.type == "Incident":
    #         contour_plot_simple_incidence(self)
    #     elif self.scenario.type == "Azimuthal":
    #         contour_plot_simple_azimuthal(self)
    #     elif self.scenario.type == "Dispersion":
    #         contour_plot_simple_dispersion(self)
