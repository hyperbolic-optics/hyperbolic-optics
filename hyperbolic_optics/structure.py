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

from hyperbolic_optics.axes import A, F, assert_canonical, canonicalize, present
from hyperbolic_optics.layers import LayerFactory
from hyperbolic_optics.materials import create_material
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
        self.t_pp = None
        self.t_ss = None
        self.t_ps = None
        self.t_sp = None
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

    def resolve_frequency(self, layer_data_list: list[dict[str, Any]]) -> np.ndarray:
        """Resolve the frequency array (cm⁻¹) for the simulation.

        Precedence: an explicit ``ScenarioData['frequency']`` (scalar or list)
        wins; otherwise fall back to the default range of the *last
        material-bearing layer* (the bulk crystal — an isotropic exit layer has
        no dispersive range, so it is skipped automatically).

        Args:
            layer_data_list: The raw layer configuration dicts.

        Returns:
            A 1-D frequency array (length 1 for single-frequency scenarios).

        Raises:
            ValueError: If no frequency is given and no material can supply a range.
        """
        if self.frequency is not None:
            return np.atleast_1d(np.asarray(self.frequency, dtype=np.float64))
        for layer in reversed(layer_data_list):
            material = layer.get("material")
            if material is not None:
                freq = create_material(material).frequency
                if freq is not None:
                    return np.asarray(freq, dtype=np.float64)
        raise ValueError(
            "No frequency given and no dispersive material to derive a range from; "
            "set ScenarioData['frequency']."
        )

    def calculate_kx_k0(self) -> None:
        """Calculate parallel wavevector and free-space wavenumber.

        Computes:
            kx = n_prism · sin(θ) where n_prism = √ε_prism
            k0 = ω / c = 2π · frequency

        Note:
            kx is conserved across all interfaces (phase matching condition).
        """
        incident_angle = np.asarray(self.incident_angle, dtype=np.float64)
        kx = np.sqrt(np.float64(self.eps_prism)) * np.sin(incident_angle)
        # Canonical layout (see hyperbolic_optics.axes): kx lives on the A axis,
        # k0 on the F axis; all other batch axes (incl. thickness T) are size 1.
        # A swept layer introduces T>1 via broadcasting, so neither needs to know
        # about it here.
        self.k_x = canonicalize(np.atleast_1d(kx).astype(np.float64), batch_axes=(A,))
        k0 = np.atleast_1d(np.asarray(self.frequency, dtype=np.float64)) * 2.0 * m.pi
        self.k_0 = canonicalize(k0, batch_axes=(F,))
        # Boundary-in: kx and k0 enter the pipeline canonical [A, 1, 1] / [1, 1, F].
        assert_canonical(self.k_x, matrix_ndim=0, name="kx")
        assert_canonical(self.k_0, matrix_ndim=0, name="k0")

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
        self._validate_thickness_sweep(layer_data_list)
        # Resolve the frequency array once and share it with the scenario so
        # every layer evaluates its material over the same frequencies.
        self.frequency = self.resolve_frequency(layer_data_list)
        self.scenario.frequency = self.frequency
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

    @staticmethod
    def _validate_thickness_sweep(layer_data_list: list[dict[str, Any]]) -> None:
        """Allow at most one layer with a list-valued ``thickness`` (the T axis).

        A single swept layer defines the canonical thickness axis; multiple list
        thicknesses would need independent axes, which is out of scope. Scalars
        are unaffected.

        Raises:
            ValueError: If more than one layer carries a list/array ``thickness``.
        """
        swept = [
            i
            for i, layer in enumerate(layer_data_list)
            if layer.get("thickness") is not None and not np.isscalar(layer.get("thickness"))
        ]
        if len(swept) > 1:
            raise ValueError(
                "Only one layer may have a list-valued 'thickness' (the canonical T "
                f"axis); got swept layers at indices {swept}. For a 2-D thickness x "
                "thickness grid, put a list thickness on one layer and use "
                "hyperbolic_optics.sweep.ThicknessSweep for the other."
            )

    def calculate(self) -> None:
        """Calculate total transfer matrix by multiplying layer matrices.

        Performs matrix multiplication of all layer transfer matrices from
        incident to exit medium: M_total = M_exit · ... · M_2 · M_1 · M_prism

        Note:
            Uses functools.reduce with operator.matmul for efficient
            sequential multiplication.
        """
        transfer_matrices = [layer.matrix for layer in self.layers]
        self.transfer_matrix = functools.reduce(operator.matmul, transfer_matrices)

    def calculate_reflectivity(self) -> None:
        """Extract reflection coefficients from total transfer matrix.

        Solves the system of equations to obtain r_pp, r_ss, r_ps, r_sp
        from the boundary conditions encoded in the transfer matrix.

        Note:
            Reflection coefficients are complex and relate incident field
            amplitudes to reflected field amplitudes:
            E_reflected = r · E_incident
        """
        # Boundary-out: the assembled transfer matrix is canonical [A, B, F, 4, 4].
        assert_canonical(self.transfer_matrix, matrix_ndim=2, name="transfer_matrix")
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

        # Boundary-out (single presentation rule, see canonical-shape plan 4.6):
        # coefficients are canonical [A, B, F]; reorder to (F, A, B) then squeeze
        # the size-1 axes. This reproduces every scenario's historical output
        # shape (Incident/Azimuthal -> (F, angle); Dispersion -> (A, B);
        # FullSweep -> (F, A, B); Simple -> scalar).
        self.r_pp = self._present(self.r_pp)
        self.r_ps = self._present(self.r_ps)
        self.r_sp = self._present(self.r_sp)
        self.r_ss = self._present(self.r_ss)

    @staticmethod
    def _present(coefficient: np.ndarray) -> np.ndarray:
        """Map a canonical [A, B, F] coefficient to its presentation shape.

        Thin wrapper over :func:`hyperbolic_optics.axes.present` (shared with
        :mod:`hyperbolic_optics.fields` so coefficients and field-resolved
        quantities present identically).
        """
        return present(coefficient)

    def calculate_transmissivity(self) -> None:
        """Extract transmission *amplitude* coefficients from the transfer matrix.

        Computes ``t_pp, t_ps, t_sp, t_ss``. The results are returned in the same
        presentation layout as the reflection coefficients (see :meth:`_present`),
        so ``t_*`` and ``r_*`` share axis ordering. Not called by ``execute()`` by
        default — invoke explicitly after ``calculate()`` if transmission is needed.

        Note:
            These are bare amplitude coefficients. For *power* transmittance,
            layer-resolved absorption, and field profiles computed numerically
            from the propagated fields (energy-conserving ``R + T + ΣA = 1``), use
            :class:`hyperbolic_optics.fields.FieldProfile`, which is the blessed path.
        """
        bottom_line = (
            self.transfer_matrix[..., 0, 0] * self.transfer_matrix[..., 2, 2]
            - self.transfer_matrix[..., 0, 2] * self.transfer_matrix[..., 2, 0]
        )
        self.t_pp = self._present(self.transfer_matrix[..., 0, 0] / bottom_line)
        self.t_ps = self._present(-self.transfer_matrix[..., 0, 2] / bottom_line)
        self.t_sp = self._present(-self.transfer_matrix[..., 2, 0] / bottom_line)
        self.t_ss = self._present(self.transfer_matrix[..., 2, 2] / bottom_line)

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
