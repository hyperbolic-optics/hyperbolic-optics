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
import warnings
from typing import Any

import numpy as np

from hyperbolic_optics.axes import A, F, assert_canonical, canonicalize, present
from hyperbolic_optics.layers import LayerFactory
from hyperbolic_optics.materials import create_material
from hyperbolic_optics.scattering import scattering_coefficients
from hyperbolic_optics.scenario import ScenarioSetup

#: Fraction of a subtraction that must survive for its result to carry signal.
#: Double precision holds ~2.2e-16; below this the difference is rounding noise.
#: Well-conditioned points here sit around 1e-3, so there is no grey zone.
MINOR_TRUST_FLOOR = 1e-11


def _minor(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``a - b`` together with the fraction of it that survived.

    A trust of 1 means no cancellation. A trust near the machine epsilon means
    every significant digit was lost to the subtraction and what remains is the
    rounding floor, not a small number.
    """
    difference = a - b
    scale = np.maximum(np.abs(a), np.abs(b))
    trust = np.divide(
        np.abs(difference), scale, out=np.ones_like(scale, dtype=np.float64), where=scale > 0
    )
    return difference, trust


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
        #: Which backend last populated the coefficients, or None before execute.
        self.backend = None
        #: Fraction of batch points the transfer path could not resolve and that
        #: were recomputed with the scattering cascade. None before execute.
        self.repaired_fraction = None
        self._minor_trust = None

    def _reset(self) -> None:
        """Clear per-run state so ``execute`` can be called more than once.

        ``get_layers`` appends, so without this a second ``execute`` on the same
        object stacked a whole new copy of the structure onto the old one --
        two layers became four, then six. Reflection happened to survive it
        (everything appended sits behind a semi-infinite exit and contributes
        nothing), which is what made it dangerous: ``r_pp`` looked right while
        ``layer_absorption`` returned an entry per phantom layer, negative values
        among them, and the scattering cascade hit a singular interface.
        """
        self.layers = []
        self.transfer_matrix = None
        self.backend = None
        self.repaired_fraction = None
        self._minor_trust = None
        for name in ("r_pp", "r_ss", "r_ps", "r_sp", "t_pp", "t_ss", "t_ps", "t_sp"):
            setattr(self, name, None)

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
        self._warn_frequency_out_of_range(layer_data_list)
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

    def _warn_frequency_out_of_range(self, layer_data_list: list[dict[str, Any]]) -> None:
        """Warn when the shared frequency grid leaves a material's fitted band.

        Every layer is evaluated on one frequency grid, which
        :meth:`resolve_frequency` takes from whichever layer supplies one. A
        stack of two dispersive materials therefore evaluates at least one of
        them outside the range its oscillator parameters were fitted over, where
        the factorized form is an extrapolation and can return ``Im(eps) < 0``.
        That is gain: it shows up downstream as negative layer absorptance and a
        reflectance above 1, with nothing to indicate why.
        """
        low, high = float(np.min(self.frequency)), float(np.max(self.frequency))

        for index, layer in enumerate(layer_data_list):
            name = layer.get("material")
            if not isinstance(name, str):
                continue
            band = getattr(create_material(name), "frequency", None)
            if band is None:  # non-dispersive: valid everywhere
                continue

            band_low, band_high = float(np.min(band)), float(np.max(band))
            if low < band_low or high > band_high:
                warnings.warn(
                    f"Layer {index} ({name}) is evaluated over "
                    f"{low:.1f}-{high:.1f} cm^-1, outside the "
                    f"{band_low:.1f}-{band_high:.1f} cm^-1 range its parameters "
                    "were fitted over. Extrapolated permittivity can be "
                    "unphysical (negative absorptance, reflectance above 1). "
                    "Set ScenarioData['frequency'] to a range valid for every "
                    "material in the stack.",
                    UserWarning,
                    stacklevel=3,
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

    def calculate_reflectivity(self, stabilize: bool = True) -> None:
        """Extract reflection coefficients from total transfer matrix.

        Args:
            stabilize: Recompute any batch point whose 2x2 minors cancelled to
                the rounding floor using the scattering cascade. Pass ``False``
                for the literal transfer-matrix result.

        Solves the system of equations to obtain r_pp, r_ss, r_ps, r_sp
        from the boundary conditions encoded in the transfer matrix.

        Note:
            Reflection coefficients are complex and relate incident field
            amplitudes to reflected field amplitudes:
            E_reflected = r · E_incident
        """
        # Boundary-out: the assembled transfer matrix is canonical [A, B, F, 4, 4].
        assert_canonical(self.transfer_matrix, matrix_ndim=2, name="transfer_matrix")
        t = self.transfer_matrix
        bottom_line, bottom_trust = _minor(t[..., 0, 0] * t[..., 2, 2], t[..., 0, 2] * t[..., 2, 0])
        pp, pp_trust = _minor(t[..., 0, 0] * t[..., 3, 2], t[..., 3, 0] * t[..., 0, 2])
        ps, ps_trust = _minor(t[..., 0, 0] * t[..., 1, 2], t[..., 1, 0] * t[..., 0, 2])
        sp, sp_trust = _minor(t[..., 3, 0] * t[..., 2, 2], t[..., 3, 2] * t[..., 2, 0])
        ss, ss_trust = _minor(t[..., 1, 0] * t[..., 2, 2], t[..., 1, 2] * t[..., 2, 0])

        self.r_pp = pp / bottom_line
        self.r_ps = ps / bottom_line
        self.r_sp = sp / bottom_line
        self.r_ss = ss / bottom_line

        # Every coefficient is a 2x2 minor over a denominator. A layer carrying a
        # growing exponential drives the assembled matrix towards rank 1, where
        # all such minors vanish analytically -- so each is computed as the
        # difference of two nearly equal products. r_pp and r_ss survive that
        # because their numerator and the denominator lose precision together and
        # the error cancels in the ratio. A coefficient that is zero by symmetry
        # has no such partner: its numerator cancels all the way to the rounding
        # floor and what is left is noise, returned as a confident 1e-2.
        #
        # _minor reports how much of each subtraction survived, so the points
        # where that has happened are known exactly rather than guessed at.
        self._minor_trust = np.minimum.reduce(
            [bottom_trust, pp_trust, ps_trust, sp_trust, ss_trust]
        )
        if stabilize:
            self._repair_lost_minors()

        # Boundary-out (single presentation rule, see canonical-shape plan 4.6):
        # coefficients are canonical [A, B, F]; reorder to (F, A, B) then squeeze
        # the size-1 axes. This reproduces every scenario's historical output
        # shape (Incident/Azimuthal -> (F, angle); Dispersion -> (A, B);
        # FullSweep -> (F, A, B); Simple -> scalar).
        self.r_pp = self._present(self.r_pp)
        self.r_ps = self._present(self.r_ps)
        self.r_sp = self._present(self.r_sp)
        self.r_ss = self._present(self.r_ss)

    def _repair_lost_minors(self) -> None:
        """Recompute points whose minors cancelled away, via the stable cascade.

        The substitution is per batch point, not per stack: within one sweep only
        a handful of (frequency, angle) points usually cross into the regime
        where a layer's propagation term has driven the assembled matrix to rank
        one. Everywhere else the transfer product is exact and is kept, so this
        changes nothing for a well-conditioned calculation.

        The Redheffer cascade reads the same per-layer eigenmodes but never forms
        a growing exponential, so it has digits left where the product has none.
        """
        lost = self._minor_trust < MINOR_TRUST_FLOOR
        self.repaired_fraction = float(np.mean(lost))
        if not np.any(lost):
            return

        # Best-effort: the cascade has its own singular cases, and failing to
        # improve a point must not turn a returned answer into an exception.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                stable = scattering_coefficients(self.layers, self.k_0)
        except np.linalg.LinAlgError:
            self.repaired_fraction = 0.0
            warnings.warn(
                f"{np.mean(lost):.1%} of batch points lost their reflection "
                "coefficients to cancellation, and the scattering cascade could "
                "not resolve them either (singular matrix). Those points are "
                "unreliable.",
                UserWarning,
                stacklevel=4,
            )
            return

        for name in ("r_pp", "r_ps", "r_sp", "r_ss"):
            current = getattr(self, name)
            replacement = np.broadcast_to(stable[name], np.shape(current))
            # Only take a replacement that is itself a number.
            usable = lost & np.isfinite(replacement)
            setattr(self, name, np.where(usable, replacement, current))

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

    def calculate_scattering(self) -> None:
        """Fill reflection/transmission coefficients via the scattering backend.

        Uses :func:`hyperbolic_optics.scattering.scattering_coefficients` — a
        numerically-stable Redheffer scattering-matrix cascade built from each
        layer's eigenmodes — instead of the transfer-matrix product. Sets
        ``r_pp … t_ss`` in the same presentation layout as
        :meth:`calculate_reflectivity`.
        """
        coefficients = scattering_coefficients(self.layers, self.k_0)
        for name, value in coefficients.items():
            setattr(self, name, present(value))

    def execute(self, payload: dict[str, Any], backend: str = "transfer") -> None:
        """Execute complete simulation from configuration payload.

        Args:
            payload: Dictionary with 'ScenarioData' and 'Layers' keys.
            backend: ``"transfer"`` (default) for the 4×4 transfer-matrix product,
                or ``"scattering"`` for the numerically-stable scattering-matrix
                backend (correct for thick / lossy / strongly-evanescent stacks
                where the transfer product overflows). Both yield the same
                coefficients where the transfer method is well-conditioned.

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
        if backend not in ("transfer", "scattering"):
            raise ValueError(f"Unknown backend {backend!r}; use 'transfer' or 'scattering'.")

        self._reset()

        # Get the scenario data
        self.get_scenario(payload.get("ScenarioData"))

        if backend == "scattering":
            # The per-layer transfer matrices built in get_layers can overflow for
            # thick/evanescent layers -- harmless here, since the scattering
            # backend reads each layer's eigenmodes (profile), not its matrix. The
            # errstate silences that expected overflow (the scattering engine
            # itself never forms a growing exponential).
            with np.errstate(over="ignore", invalid="ignore"):
                self.get_layers(payload.get("Layers", None))
                self.calculate_scattering()
            self.backend = backend
            return

        # Get the layers (builds each layer's eigenmodes / matrices)
        self.get_layers(payload.get("Layers", None))

        # Calculate the transfer matrix
        self.calculate()

        # Calculate the reflectivity
        self.calculate_reflectivity()
        self.backend = backend
        self._warn_if_ill_conditioned()

    def _warn_if_ill_conditioned(self) -> None:
        """Flag transfer-matrix results that the product has already ruined.

        The transfer-matrix product carries growing exponentials for thick,
        lossy or strongly evanescent layers. It does not fail cleanly: well
        before the terms overflow to inf, cancellation between them costs enough
        significant digits to return finite, plausible, wrong coefficients. Both
        symptoms -- non-finite entries, and a passive stack reflecting more than
        it receives -- point at the same fix, so name it.
        """
        coefficients = [self.r_pp, self.r_ss, self.r_ps, self.r_sp]
        if any(c is None for c in coefficients):
            return

        arrays = [np.asarray(c) for c in coefficients]
        if not all(np.isfinite(a).all() for a in arrays):
            warnings.warn(
                "The transfer-matrix product produced non-finite reflection "
                "coefficients, which happens when a layer is thick, lossy or "
                "strongly evanescent. Re-run with "
                "structure.execute(payload, backend='scattering').",
                UserWarning,
                stacklevel=3,
            )
            return

        reflected_p = np.abs(arrays[0]) ** 2 + np.abs(arrays[2]) ** 2
        reflected_s = np.abs(arrays[1]) ** 2 + np.abs(arrays[3]) ** 2
        excess = max(float(np.max(reflected_p)), float(np.max(reflected_s)))
        if excess > 1.0 + 1e-6:
            warnings.warn(
                f"Reflectance reaches {excess:.3g} > 1 from a passive stack. "
                "This is usually the transfer-matrix product losing precision to "
                "a growing exponential; re-run with "
                "structure.execute(payload, backend='scattering'). It can also "
                "mean a material is being evaluated outside its fitted "
                "frequency range.",
                UserWarning,
                stacklevel=3,
            )
