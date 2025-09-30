"""Material definitions and permittivity/permeability calculations.

This module provides material classes for various crystal types.

Uniaxial materials (single optical axis):

- Quartz (α-SiO₂)
- Sapphire (α-Al₂O₃)
- Calcite (CaCO₃) - upper and lower reststrahlen bands

Monoclinic materials (non-zero off-diagonal components):

- Gallium Oxide (β-Ga₂O₃)

Arbitrary materials:

- User-defined permittivity and permeability tensors

Isotropic materials:

- Air/vacuum

Note:
    All materials implement frequency-dependent permittivity using Lorentz
    oscillator models with parameters loaded from material_params.json.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_material_parameters() -> dict[str, Any]:
    """Load material parameters from JSON configuration file.

    Returns:
        Dictionary containing all material parameters organized by material type
        (uniaxial, monoclinic, arbitrary, isotropic)

    Note:
        The configuration file is located at hyperbolic_optics/material_params.json
    """
    config_path = Path(__file__).parent / "material_params.json"
    with open(config_path, "r") as f:
        return json.load(f)


class BaseMaterial:
    """Base class for all materials providing common functionality."""

    def __init__(self, frequency_length: int = 410) -> None:
        """Initialize base material with frequency array length.

        Args:
            frequency_length: Number of frequency points for dispersion calculations
        """
        self.frequency_length = frequency_length
        self.name = "Base Material"
        self.frequency = None
        self.mu_r = 1.0  # Default magnetic permeability

    def _initialize_frequency_range(
        self, params: dict[str, Any], freq_min: float | None = None, freq_max: float | None = None
    ) -> None:
        """Initialize frequency range from parameters or defaults.

        Args:
            params: Material parameters dictionary containing frequency_range
            freq_min: Override minimum frequency in cm⁻¹
            freq_max: Override maximum frequency in cm⁻¹

        Note:
            If freq_min/freq_max are not provided, uses default values from
            the material parameters.
        """
        if "frequency_range" not in params:
            return

        freq_range = params["frequency_range"]
        if freq_min is None:
            freq_min = freq_range["default_min"]
        if freq_max is None:
            freq_max = freq_range["default_max"]

        self.frequency = np.linspace(freq_min, freq_max, self.frequency_length, dtype=np.float64)

    def _create_isotropic_mu_tensor_like(self, eps_tensor: np.ndarray) -> np.ndarray:
        """Create isotropic magnetic permeability tensor matching ε tensor shape.

        Args:
            eps_tensor: Permittivity tensor to match shape

        Returns:
            Magnetic permeability tensor with μ = μᵣ·I where I is identity matrix

        Note:
            Default behavior is non-magnetic (μᵣ = 1), but can be overridden
            by setting self.mu_r in subclasses.
        """
        # Create identity matrix with same shape as eps_tensor
        shape = eps_tensor.shape[:-2] + (3, 3)
        mu_tensor = np.zeros(shape, dtype=np.complex128)

        # Fill diagonal with mu_r
        if len(shape) == 2:  # Simple 3x3 case
            np.fill_diagonal(mu_tensor, self.mu_r)
        else:  # Handle batch dimensions
            # Reshape to 2D, fill diagonal, reshape back
            original_shape = mu_tensor.shape
            n_matrices = np.prod(original_shape[:-2])
            mu_tensor_2d = mu_tensor.reshape(n_matrices, 3, 3)
            for i in range(n_matrices):
                np.fill_diagonal(mu_tensor_2d[i], self.mu_r)
            mu_tensor = mu_tensor_2d.reshape(original_shape)

        return mu_tensor.astype(np.complex128)

    def fetch_magnetic_tensor(self) -> np.ndarray:
        """Fetch magnetic permeability tensor for full frequency range.

        Returns:
            Complex magnetic permeability tensor with shape matching permittivity

        Note:
            Default implementation returns isotropic tensor. Override in subclasses
            for magnetic materials.
        """
        eps_tensor = self.fetch_permittivity_tensor()
        return self._create_isotropic_mu_tensor_like(eps_tensor)

    def fetch_magnetic_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Fetch magnetic permeability tensor for specific frequency.

        Args:
            requested_frequency: Frequency in cm⁻¹

        Returns:
            Complex magnetic permeability tensor at the requested frequency
        """
        eps_tensor = self.fetch_permittivity_tensor_for_freq(requested_frequency)
        return self._create_isotropic_mu_tensor_like(eps_tensor)


class UniaxialMaterial(BaseMaterial):
    """Base class for anisotropic materials with a single optical axis."""

    def permittivity_calc_for_freq(
        self,
        frequency: float,
        high_freq: float,
        omega_tn: np.ndarray,
        gamma_tn: np.ndarray,
        omega_ln: np.ndarray,
        gamma_ln: np.ndarray,
    ) -> complex:
        """Calculate permittivity at a single frequency using Lorentz oscillator model.

        Args:
            frequency: Frequency in cm⁻¹
            high_freq: High-frequency dielectric constant (ε∞)
            omega_tn: Transverse optical phonon frequencies
            gamma_tn: Transverse phonon damping constants
            omega_ln: Longitudinal optical phonon frequencies
            gamma_ln: Longitudinal phonon damping constants

        Returns:
            Complex permittivity at the specified frequency

        Note:
            Uses the factorized form: ε(ω) = ε∞ ∏ᵢ (ωₗᵢ² - ω² - iωγₗᵢ)/(ωₜᵢ² - ω² - iωγₜᵢ)
        """
        frequency = np.array([frequency], dtype=np.float64)

        # Convert parameters to numpy arrays
        omega_ln = np.asarray(omega_ln, dtype=np.complex128)
        gamma_ln = np.asarray(gamma_ln, dtype=np.complex128)
        omega_tn = np.asarray(omega_tn, dtype=np.complex128)
        gamma_tn = np.asarray(gamma_tn, dtype=np.complex128)

        # Expand dimensions for broadcasting
        omega_ln_expanded = omega_ln[:, np.newaxis]
        gamma_ln_expanded = gamma_ln[:, np.newaxis]
        omega_tn_expanded = omega_tn[:, np.newaxis]
        gamma_tn_expanded = gamma_tn[:, np.newaxis]

        top_line = omega_ln_expanded**2.0 - frequency**2.0 - 1j * frequency * gamma_ln_expanded
        bottom_line = omega_tn_expanded**2.0 - frequency**2.0 - 1j * frequency * gamma_tn_expanded
        result = top_line / bottom_line

        return (high_freq * np.prod(result, axis=0))[0]

    def permittivity_calc(
        self,
        high_freq: float,
        omega_tn: np.ndarray,
        gamma_tn: np.ndarray,
        omega_ln: np.ndarray,
        gamma_ln: np.ndarray,
    ) -> np.ndarray:
        """Calculate permittivity over full frequency range.

        Args:
            high_freq: High-frequency dielectric constant
            omega_tn: Transverse optical phonon frequencies
            gamma_tn: Transverse phonon damping constants
            omega_ln: Longitudinal optical phonon frequencies
            gamma_ln: Longitudinal phonon damping constants

        Returns:
            Complex permittivity array over all frequencies
        """
        frequency = np.expand_dims(self.frequency, 0)

        # Convert parameters to numpy arrays
        omega_ln = np.asarray(omega_ln, dtype=np.complex128)
        gamma_ln = np.asarray(gamma_ln, dtype=np.complex128)
        omega_tn = np.asarray(omega_tn, dtype=np.complex128)
        gamma_tn = np.asarray(gamma_tn, dtype=np.complex128)

        omega_ln_expanded = omega_ln[:, np.newaxis]
        gamma_ln_expanded = gamma_ln[:, np.newaxis]
        omega_tn_expanded = omega_tn[:, np.newaxis]
        gamma_tn_expanded = gamma_tn[:, np.newaxis]

        top_line = omega_ln_expanded**2.0 - frequency**2.0 - 1j * frequency * gamma_ln_expanded
        bottom_line = omega_tn_expanded**2.0 - frequency**2.0 - 1j * frequency * gamma_tn_expanded
        result = top_line / bottom_line

        return high_freq * np.prod(result, axis=0)

    def _create_permittivity_tensor(
        self, eps_ext: complex | np.ndarray, eps_ord: complex | np.ndarray
    ) -> np.ndarray:
        """Create diagonal permittivity tensor from extraordinary and ordinary values.

        Args:
            eps_ext: Extraordinary (parallel to optical axis) permittivity
            eps_ord: Ordinary (perpendicular to optical axis) permittivity

        Returns:
            Diagonal tensor with [eps_ord, eps_ord, eps_ext] on diagonal

        Note:
            For uniaxial materials, two components are equal (ordinary) and one
            is different (extraordinary).
        """
        if np.isscalar(eps_ext):
            # Single frequency case
            return np.diag([eps_ord, eps_ord, eps_ext]).astype(np.complex128)
        else:
            # Multiple frequency case - vectorized diagonal matrix creation
            diag_tensors = np.stack([eps_ord, eps_ord, eps_ext], axis=-1)
            # Create diagonal matrices vectorized
            result = np.zeros(diag_tensors.shape[:-1] + (3, 3), dtype=np.complex128)
            diagonal_indices = np.arange(3)
            result[..., diagonal_indices, diagonal_indices] = diag_tensors
            return result

    def fetch_permittivity_tensor(self) -> np.ndarray:
        """Fetch full permittivity tensor for all frequencies.

        Returns:
            Permittivity tensor with shape [N, 3, 3] where N is number of frequencies
        """
        eps_ext, eps_ord = self.permittivity_fetch()
        return self._create_permittivity_tensor(eps_ext, eps_ord)

    def fetch_permittivity_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Fetch permittivity tensor at specific frequency.

        Args:
            requested_frequency: Frequency in cm⁻¹

        Returns:
            Permittivity tensor with shape [3, 3]
        """
        params = self.permittivity_parameters()
        eps_ext = self.permittivity_calc_for_freq(requested_frequency, **params["extraordinary"])
        eps_ord = self.permittivity_calc_for_freq(requested_frequency, **params["ordinary"])
        return self._create_permittivity_tensor(eps_ext, eps_ord)

    def permittivity_fetch(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetch extraordinary and ordinary permittivity values.

        Returns:
            Tuple of (eps_extraordinary, eps_ordinary) arrays
        """
        params = self.permittivity_parameters()
        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])
        return eps_ext, eps_ord


class ParameterizedUniaxialMaterial(UniaxialMaterial):
    """Base class for uniaxial materials with parameters from configuration."""

    def __init__(
        self,
        material_type: str,
        freq_min: float | None = None,
        freq_max: float | None = None,
        mu_r: float = 1.0,
    ) -> None:
        """Initialize uniaxial material from parameter configuration.

        Args:
            material_type: Material identifier in configuration ('quartz', 'sapphire', etc.)
            freq_min: Override minimum frequency in cm⁻¹
            freq_max: Override maximum frequency in cm⁻¹
            mu_r: Relative magnetic permeability (default: 1.0 for non-magnetic)
        """
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"][material_type]
        self.name = params.get("name", "Unnamed Material")
        self.material_type = material_type
        self.mu_r = mu_r

        if "frequency_range" in params:
            self._initialize_frequency_range(params, freq_min, freq_max)
        else:
            self.frequency = None

    def permittivity_parameters(self) -> dict[str, dict[str, np.ndarray]]:
        """Get permittivity parameters from JSON configuration.

        Returns:
            Dictionary containing ordinary and extraordinary axis parameters
        """
        params = load_material_parameters()["uniaxial_materials"][self.material_type]["parameters"]
        return {
            axis: {key: np.array(value, dtype=np.complex128) for key, value in axis_params.items()}
            for axis, axis_params in params.items()
        }


# Concrete uniaxial materials
class Quartz(ParameterizedUniaxialMaterial):
    """Quartz material implementation."""

    # Quartz
    def __init__(
        self, freq_min: float | None = None, freq_max: float | None = None, mu_r: float = 1.0
    ) -> None:
        """Initialize Quartz (α-SiO₂) material.

        Args:
            freq_min: Override minimum frequency (default: 410 cm⁻¹)
            freq_max: Override maximum frequency (default: 600 cm⁻¹)
            mu_r: Relative magnetic permeability (default: 1.0)

        Note:
            Quartz is a uniaxial positive crystal supporting hyperbolic phonon
            polaritons in the far-infrared.
        """
        super().__init__("quartz", freq_min, freq_max, mu_r)


class Sapphire(ParameterizedUniaxialMaterial):
    """Sapphire material implementation."""

    def __init__(
        self, freq_min: float | None = None, freq_max: float | None = None, mu_r: float = 1.0
    ) -> None:
        """Initialize Sapphire (α-Al₂O₃) material.

        Args:
            freq_min: Override minimum frequency (default: 210 cm⁻¹)
            freq_max: Override maximum frequency (default: 1000 cm⁻¹)
            mu_r: Relative magnetic permeability (default: 1.0)

        Note:
            Sapphire is a uniaxial crystal with hyperbolic
            dispersion.
        """
        super().__init__("sapphire", freq_min, freq_max, mu_r)


class Calcite(ParameterizedUniaxialMaterial):
    """Calcite material implementation."""

    def __init__(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        variant: str | None = None,
        mu_r: float = 1.0,
    ) -> None:
        """Initialize Calcite (CaCO₃) material with specified reststrahlen band.

        Args:
            freq_min: Override minimum frequency
            freq_max: Override maximum frequency
            variant: 'lower' for 860-920 cm⁻¹ or 'upper' for 1300-1600 cm⁻¹
            mu_r: Relative magnetic permeability (default: 1.0)

        Raises:
            ValueError: If variant is not 'lower' or 'upper'

        Note:
            Calcite must be instantiated through CalciteLower or CalciteUpper
            subclasses rather than directly.
        """
        if variant is None:
            raise ValueError(
                "Calcite material must be instantiated with a variant ('lower' or 'upper')"
            )

        calcite_config = load_material_parameters()["uniaxial_materials"]["calcite"]
        super().__init__("calcite", freq_min, freq_max, mu_r)

        if variant not in calcite_config["variants"]:
            raise ValueError("Calcite variant must be either 'lower' or 'upper'")

        variant_params = calcite_config["variants"][variant]
        self.name = variant_params.get("name", self.name)
        self._initialize_frequency_range(variant_params, freq_min, freq_max)


class CalciteLower(Calcite):
    """Lower frequency range Calcite implementation."""

    def __init__(
        self, freq_min: float | None = None, freq_max: float | None = None, mu_r: float = 1.0
    ) -> None:
        """Initialize Calcite lower reststrahlen band (860-920 cm⁻¹).

        Args:
            freq_min: Override minimum frequency
            freq_max: Override maximum frequency
            mu_r: Relative magnetic permeability (default: 1.0)
        """
        super().__init__(freq_min, freq_max, variant="lower", mu_r=mu_r)


class CalciteUpper(Calcite):
    """Upper frequency range Calcite implementation."""

    def __init__(
        self, freq_min: float | None = None, freq_max: float | None = None, mu_r: float = 1.0
    ) -> None:
        """Initialize Calcite upper reststrahlen band (1300-1600 cm⁻¹).

        Args:
            freq_min: Override minimum frequency
            freq_max: Override maximum frequency
            mu_r: Relative magnetic permeability (default: 1.0)

        Note:
            The upper band exhibits type-II hyperbolic dispersion (ε_∥ < 0, ε_⊥ > 0).
        """
        super().__init__(freq_min, freq_max, variant="upper", mu_r=mu_r)


class MonoclinicMaterial(BaseMaterial):
    """Base class for monoclinic materials with more complex permittivity tensors."""

    def _calculate_bu_components(
        self, parameters: dict[str, Any], frequency: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bu symmetry mode contributions to permittivity.

        Args:
            parameters: Material parameters including Bu mode data
            frequency: Frequency array in cm⁻¹

        Returns:
            Tuple of (eps_xx_bu, eps_xy_bu, eps_yy_bu) contributions

        Note:
            Bu modes couple x and y components, creating off-diagonal permittivity
            elements characteristic of monoclinic crystals.
        """
        partial_calc_tn_bu = parameters["Bu"]["amplitude"] ** 2.0 / (
            parameters["Bu"]["omega_tn"] ** 2.0
            - frequency**2.0
            - 1j * frequency * parameters["Bu"]["gamma_tn"]
        )

        alpha_rad = parameters["Bu"]["alpha_tn"] * np.pi / 180.0
        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        eps_xx_bu = np.sum(partial_calc_tn_bu * cos_alpha**2.0, axis=1)
        eps_xy_bu = np.sum(partial_calc_tn_bu * sin_alpha * cos_alpha, axis=1)
        eps_yy_bu = np.sum(partial_calc_tn_bu * sin_alpha**2.0, axis=1)

        return eps_xx_bu, eps_xy_bu, eps_yy_bu

    def _calculate_au_component(
        self, parameters: dict[str, Any], frequency: np.ndarray
    ) -> np.ndarray:
        """Calculate Au symmetry mode contribution to zz permittivity component.

        Args:
            parameters: Material parameters including Au mode data
            frequency: Frequency array in cm⁻¹

        Returns:
            eps_zz contribution from Au modes

        Note:
            Au modes affect only the zz component and are decoupled from
            in-plane components.
        """
        partial_calc_tn_au = parameters["Au"]["amplitude"] ** 2.0 / (
            parameters["Au"]["omega_tn"] ** 2.0
            - frequency**2.0
            - 1j * frequency * parameters["Au"]["gamma_tn"]
        )
        return np.sum(partial_calc_tn_au, axis=1)


class GalliumOxide(MonoclinicMaterial):
    """Gallium Oxide implementation."""

    def __init__(
        self, freq_min: float | None = None, freq_max: float | None = None, mu_r: float = 1.0
    ) -> None:
        """Initialize β-Ga₂O₃ (monoclinic) material.

        Args:
            freq_min: Override minimum frequency (default: 350 cm⁻¹)
            freq_max: Override maximum frequency (default: 800 cm⁻¹)
            mu_r: Relative magnetic permeability (default: 1.0)

        Note:
            Gallium oxide is a monoclinic crystal with non-zero ε_xy coupling,
            supporting hyperbolic polaritons with in-plane anisotropy.
        """
        super().__init__()
        params = load_material_parameters()["monoclinic_materials"]["gallium_oxide"]
        self.name = params["name"]
        self.mu_r = mu_r
        self._initialize_frequency_range(params, freq_min, freq_max)

    def permittivity_parameters(self) -> dict[str, dict[str, Any]]:
        """Get Gallium Oxide symmetry mode parameters.

        Returns:
            Dictionary with 'Au' and 'Bu' mode parameters including oscillator
            strengths, frequencies, dampings, and orientation angles
        """
        params = load_material_parameters()["monoclinic_materials"]["gallium_oxide"]["parameters"]
        # Convert all numeric values to numpy arrays
        result = {}
        for mode, mode_params in params.items():
            result[mode] = {}
            for key, value in mode_params.items():
                if isinstance(value, dict):
                    result[mode][key] = value  # Keep high_freq dict as is
                elif isinstance(value, list):
                    result[mode][key] = np.array(value, dtype=np.complex128)
                else:
                    result[mode][key] = np.complex128(value)
        return result

    def _create_permittivity_tensor(
        self, eps_xx: np.ndarray, eps_yy: np.ndarray, eps_zz: np.ndarray, eps_xy: np.ndarray
    ) -> np.ndarray:
        """Create full permittivity tensor with off-diagonal coupling.

        Args:
            eps_xx: xx component
            eps_yy: yy component
            eps_zz: zz component
            eps_xy: xy coupling component

        Returns:
            Full 3×3 permittivity tensor with monoclinic symmetry

        Note:
            The tensor has the form:
            [eps_xx  eps_xy    0   ]
            [eps_xy  eps_yy    0   ]
            [  0       0    eps_zz]
        """
        zeros = np.zeros_like(eps_xx)
        tensor = np.array(
            [[eps_xx, eps_xy, zeros], [eps_xy, eps_yy, zeros], [zeros, zeros, eps_zz]],
            dtype=np.complex128,
        )

        # Move frequency axis to first dimension if needed
        if tensor.shape[-1] != 3:
            tensor = np.moveaxis(tensor, -1, 0)

        return tensor

    def permittivity_calc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate all permittivity tensor components over frequency range.

        Returns:
            Tuple of (eps_xx, eps_yy, eps_zz, eps_xy) arrays
        """
        parameters = self.permittivity_parameters()
        frequency = self.frequency[:, np.newaxis]

        eps_xx_bu, eps_xy_bu, eps_yy_bu = self._calculate_bu_components(parameters, frequency)
        eps_zz_au = self._calculate_au_component(parameters, frequency)

        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx, eps_yy, eps_zz, eps_xy

    def fetch_permittivity_tensor(self) -> np.ndarray:
        """Fetch full permittivity tensor for all frequencies.

        Returns:
            Permittivity tensor with shape [N, 3, 3]
        """
        eps_xx, eps_yy, eps_zz, eps_xy = self.permittivity_calc()
        return self._create_permittivity_tensor(eps_xx, eps_yy, eps_zz, eps_xy)

    def fetch_permittivity_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Fetch permittivity tensor at specific frequency.

        Args:
            requested_frequency: Frequency in cm⁻¹

        Returns:
            Permittivity tensor with shape [3, 3]
        """
        parameters = self.permittivity_parameters()
        frequency = np.array([[requested_frequency]], dtype=np.float64)

        eps_xx_bu, eps_xy_bu, eps_yy_bu = self._calculate_bu_components(parameters, frequency)
        eps_zz_au = self._calculate_au_component(parameters, frequency)

        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu[0]
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu[0]
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu[0]
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au[0]

        return self._create_permittivity_tensor(eps_xx, eps_yy, eps_zz, eps_xy)


class ArbitraryMaterial(BaseMaterial):
    """Material with arbitrary permittivity and permeability tensor components."""

    def __init__(self, material_data: dict[str, Any] | None = None) -> None:
        """Initialize material with arbitrary permittivity and permeability tensors.

        Args:
            material_data: Dictionary with tensor components (eps_xx, eps_yy, etc.)
                        If None, uses default identity-like values

        Example:
            >>> mat_data = {
            ...     "eps_xx": {"real": 2.5, "imag": 0.1},
            ...     "eps_yy": {"real": 3.0, "imag": 0.0},
            ...     "eps_zz": {"real": -4.0, "imag": 0.5},
            ...     "mu_r": 1.0
            ... }
            >>> material = ArbitraryMaterial(mat_data)
        """
        super().__init__()
        self.name = "Arbitrary Material"

        if material_data is None:
            material_data = load_material_parameters()["arbitrary_materials"]["default"]

        self._init_tensor_components(material_data)

    def _to_complex(self, value: Any) -> complex:
        """Convert various input formats to complex numbers.

        Args:
            value: Input value (dict, string, number, or None)

        Returns:
            Complex number representation

        Note:
            Handles dict with 'real'/'imag' keys, string representations,
            and numeric values.
        """
        if value is None:
            return complex(0, 0)
        if isinstance(value, dict):
            return complex(value.get("real", 0), value.get("imag", 0))
        if isinstance(value, str):
            try:
                return complex(value.replace(" ", ""))
            except ValueError:
                return complex(0, 0)
        return complex(value, 0)

    def _init_tensor_components(self, material_data: dict[str, Any]) -> None:
        """Initialize permittivity and permeability tensor components.

        Args:
            material_data: Dictionary with component values

        Note:
            Sets attributes for eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz
            and corresponding mu components. Missing values default to
            appropriate identity-like values.
        """
        # Permittivity components
        eps_components = {
            "eps_xx": 1.0,
            "eps_yy": 1.0,
            "eps_zz": 1.0,
            "eps_xy": 0.0,
            "eps_xz": 0.0,
            "eps_yz": 0.0,
        }

        # Magnetic permeability components
        mu_components = {
            "mu_xx": 1.0,
            "mu_yy": 1.0,
            "mu_zz": 1.0,
            "mu_xy": 0.0,
            "mu_xz": 0.0,
            "mu_yz": 0.0,
        }

        all_components = {**eps_components, **mu_components}

        for key, default in all_components.items():
            value = material_data.get(key, default)
            setattr(self, key, self._to_complex(value))

        # Backward compatibility: if only mu_r is specified
        if "mu_r" in material_data:
            mu_r_val = self._to_complex(material_data["mu_r"])
            self.mu_xx = self.mu_yy = self.mu_zz = mu_r_val

    def fetch_permittivity_tensor(self) -> np.ndarray:
        """Construct full permittivity tensor from components.

        Returns:
            3×3 complex permittivity tensor
        """
        tensor_elements = [
            [self.eps_xx, self.eps_xy, self.eps_xz],
            [self.eps_xy, self.eps_yy, self.eps_yz],
            [self.eps_xz, self.eps_yz, self.eps_zz],
        ]
        return np.array(tensor_elements, dtype=np.complex128)

    def fetch_permittivity_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Return frequency-independent permittivity tensor.

        Args:
            requested_frequency: Frequency in cm⁻¹ (ignored)

        Returns:
            3×3 complex permittivity tensor

        Note:
            Arbitrary materials are frequency-independent by definition.
        """
        return self.fetch_permittivity_tensor()

    def fetch_magnetic_tensor(self) -> np.ndarray:
        """Construct full magnetic permeability tensor from components.

        Returns:
            3×3 complex permeability tensor
        """
        tensor_elements = [
            [self.mu_xx, self.mu_xy, self.mu_xz],
            [self.mu_xy, self.mu_yy, self.mu_yz],
            [self.mu_xz, self.mu_yz, self.mu_zz],
        ]
        return np.array(tensor_elements, dtype=np.complex128)

    def fetch_magnetic_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Return frequency-independent magnetic tensor.

        Args:
            requested_frequency: Frequency in cm⁻¹ (ignored)

        Returns:
            3×3 complex permeability tensor

        Note:
            Arbitrary materials are frequency-independent by definition.
        """
        return self.fetch_magnetic_tensor()


class IsotropicMaterial(BaseMaterial):
    """Base class for isotropic materials like air."""

    def __init__(
        self,
        permittivity: float | complex | dict[str, float] | None = None,
        permeability: float | complex | dict[str, float] | None = None,
    ) -> None:
        """Initialize isotropic material with scalar permittivity and permeability.

        Args:
            permittivity: Relative permittivity (scalar or dict with 'real'/'imag')
            permeability: Relative permeability (scalar or dict with 'real'/'imag')

        Note:
            For isotropic materials, all diagonal tensor components are equal
            and off-diagonal components are zero.
        """
        super().__init__()
        self.permittivity = self._process_permittivity(permittivity)
        self.permeability = (
            self._process_permittivity(permeability)
            if permeability is not None
            else complex(1.0, 0.0)
        )

    def _process_permittivity(
        self, permittivity: float | complex | dict[str, float] | None
    ) -> complex:
        """Convert permittivity input to complex number.

        Args:
            permittivity: Input in various formats

        Returns:
            Complex permittivity value

        Note:
            Handles None (defaults to 1.0), scalars, and dicts with 'real'/'imag'.
        """
        if permittivity is None:
            return complex(1.0, 0.0)

        if isinstance(permittivity, dict):
            return complex(permittivity.get("real", 0), permittivity.get("imag", 0))
        if isinstance(permittivity, (int, float, complex)):
            return complex(permittivity)
        return permittivity

    def construct_tensor_singular(self) -> np.ndarray:
        """Create diagonal tensor with scalar permittivity value.

        Returns:
            3×3 diagonal tensor with permittivity on diagonal
        """
        return np.array(
            [
                [self.permittivity, 0.0, 0.0],
                [0.0, self.permittivity, 0.0],
                [0.0, 0.0, self.permittivity],
            ],
            dtype=np.complex128,
        )

    def fetch_permittivity_tensor(self) -> np.ndarray:
        """Get permittivity tensor for isotropic material.

        Returns:
            3×3 diagonal permittivity tensor
        """
        return self.construct_tensor_singular()

    def fetch_permittivity_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Get frequency-independent permittivity tensor.

        Args:
            requested_frequency: Frequency in cm⁻¹ (ignored)

        Returns:
            3×3 diagonal permittivity tensor
        """
        return self.construct_tensor_singular()

    def fetch_magnetic_tensor(self) -> np.ndarray:
        """Get magnetic permeability tensor for isotropic material.

        Returns:
            3×3 diagonal permeability tensor
        """
        return np.array(
            [
                [self.permeability, 0.0, 0.0],
                [0.0, self.permeability, 0.0],
                [0.0, 0.0, self.permeability],
            ],
            dtype=np.complex128,
        )

    def fetch_magnetic_tensor_for_freq(self, requested_frequency: float) -> np.ndarray:
        """Get frequency-independent magnetic tensor.

        Args:
            requested_frequency: Frequency in cm⁻¹ (ignored)

        Returns:
            3×3 diagonal permeability tensor
        """
        return self.fetch_magnetic_tensor()


class Air(IsotropicMaterial):
    """Air material implementation."""

    def __init__(
        self,
        permittivity: float | complex | dict[str, float] | None = None,
        permeability: float | complex | dict[str, float] | None = None,
    ) -> None:
        """Initialize Air material (vacuum approximation).

        Args:
            permittivity: Relative permittivity (default: 1.0 from config)
            permeability: Relative permeability (default: 1.0)

        Note:
            Air is treated as an isotropic, non-dispersive material with
            ε ≈ 1.0 and μ ≈ 1.0 across all frequencies.
        """
        if permittivity is None:
            params = load_material_parameters()["isotropic_materials"]["air"]
            permittivity = params["permittivity"]

        if permeability is None:
            permeability = 1.0

        super().__init__(permittivity=permittivity, permeability=permeability)
        self.name = "Air"
