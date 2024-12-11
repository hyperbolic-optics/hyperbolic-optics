"""
Material Parameters

This module defines material parameters for various anisotropic and isotropic materials.
Materials are organized into categories (uniaxial, monoclinic, arbitrary, isotropic)
with shared base functionality within each category.

The module prioritizes code reuse and maintainability while preserving the physics
calculations for each material type.
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from hyperbolic_optics.device_config import run_on_device

def load_material_parameters():
    """Load material parameters from the JSON configuration file."""
    config_path = Path(__file__).parent / "material_params.json"
    with open(config_path, "r") as f:
        return json.load(f)

class BaseMaterial:
    """Base class for all materials providing common functionality."""
    
    def __init__(self, frequency_length=410, run_on_device_decorator=run_on_device):
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator
        self.name = "Base Material"
    
    def _initialize_frequency_range(self, params, freq_min=None, freq_max=None):
        """Initialize frequency range based on params or defaults."""
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length),
            dtype=tf.complex128
        )

class UniaxialMaterial(BaseMaterial):
    """Base class for anisotropic materials with a single optical axis."""
    
    @run_on_device
    def permittivity_calc_for_freq(self, frequency, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """Calculate permittivity for a given frequency using provided parameters."""
        frequency = tf.expand_dims(tf.constant([frequency], dtype=tf.complex128), 0)
        omega_ln_expanded = tf.expand_dims(omega_ln, 1)
        gamma_ln_expanded = tf.expand_dims(gamma_ln, 1)
        omega_tn_expanded = tf.expand_dims(omega_tn, 1)
        gamma_tn_expanded = tf.expand_dims(gamma_tn, 1)

        top_line = (
            omega_ln_expanded**2.0
            - frequency**2.0
            - 1j * frequency * gamma_ln_expanded
        )
        bottom_line = (
            omega_tn_expanded**2.0
            - frequency**2.0
            - 1j * frequency * gamma_tn_expanded
        )
        result = top_line / bottom_line

        return (high_freq * tf.reduce_prod(result, axis=0))[0]

    @run_on_device
    def permittivity_calc(self, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """Calculate permittivity over the frequency range."""
        frequency = tf.expand_dims(self.frequency, 0)
        omega_ln_expanded = tf.expand_dims(omega_ln, 1)
        gamma_ln_expanded = tf.expand_dims(gamma_ln, 1)
        omega_tn_expanded = tf.expand_dims(omega_tn, 1)
        gamma_tn_expanded = tf.expand_dims(gamma_tn, 1)

        top_line = (
            omega_ln_expanded**2.0
            - frequency**2.0
            - 1j * frequency * gamma_ln_expanded
        )
        bottom_line = (
            omega_tn_expanded**2.0
            - frequency**2.0
            - 1j * frequency * gamma_tn_expanded
        )
        result = top_line / bottom_line

        return high_freq * tf.reduce_prod(result, axis=0)

    def _create_permittivity_tensor(self, eps_ext, eps_ord):
        """Create permittivity tensor from extraordinary and ordinary values."""
        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=-1)
        return tf.linalg.diag(diag_tensors)

    @run_on_device
    def fetch_permittivity_tensor(self):
        """Fetch full permittivity tensor."""
        eps_ext, eps_ord = self.permittivity_fetch()
        return self._create_permittivity_tensor(eps_ext, eps_ord)

    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Fetch permittivity tensor for a specific frequency."""
        params = self.permittivity_parameters()
        eps_ext = self.permittivity_calc_for_freq(requested_frequency, **params["extraordinary"])
        eps_ord = self.permittivity_calc_for_freq(requested_frequency, **params["ordinary"])
        return self._create_permittivity_tensor(eps_ext, eps_ord)

    @run_on_device
    def permittivity_fetch(self):
        """Fetch permittivity values for ordinary and extraordinary axes."""
        params = self.permittivity_parameters()
        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])
        return eps_ext, eps_ord

class ParameterizedUniaxialMaterial(UniaxialMaterial):
    """Base class for uniaxial materials with parameters from configuration."""
    
    def __init__(self, material_type, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"][material_type]
        self.name = params["name"]
        self.material_type = material_type
        self._initialize_frequency_range(params, freq_min, freq_max)

    @run_on_device
    def permittivity_parameters(self):
        """Get permittivity parameters from configuration."""
        params = load_material_parameters()["uniaxial_materials"][self.material_type]["parameters"]
        return {
            axis: {
                key: tf.constant(value, dtype=tf.complex128)
                for key, value in axis_params.items()
            }
            for axis, axis_params in params.items()
        }

# Concrete uniaxial materials
class Quartz(ParameterizedUniaxialMaterial):
    """Quartz material implementation."""
    def __init__(self, freq_min=None, freq_max=None):
        super().__init__("quartz", freq_min, freq_max)

class Sapphire(ParameterizedUniaxialMaterial):
    """Sapphire material implementation."""
    def __init__(self, freq_min=None, freq_max=None):
        super().__init__("sapphire", freq_min, freq_max)

class Calcite(UniaxialMaterial):
    """Base class for Calcite variants."""
    @run_on_device
    def permittivity_parameters(self):
        """Get permittivity parameters for Calcite."""
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["parameters"]
        return {
            axis: {
                key: tf.constant(value, dtype=tf.complex128)
                for key, value in axis_params.items()
            }
            for axis, axis_params in params.items()
        }

class CalciteLower(Calcite):
    """Lower frequency range Calcite implementation."""
    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["variants"]["lower"]
        self.name = params["name"]
        self._initialize_frequency_range(params, freq_min, freq_max)

class CalciteUpper(Calcite):
    """Upper frequency range Calcite implementation."""
    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["variants"]["upper"]
        self.name = params["name"]
        self._initialize_frequency_range(params, freq_min, freq_max)

class MonoclinicMaterial(BaseMaterial):
    """Base class for monoclinic materials with more complex permittivity tensors."""
    
    def _calculate_bu_components(self, parameters, frequency):
        """Calculate Bu mode components of permittivity."""
        partial_calc_tn_bu = parameters["Bu"]["amplitude"]**2. / (
            parameters["Bu"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1j * frequency * parameters["Bu"]["gamma_tn"]
        )

        alpha_rad = parameters["Bu"]["alpha_tn"] * np.pi / 180.0
        cos_alpha = tf.cos(alpha_rad)
        sin_alpha = tf.sin(alpha_rad)

        eps_xx_bu = tf.reduce_sum(partial_calc_tn_bu * cos_alpha**2.0, axis=1)
        eps_xy_bu = tf.reduce_sum(partial_calc_tn_bu * sin_alpha * cos_alpha, axis=1)
        eps_yy_bu = tf.reduce_sum(partial_calc_tn_bu * sin_alpha**2.0, axis=1)

        return eps_xx_bu, eps_xy_bu, eps_yy_bu

    def _calculate_au_component(self, parameters, frequency):
        """Calculate Au mode component of permittivity."""
        partial_calc_tn_au = parameters["Au"]["amplitude"]**2. / (
            parameters["Au"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1j * frequency * parameters["Au"]["gamma_tn"]
        )
        return tf.reduce_sum(partial_calc_tn_au, axis=1)

class GalliumOxide(MonoclinicMaterial):
    """Gallium Oxide implementation."""
    
    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["monoclinic_materials"]["gallium_oxide"]
        self.name = params["name"]
        self._initialize_frequency_range(params, freq_min, freq_max)

    @run_on_device
    def permittivity_parameters(self):
        """Get Gallium Oxide permittivity parameters."""
        return load_material_parameters()["monoclinic_materials"]["gallium_oxide"]["parameters"]

    def _create_permittivity_tensor(self, eps_xx, eps_yy, eps_zz, eps_xy):
        """Create the full permittivity tensor."""
        return tf.stack(
            [
                [eps_xx, eps_xy, tf.zeros_like(eps_xx)],
                [eps_xy, eps_yy, tf.zeros_like(eps_xx)],
                [tf.zeros_like(eps_xx), tf.zeros_like(eps_xx), eps_zz]
            ],
            axis=-1,
        )

    @run_on_device
    def permittivity_calc(self):
        """Calculate all permittivity components."""
        parameters = self.permittivity_parameters()
        frequency = self.frequency[:, tf.newaxis]
        
        eps_xx_bu, eps_xy_bu, eps_yy_bu = self._calculate_bu_components(parameters, frequency)
        eps_zz_au = self._calculate_au_component(parameters, frequency)

        # Combine with high frequency components
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx, eps_yy, eps_zz, eps_xy

    @run_on_device
    def fetch_permittivity_tensor(self):
        """Get the full permittivity tensor."""
        eps_xx, eps_yy, eps_zz, eps_xy = self.permittivity_calc()
        tensor = self._create_permittivity_tensor(eps_xx, eps_yy, eps_zz, eps_xy)
        return tf.transpose(tensor, perm=[1, 2, 0])

    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Get permittivity tensor for a specific frequency."""
        parameters = self.permittivity_parameters()
        frequency = tf.constant([[requested_frequency]], dtype=tf.complex128)
        
        eps_xx_bu, eps_xy_bu, eps_yy_bu = self._calculate_bu_components(parameters, frequency)
        eps_zz_au = self._calculate_au_component(parameters, frequency)

        # Combine with high frequency components
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu[0]
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu[0]
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu[0]
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au[0]

        return self._create_permittivity_tensor(eps_xx, eps_yy, eps_zz, eps_xy)

class ArbitraryMaterial(BaseMaterial):
    """Material with arbitrary permittivity tensor components."""
    
    def __init__(self, material_data=None, run_on_device_decorator=run_on_device):
        super().__init__(run_on_device_decorator=run_on_device_decorator)
        self.name = "Arbitrary Material"
        
        # Load default parameters if none provided
        if material_data is None:
            material_data = load_material_parameters()["arbitrary_materials"]["default"]
        
        # Initialize tensor components
        self._init_tensor_components(material_data)
    
    def _to_complex(self, value):
        """Convert various input formats to complex numbers."""
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
    
    def _init_tensor_components(self, material_data):
        """Initialize tensor components from material data."""
        components = {
            'eps_xx': 1.0, 'eps_yy': 1.0, 'eps_zz': 1.0,
            'eps_xy': 0.0, 'eps_xz': 0.0, 'eps_yz': 0.0,
            'mu_r': 1.0
        }
        
        for key, default in components.items():
            value = material_data.get(key, default)
            setattr(self, key, self._to_complex(value))
    
    @run_on_device
    def fetch_permittivity_tensor(self):
        """Construct and return the complete permittivity tensor."""
        tensor_elements = [
            [self.eps_xx, self.eps_xy, self.eps_xz],
            [self.eps_xy, self.eps_yy, self.eps_yz],
            [self.eps_xz, self.eps_yz, self.eps_zz]
        ]
        return tf.constant(tensor_elements, dtype=tf.complex128)
    
    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Return frequency-independent permittivity tensor."""
        return self.fetch_permittivity_tensor()
    
    def construct_magnetic_tensor(self):
        """Construct scaled magnetic permeability tensor."""
        base_tensor = Air().construct_tensor_singular()
        return base_tensor * self.mu_r

class IsotropicMaterial(BaseMaterial):
    """Base class for isotropic materials like air."""
    
    def __init__(self, permittivity=None, run_on_device_decorator=run_on_device):
        """
        Initialize with optional custom permittivity.
        
        Args:
            permittivity: Can be None (uses default), a complex number, 
                         a dict with 'real'/'imag' keys, or a number (treated as real part)
            run_on_device_decorator: Device execution decorator function
        """
        super().__init__(run_on_device_decorator=run_on_device_decorator)
        self.permittivity = self._process_permittivity(permittivity)
    
    def _process_permittivity(self, permittivity):
        """
        Convert permittivity input to complex number.
        
        Handles various input formats flexibly to maintain compatibility
        with existing code while providing enhanced functionality.
        """
        if permittivity is None:
            # Load default from configuration
            return complex(1.0, 0.0)
        
        if isinstance(permittivity, dict):
            return complex(
                permittivity.get('real', 0),
                permittivity.get('imag', 0)
            )
        if isinstance(permittivity, (int, float, complex)):
            return complex(permittivity)
        return permittivity
    
    @run_on_device
    def construct_tensor_singular(self):
        """Create diagonal tensor with permittivity value."""
        return tf.constant(
            [[self.permittivity, 0.0, 0.0],
             [0.0, self.permittivity, 0.0],
             [0.0, 0.0, self.permittivity]],
            dtype=tf.complex128
        )

class Air(IsotropicMaterial):
    """
    Air material implementation.
    
    This class represents air or an air-like medium with configurable permittivity.
    It maintains backward compatibility with existing code while providing
    enhanced functionality through its parent class.
    """
    
    def __init__(self, permittivity=None, run_on_device_decorator=run_on_device):
        """
        Initialize Air with optional custom permittivity.
        
        Args:
            permittivity: Optional custom permittivity value. If None, uses default from config
            run_on_device_decorator: Device execution decorator function
        """
        # If no permittivity specified, load from config
        if permittivity is None:
            params = load_material_parameters()["isotropic_materials"]["air"]
            permittivity = params["permittivity"]
            
        super().__init__(
            permittivity=permittivity,
            run_on_device_decorator=run_on_device_decorator
        )
        self.name = "Air"