"""
Materials refactor - Stage 1: Add magnetic tensor support to all materials
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
        self.frequency = None
        # NEW: Default magnetic permeability
        self.mu_r = 1.0
    
    def _initialize_frequency_range(self, params, freq_min=None, freq_max=None):
        """Initialize frequency range based on params or defaults."""
        if "frequency_range" not in params:
            return
        
        freq_range = params["frequency_range"]
        if freq_min is None:
            freq_min = freq_range["default_min"]
        if freq_max is None:
            freq_max = freq_range["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length),
            dtype=tf.complex128
        )
    
    def _create_isotropic_mu_tensor_like(self, eps_tensor):
        """Create isotropic magnetic tensor with same shape as eps_tensor."""
        # Get the shape of eps_tensor without the last two dimensions (which are 3x3)
        batch_shape = tf.shape(eps_tensor)[:-2]
        
        # Create identity matrix
        identity = tf.eye(3, dtype=tf.complex128)
        
        # Broadcast to match eps_tensor shape
        # Add batch dimensions to identity
        for _ in range(len(eps_tensor.shape) - 2):
            identity = tf.expand_dims(identity, 0)
        
        # Tile to match batch dimensions
        tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
        mu_tensor = tf.tile(identity, tile_multiples)
        
        return tf.cast(self.mu_r, dtype=tf.complex128) * mu_tensor
    
    # NEW: All materials must implement this
    @run_on_device
    def fetch_magnetic_tensor(self):
        """Fetch magnetic permeability tensor. Default is isotropic."""
        eps_tensor = self.fetch_permittivity_tensor()
        return self._create_isotropic_mu_tensor_like(eps_tensor)
    
    @run_on_device
    def fetch_magnetic_tensor_for_freq(self, requested_frequency):
        """Fetch magnetic tensor for specific frequency. Default is isotropic."""
        eps_tensor = self.fetch_permittivity_tensor_for_freq(requested_frequency)
        return self._create_isotropic_mu_tensor_like(eps_tensor)

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
    
    def __init__(self, material_type, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"][material_type]
        self.name = params.get("name", "Unnamed Material")
        self.material_type = material_type
        # NEW: Store magnetic permeability
        self.mu_r = mu_r
        
        if "frequency_range" in params:
            self._initialize_frequency_range(params, freq_min, freq_max)
        else:
            self.frequency = None

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

# Concrete uniaxial materials - now with magnetic support
class Quartz(ParameterizedUniaxialMaterial):
    """Quartz material implementation."""
    def __init__(self, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__("quartz", freq_min, freq_max, mu_r)

class Sapphire(ParameterizedUniaxialMaterial):
    """Sapphire material implementation."""
    def __init__(self, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__("sapphire", freq_min, freq_max, mu_r)

class Calcite(ParameterizedUniaxialMaterial):
    """Calcite material implementation."""
    def __init__(self, freq_min=None, freq_max=None, variant=None, mu_r=1.0):
        if variant is None:
            raise ValueError("Calcite material must be instantiated with a variant ('lower' or 'upper')")
        
        calcite_config = load_material_parameters()["uniaxial_materials"]["calcite"]
        super().__init__("calcite", freq_min, freq_max, mu_r)
        
        if variant not in calcite_config["variants"]:
            raise ValueError("Calcite variant must be either 'lower' or 'upper'")
                
        variant_params = calcite_config["variants"][variant]
        self.name = variant_params.get("name", self.name)
        self._initialize_frequency_range(variant_params, freq_min, freq_max)

class CalciteLower(Calcite):
    """Lower frequency range Calcite implementation."""
    def __init__(self, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__(freq_min, freq_max, variant="lower", mu_r=mu_r)

class CalciteUpper(Calcite):
    """Upper frequency range Calcite implementation."""
    def __init__(self, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__(freq_min, freq_max, variant="upper", mu_r=mu_r)

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
    
    def __init__(self, freq_min=None, freq_max=None, mu_r=1.0):
        super().__init__()
        params = load_material_parameters()["monoclinic_materials"]["gallium_oxide"]
        self.name = params["name"]
        self.mu_r = mu_r  # NEW: Store magnetic permeability
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

        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu[0]
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu[0]
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu[0]
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au[0]

        return self._create_permittivity_tensor(eps_xx, eps_yy, eps_zz, eps_xy)

class ArbitraryMaterial(BaseMaterial):
    """Material with arbitrary permittivity and permeability tensor components."""
    
    def __init__(self, material_data=None, run_on_device_decorator=run_on_device):
        super().__init__(run_on_device_decorator=run_on_device_decorator)
        self.name = "Arbitrary Material"
        
        if material_data is None:
            material_data = load_material_parameters()["arbitrary_materials"]["default"]
        
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
        # Permittivity components
        eps_components = {
            'eps_xx': 1.0, 'eps_yy': 1.0, 'eps_zz': 1.0,
            'eps_xy': 0.0, 'eps_xz': 0.0, 'eps_yz': 0.0
        }
        
        # NEW: Magnetic permeability components
        mu_components = {
            'mu_xx': 1.0, 'mu_yy': 1.0, 'mu_zz': 1.0,
            'mu_xy': 0.0, 'mu_xz': 0.0, 'mu_yz': 0.0
        }
        
        all_components = {**eps_components, **mu_components}
        
        for key, default in all_components.items():
            value = material_data.get(key, default)
            setattr(self, key, self._to_complex(value))
        
        # Backward compatibility: if only mu_r is specified, use it for diagonal
        if 'mu_r' in material_data:
            mu_r_val = self._to_complex(material_data['mu_r'])
            self.mu_xx = self.mu_yy = self.mu_zz = mu_r_val
    
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
    
    @run_on_device
    def fetch_magnetic_tensor(self):
        """Construct and return the complete magnetic permeability tensor."""
        tensor_elements = [
            [self.mu_xx, self.mu_xy, self.mu_xz],
            [self.mu_xy, self.mu_yy, self.mu_yz],
            [self.mu_xz, self.mu_yz, self.mu_zz]
        ]
        return tf.constant(tensor_elements, dtype=tf.complex128)
    
    @run_on_device
    def fetch_magnetic_tensor_for_freq(self, requested_frequency):
        """Return frequency-independent magnetic tensor."""
        return self.fetch_magnetic_tensor()

class IsotropicMaterial(BaseMaterial):
    """Base class for isotropic materials like air."""
    
    def __init__(self, permittivity=None, permeability=None, run_on_device_decorator=run_on_device):
        super().__init__(run_on_device_decorator=run_on_device_decorator)
        self.permittivity = self._process_permittivity(permittivity)
        # NEW: Support for magnetic permeability
        self.permeability = self._process_permittivity(permeability) if permeability is not None else complex(1.0, 0.0)
    
    def _process_permittivity(self, permittivity):
        """Convert permittivity input to complex number."""
        if permittivity is None:
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
    
    # NEW: Override the base class methods for isotropic materials
    @run_on_device
    def fetch_permittivity_tensor(self):
        """Get permittivity tensor for isotropic material."""
        return self.construct_tensor_singular()
    
    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Get permittivity tensor for specific frequency (frequency-independent)."""
        return self.construct_tensor_singular()
    
    @run_on_device
    def fetch_magnetic_tensor(self):
        """Get magnetic tensor for isotropic material."""
        return tf.constant(
            [[self.permeability, 0.0, 0.0],
             [0.0, self.permeability, 0.0],
             [0.0, 0.0, self.permeability]],
            dtype=tf.complex128
        )
    
    @run_on_device
    def fetch_magnetic_tensor_for_freq(self, requested_frequency):
        """Get magnetic tensor for specific frequency (frequency-independent)."""
        return self.fetch_magnetic_tensor()

class Air(IsotropicMaterial):
    """Air material implementation."""
    
    def __init__(self, permittivity=None, permeability=None, run_on_device_decorator=run_on_device):
        if permittivity is None:
            params = load_material_parameters()["isotropic_materials"]["air"]
            permittivity = params["permittivity"]
        
        # NEW: Support for magnetic air (for metamaterials, etc.)
        if permeability is None:
            permeability = 1.0
            
        super().__init__(
            permittivity=permittivity,
            permeability=permeability,
            run_on_device_decorator=run_on_device_decorator
        )
        self.name = "Air"