"""
Material Parameters

This module defines the material parameters for various anisotropic materials, such as Quartz, Sapphire, and Calcite.
It includes permittivity and permeability tensors, recommended frequencies, and other relevant parameters.

The module reads material parameters from a JSON configuration file and provides classes to interact with these materials.
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

class UniaxialMaterial:
    """
    Abstract class for anisotropic materials, such as Quartz, Sapphire, and Calcite.
    """
    def __init__(self, frequency_length=410, run_on_device_decorator=run_on_device):
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def permittivity_calc_for_freq(self, frequency, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """Calculate the permittivity for a given frequency using the provided parameters."""
        frequency = tf.expand_dims(tf.constant([frequency], dtype=tf.complex128), 0)
        omega_ln_expanded = tf.expand_dims(omega_ln, 1)
        gamma_ln_expanded = tf.expand_dims(gamma_ln, 1)
        omega_tn_expanded = tf.expand_dims(omega_tn, 1)
        gamma_tn_expanded = tf.expand_dims(gamma_tn, 1)
        complex_one_j = tf.constant(1j, dtype=tf.complex128)

        top_line = (
            omega_ln_expanded**2.0
            - frequency**2.0
            - complex_one_j * frequency * gamma_ln_expanded
        )
        bottom_line = (
            omega_tn_expanded**2.0
            - frequency**2.0
            - complex_one_j * frequency * gamma_tn_expanded
        )
        result = top_line / bottom_line

        return (high_freq * tf.reduce_prod(result, axis=0))[0]

    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Fetch the permittivity tensor for a requested frequency."""
        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc_for_freq(requested_frequency, **params["extraordinary"])
        eps_ord = self.permittivity_calc_for_freq(requested_frequency, **params["ordinary"])

        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=0)
        eps_tensor = tf.linalg.diag(diag_tensors)

        return eps_tensor

    @run_on_device
    def permittivity_calc(self, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """Calculate the permittivity over the frequency range."""
        frequency = tf.expand_dims(self.frequency, 0)
        omega_ln_expanded = tf.expand_dims(omega_ln, 1)
        gamma_ln_expanded = tf.expand_dims(gamma_ln, 1)
        omega_tn_expanded = tf.expand_dims(omega_tn, 1)
        gamma_tn_expanded = tf.expand_dims(gamma_tn, 1)
        complex_one_j = tf.constant(1j, dtype=tf.complex128)

        top_line = (
            omega_ln_expanded**2.0
            - frequency**2.0
            - complex_one_j * frequency * gamma_ln_expanded
        )
        bottom_line = (
            omega_tn_expanded**2.0
            - frequency**2.0
            - complex_one_j * frequency * gamma_tn_expanded
        )
        result = top_line / bottom_line

        return high_freq * tf.reduce_prod(result, axis=0)

    @run_on_device
    def permittivity_fetch(self):
        """Fetch the permittivity values for the ordinary and extraordinary axes."""
        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])

        return eps_ext, eps_ord

    @run_on_device
    def fetch_permittivity_tensor(self):
        """Fetch the permittivity tensor."""
        eps_ext, eps_ord = self.permittivity_fetch()

        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=1)
        eps_tensor = tf.linalg.diag(diag_tensors)

        return eps_tensor

class Quartz(UniaxialMaterial):
    """Class representing the Quartz material."""

    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["quartz"]
        self.name = params["name"]
        
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )

    @run_on_device
    def permittivity_parameters(self):
        """Get the permittivity parameters for Quartz."""
        params = load_material_parameters()["uniaxial_materials"]["quartz"]["parameters"]
        return {
            axis: {
                key: tf.constant(value, dtype=tf.complex128)
                for key, value in axis_params.items()
            }
            for axis, axis_params in params.items()
        }

class Calcite(UniaxialMaterial):
    """Base class representing the Calcite material."""

    def __init__(self):
        super().__init__()

    @run_on_device
    def permittivity_parameters(self):
        """Get the permittivity parameters for Calcite."""
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["parameters"]
        return {
            axis: {
                key: tf.constant(value, dtype=tf.complex128)
                for key, value in axis_params.items()
            }
            for axis, axis_params in params.items()
        }

class CalciteLower(Calcite):
    """Class representing the lower frequency range of Calcite."""

    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["variants"]["lower"]
        self.name = params["name"]
        
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )


class CalciteUpper(Calcite):
    """Class representing the upper frequency range of Calcite."""

    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["calcite"]["variants"]["upper"]
        self.name = params["name"]
        
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )


class Sapphire(UniaxialMaterial):
    """Class representing the Sapphire material."""

    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["uniaxial_materials"]["sapphire"]
        self.name = params["name"]
        
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )

    @run_on_device
    def permittivity_parameters(self):
        """Get the permittivity parameters for Sapphire."""
        params = load_material_parameters()["uniaxial_materials"]["sapphire"]["parameters"]
        return {
            axis: {
                key: tf.constant(value, dtype=tf.complex128)
                for key, value in axis_params.items()
            }
            for axis, axis_params in params.items()
        }


class MonoclinicMaterial:
    """Base class for monoclinic materials."""
    
    def __init__(self, frequency_length=400, run_on_device_decorator=run_on_device):
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator


class GalliumOxide(MonoclinicMaterial):
    """Class representing the Gallium Oxide material."""

    def __init__(self, freq_min=None, freq_max=None):
        super().__init__()
        params = load_material_parameters()["monoclinic_materials"]["gallium_oxide"]
        self.name = params["name"]
        
        if freq_min is None:
            freq_min = params["frequency_range"]["default_min"]
        if freq_max is None:
            freq_max = params["frequency_range"]["default_max"]
            
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )

    @run_on_device
    def permittivity_parameters(self):
        """Get the permittivity parameters for Gallium Oxide."""
        return load_material_parameters()["monoclinic_materials"]["gallium_oxide"]["parameters"]

    def permittivity_calc(self):
        """Calculate the permittivity components for Gallium Oxide."""
        parameters = self.permittivity_parameters()

        frequency = self.frequency[:, tf.newaxis]
        
        # Calculate Bu modes
        partial_calc_tn_bu = parameters["Bu"]["amplitude"]**2. / (
            parameters["Bu"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1.j * frequency * parameters["Bu"]["gamma_tn"]
        )

        eps_xx_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0,
            axis=1
        )
        eps_xy_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0) 
            * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0),
            axis=1
        )
        eps_yy_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0,
            axis=1
        )

        # Calculate Au modes
        partial_calc_tn_au = parameters["Au"]["amplitude"]**2. / (
            parameters["Au"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1.j * frequency * parameters["Au"]["gamma_tn"]
        )
        eps_zz_au = tf.reduce_sum(partial_calc_tn_au, axis=1)

        # Combine results with high frequency components
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx, eps_yy, eps_zz, eps_xy
    
    @run_on_device
    def fetch_permittivity_tensor(self):
        """Get the full permittivity tensor for Gallium Oxide."""
        eps_xx, eps_yy, eps_zz, eps_xy = self.permittivity_calc()

        eps_tensor = tf.stack(
            [
                [eps_xx, eps_xy, tf.zeros_like(eps_xx)],
                [eps_xy, eps_yy, tf.zeros_like(eps_xx)],
                [tf.zeros_like(eps_xx), tf.zeros_like(eps_xx), eps_zz]
            ],
            axis=-1,
        )
        eps_tensor = tf.transpose(eps_tensor, perm=[1, 2, 0])

        return eps_tensor
    
    @run_on_device
    def permittivity_calc_for_freq(self, frequency):
        """Calculate permittivity components for a specific frequency."""
        parameters = self.permittivity_parameters()

        frequency = tf.constant([[frequency]], dtype=tf.complex128)

        # Calculate Bu modes for specific frequency
        partial_calc_tn_bu = parameters["Bu"]["amplitude"]**2. / (
            parameters["Bu"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1.j * frequency * parameters["Bu"]["gamma_tn"]
        )

        eps_xx_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0,
            axis=1
        )
        eps_xy_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0) 
            * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0),
            axis=1
        )
        eps_yy_bu = tf.reduce_sum(
            partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0,
            axis=1
        )

        # Calculate Au modes for specific frequency
        partial_calc_tn_au = parameters["Au"]["amplitude"]**2. / (
            parameters["Au"]["omega_tn"]**2.0 
            - frequency**2.0 
            - 1.j * frequency * parameters["Au"]["gamma_tn"]
        )
        eps_zz_au = tf.reduce_sum(partial_calc_tn_au, axis=1)

        # Combine results with high frequency components
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx[0], eps_yy[0], eps_zz[0], eps_xy[0]

    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """Get the permittivity tensor for a specific frequency."""
        eps_xx, eps_yy, eps_zz, eps_xy = self.permittivity_calc_for_freq(requested_frequency)

        eps_tensor = tf.stack(
            [
                [eps_xx, eps_xy, tf.zeros_like(eps_xx)],
                [eps_xy, eps_yy, tf.zeros_like(eps_xx)],
                [tf.zeros_like(eps_xx), tf.zeros_like(eps_xx), eps_zz]
            ],
            axis=0,
        )

        return eps_tensor


class ArbitraryMaterial:
    """
    Class representing a material with arbitrary permittivity tensor components.
    
    This class allows for the creation of materials with custom permittivity tensors,
    making it useful for modeling novel or hypothetical materials, or for creating
    simplified material models.
    """
    
    def __init__(self, material_data=None, run_on_device_decorator=run_on_device):
        """
        Initialize an arbitrary material with custom permittivity tensor components.
        
        Args:
            material_data (dict): Dictionary containing the material parameters. If None,
                                defaults will be loaded from the configuration file.
            run_on_device_decorator (function): Device execution decorator function.
        """
        self.name = "Arbitrary Material"
        self.run_on_device = run_on_device_decorator
        
        # Load default parameters if no material data provided
        if material_data is None:
            material_data = load_material_parameters()["arbitrary_materials"]["default"]
        
        # Helper function to convert various formats to complex numbers
        def to_complex(value):
            """Convert various input formats to complex numbers."""
            if value is None:
                return complex(0, 0)
            if isinstance(value, dict):
                # Handle {"real": x, "imag": y} format
                return complex(value.get("real", 0), value.get("imag", 0))
            if isinstance(value, str):
                # Handle "x+yj" format by letting Python's complex() handle it
                try:
                    return complex(value.replace(" ", ""))
                except ValueError:
                    return complex(0, 0)
            # Handle real numbers
            return complex(value, 0)
        
        # Convert each component with defaults
        self.eps_xx = to_complex(material_data.get("eps_xx", 1.0))
        self.eps_yy = to_complex(material_data.get("eps_yy", 1.0))
        self.eps_zz = to_complex(material_data.get("eps_zz", 1.0))
        self.eps_xy = to_complex(material_data.get("eps_xy", 0.0))
        self.eps_xz = to_complex(material_data.get("eps_xz", 0.0))
        self.eps_yz = to_complex(material_data.get("eps_yz", 0.0))
        self.mu_r = to_complex(material_data.get("mu_r", 1.0))
    
    @run_on_device
    def fetch_permittivity_tensor(self):
        """
        Construct and return the complete permittivity tensor for the material.
        
        Returns:
            tf.Tensor: The 3x3 permittivity tensor with complex values.
        """
        tensor_elements = [
            [self.eps_xx, self.eps_xy, self.eps_xz],
            [self.eps_xy, self.eps_yy, self.eps_yz],
            [self.eps_xz, self.eps_yz, self.eps_zz]
        ]
        return tf.constant(tensor_elements, dtype=tf.complex128)
    
    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """
        Return the permittivity tensor for a specific frequency.
        
        For arbitrary materials, the permittivity is frequency-independent,
        so this returns the same tensor regardless of frequency.
        
        Args:
            requested_frequency (float): The frequency at which to evaluate the tensor.
        
        Returns:
            tf.Tensor: The frequency-independent permittivity tensor.
        """
        return self.fetch_permittivity_tensor()
    
    def construct_magnetic_tensor(self):
        """
        Construct the magnetic permeability tensor scaled by mu_r.
        
        Returns:
            tf.Tensor: The scaled magnetic permeability tensor.
        """
        base_tensor = Air().construct_tensor_singular()
        return base_tensor * self.mu_r


class Air:
    """
    Class representing air or any isotropic medium.
    
    This class provides a more flexible implementation that can handle complex
    permittivity values, making it useful for modeling not just air but any
    isotropic medium.
    """
    
    def __init__(self, permittivity=None, run_on_device_decorator=run_on_device):
        """
        Initialize the Air/isotropic medium class.
        
        Args:
            permittivity (complex, dict, float, optional): The permittivity value. Can be:
                - Complex number
                - Dict with 'real' and 'imag' keys
                - Float (treated as real part with 0 imaginary)
                If None, loads default from configuration.
            run_on_device_decorator (function): Device execution decorator function.
        """
        self.run_on_device = run_on_device_decorator
        
        # Load default permittivity if none provided
        if permittivity is None:
            permittivity = load_material_parameters()["isotropic_materials"]["air"]["permittivity"]
        
        # Handle different permittivity input formats
        if isinstance(permittivity, dict):
            self.permittivity = complex(permittivity.get('real', 0), permittivity.get('imag', 0))
        elif isinstance(permittivity, (int, float)):
            self.permittivity = complex(permittivity, 0)
        else:
            self.permittivity = permittivity
    
    @run_on_device
    def construct_tensor_singular(self):
        """
        Construct the singular tensor for the isotropic medium.
        
        Creates a diagonal tensor with the permittivity value along the diagonal,
        representing the isotropic nature of the medium.
        
        Returns:
            tf.Tensor: The constructed singular tensor.
        """
        tensor = tf.constant(
            [[self.permittivity, 0.0, 0.0], 
             [0.0, self.permittivity, 0.0], 
             [0.0, 0.0, self.permittivity]], 
            dtype=tf.complex128
        )
        return tensor
        