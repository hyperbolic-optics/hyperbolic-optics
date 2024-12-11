"""
Material Parameters

This module defines the material parameters for various anisotropic materials, such as Quartz, Sapphire, and Calcite.
It includes permittivity and permeability tensors, recommended frequencies, and other relevant parameters.

The module also includes ambient incident and exit mediums.

Future plan is to integrate this better with Layers.py, to separate the parameters.
"""

import tensorflow as tf
from scipy import constants
import numpy as np
from hyperbolic_optics.device_config import run_on_device


class UniaxialMaterial:
    """
    Abstract class for anisotropic materials, such as Quartz, Sapphire, and Calcite.

    This class provides methods to calculate permittivity tensor components for anisotropic materials.
    It assumes that these materials are not magnetic.

    Attributes:
        frequency_length (int): The length of the frequency range.
        run_on_device (function): Decorator function for device execution.
    """

    def __init__(self, frequency_length=410, run_on_device_decorator=run_on_device):
        """
        Initialize the AnisotropicMaterial class.

        Args:
            frequency_length (int): The length of the frequency range. Default is 410.
            run_on_device_decorator (function): Decorator function for device execution. Default is run_on_device.
        """
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def permittivity_calc_for_freq(self, frequency, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """
        Calculate the permittivity for a given frequency using the provided parameters.

        Args:
            frequency (float): The frequency at which to calculate the permittivity.
            high_freq (float): The high-frequency permittivity.
            omega_tn (tf.Tensor): The transverse phonon frequencies.
            gamma_tn (tf.Tensor): The transverse phonon damping coefficients.
            omega_ln (tf.Tensor): The longitudinal phonon frequencies.
            gamma_ln (tf.Tensor): The longitudinal phonon damping coefficients.

        Returns:
            tf.Tensor: The calculated permittivity at the given frequency.
        """
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
        """
        Fetch the permittivity tensor for a requested frequency.

        Args:
            requested_frequency (float): The requested frequency.

        Returns:
            tf.Tensor: The permittivity tensor at the requested frequency.
        """
        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc_for_freq(requested_frequency, **params["extraordinary"])
        eps_ord = self.permittivity_calc_for_freq(requested_frequency, **params["ordinary"])

        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=0)
        eps_tensor = tf.linalg.diag(diag_tensors)

        return eps_tensor

    @run_on_device
    def permittivity_calc(self, high_freq, omega_tn, gamma_tn, omega_ln, gamma_ln):
        """
        Calculate the permittivity over the frequency range.

        Args:
            high_freq (float): The high-frequency permittivity.
            omega_tn (tf.Tensor): The transverse phonon frequencies.
            gamma_tn (tf.Tensor): The transverse phonon damping coefficients.
            omega_ln (tf.Tensor): The longitudinal phonon frequencies.
            gamma_ln (tf.Tensor): The longitudinal phonon damping coefficients.

        Returns:
            tf.Tensor: The calculated permittivity over the frequency range.
        """
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
        """
        Fetch the permittivity values for the ordinary and extraordinary axes.

        Returns:
            tuple: A tuple containing the permittivity values for the extraordinary and ordinary axes.
        """
        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])

        return eps_ext, eps_ord

    @run_on_device
    def fetch_permittivity_tensor(self):
        """
        Fetch the permittivity tensor.

        Returns:
            tf.Tensor: The permittivity tensor.
        """
        eps_ext, eps_ord = self.permittivity_fetch()

        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=1)
        eps_tensor = tf.linalg.diag(diag_tensors)

        return eps_tensor


class Quartz(UniaxialMaterial):
    """
    Class representing the Quartz material.

    Quartz is an anisotropic material with specific permittivity parameters.

    Attributes:
        name (str): The name of the material.
        frequency (tf.Tensor): The frequency range for Quartz.
    """

    def __init__(self, freq_min=410.0, freq_max=600.0):
        """
        Initialize the Quartz class.

        Args:
            freq_min (float): The minimum frequency value. Default is 410.0.
            freq_max (float): The maximum frequency value. Default is 600.0.
        """
        super().__init__()
        self.name = "Quartz"
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )

    @run_on_device
    def permittivity_parameters(self):
        """
        Get the permittivity parameters for Quartz.

        Returns:
            dict: A dictionary containing the permittivity parameters for the ordinary and extraordinary axes.
        """
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(2.356, dtype=tf.complex128),
                "omega_tn": tf.constant(
                    [393.5, 450.0, 695.0, 797.0, 1065.0, 1158.0], dtype=tf.complex128
                ),
                "gamma_tn": tf.constant(
                    [2.1, 4.5, 13.0, 6.9, 7.2, 9.3], dtype=tf.complex128
                ),
                "omega_ln": tf.constant(
                    [403.0, 507.0, 697.6, 810.0, 1226.0, 1155.0], dtype=tf.complex128
                ),
                "gamma_ln": tf.constant(
                    [2.8, 3.5, 13.0, 6.9, 12.5, 9.3], dtype=tf.complex128
                ),
            },
            "extraordinary": {
                "high_freq": tf.constant(2.383, dtype=tf.complex128),
                "omega_tn": tf.constant(
                    [363.5, 487.5, 777.0, 1071.0], dtype=tf.complex128
                ),
                "gamma_tn": tf.constant([4.8, 4.0, 6.7, 6.8], dtype=tf.complex128),
                "omega_ln": tf.constant(
                    [386.7, 550.0, 790.0, 1229.0], dtype=tf.complex128
                ),
                "gamma_ln": tf.constant([7.0, 3.2, 6.7, 12.0], dtype=tf.complex128),
            },
        }

        return parameters


class Calcite(UniaxialMaterial):
    """
    Class representing the Calcite material.

    Calcite is an anisotropic material with specific permittivity parameters.
    """

    def __init__(self):
        """
        Initialize the Calcite class.
        """
        super().__init__()

    @run_on_device
    def permittivity_parameters(self):
        """
        Get the permittivity parameters for Calcite.

        Returns:
            dict: A dictionary containing the permittivity parameters for the ordinary and extraordinary axes.
        """
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(2.7, dtype=tf.complex128),
                "omega_tn": tf.constant(
                    [712, 1410.0, 297.0, 223.0, 102.0], dtype=tf.complex128
                ),
                "gamma_tn": tf.constant(
                    [5.0, 10.0, 14.4, 11.4, 5.7], dtype=tf.complex128
                ),
                "omega_ln": tf.constant(
                    [715, 1550.0, 381.0, 239.0, 123.0], dtype=tf.complex128
                ),
                "gamma_ln": tf.constant(
                    [5.0, 10.0, 14.4, 11.4, 5.7], dtype=tf.complex128
                ),
            },
            "extraordinary": {
                "high_freq": tf.constant(2.4, dtype=tf.complex128),
                "omega_tn": tf.constant([871.0, 303.0, 92.0], dtype=tf.complex128),
                "gamma_tn": tf.constant([3.0, 9.1, 5.6], dtype=tf.complex128),
                "omega_ln": tf.constant([890.0, 387.0, 136.0], dtype=tf.complex128),
                "gamma_ln": tf.constant([3.0, 9.1, 5.6], dtype=tf.complex128),
            },
        }

        return parameters


class CalciteLower(Calcite):
    """
    Class representing the lower frequency range of Calcite.

    Attributes:
        name (str): The name of the material.
        frequency (tf.Tensor): The lower frequency range for Calcite.
    """

    def __init__(self, freq_min=860.0, freq_max=920.0):
        """
        Initialize the CalciteLower class.

        Args:
            freq_min (float): The minimum frequency value. Default is 860.0.
            freq_max (float): The maximum frequency value. Default is 920.0.
        """
        super().__init__()
        self.name = "Calcite-Lower"
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )


class CalciteUpper(Calcite):
    """
    Class representing the upper frequency range of Calcite.

    Attributes:
        name (str): The name of the material.
        frequency (tf.Tensor): The upper frequency range for Calcite.
    """

    def __init__(self, freq_min=1400.0, freq_max=1550.0):
        """
        Initialize the CalciteUpper class.

        Args:
            freq_min (float): The minimum frequency value. Default is 1300.0.
            freq_max (float): The maximum frequency value. Default is 1600.0.
        """
        super().__init__()
        self.name = "Calcite-Upper"
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )


class Sapphire(UniaxialMaterial):
    """
    Class representing the Sapphire material.

    Sapphire is an anisotropic material with specific permittivity parameters.

    Attributes:
        name (str): The name of the material.
        frequency (tf.Tensor): The frequency range for Sapphire.
    """

    def __init__(self, freq_min=210.0, freq_max=1000.0):
        """
        Initialize the Sapphire class.

        Args:
            freq_min (float): The minimum frequency value. Default is 210.0.
            freq_max (float): The maximum frequency value. Default is 1000.0.
        """
        super().__init__()
        self.name = "Sapphire"
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )

    @run_on_device
    def permittivity_parameters(self):
        """
        Get the permittivity parameters for Sapphire.

        Returns:
            dict: A dictionary containing the permittivity parameters for the ordinary and extraordinary axes.
        """
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(3.077, dtype=tf.complex128),
                "omega_tn": tf.constant(
                    [384.99, 439.1, 569.0, 633.63], dtype=tf.complex128
                ),
                "gamma_tn": tf.constant([3.3, 3.1, 4.7, 5.0], dtype=tf.complex128),
                "omega_ln": tf.constant(
                    [387.60, 481.68, 629.50, 906.6], dtype=tf.complex128
                ),
                "gamma_ln": tf.constant([3.1, 1.9, 5.9, 14.7], dtype=tf.complex128),
            },
            "extraordinary": {
                "high_freq": tf.constant(3.072, dtype=tf.complex128),
                "omega_tn": tf.constant([397.52, 582.41], dtype=tf.complex128),
                "gamma_tn": tf.constant([5.3, 3.0], dtype=tf.complex128),
                "omega_ln": tf.constant([510.87, 881.1], dtype=tf.complex128),
                "gamma_ln": tf.constant([1.1, 15.4], dtype=tf.complex128),
            },
        }
        return parameters
    

class ArbitraryMaterial:
    def __init__(self, material_data, run_on_device_decorator=run_on_device):
        self.name = "Arbitrary Material"
        self.run_on_device = run_on_device_decorator
        
        # Helper function to convert various formats to complex numbers
        def to_complex(value):
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
        
        # Convert each component, with defaults
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
        Uses tf.constant to create the tensor directly from complex numbers.
        """
        tensor_elements = [
            [self.eps_xx, self.eps_xy, self.eps_xz],
            [self.eps_xy, self.eps_yy, self.eps_yz],
            [self.eps_xz, self.eps_yz, self.eps_zz]
        ]
        print(tf.constant(tensor_elements, dtype=tf.complex128))
        return tf.constant(tensor_elements, dtype=tf.complex128)
    
    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
        """
        For this material, the permittivity is frequency-independent,
        so we just return the same tensor regardless of frequency.
        """
        return self.fetch_permittivity_tensor()
        
    def construct_magnetic_tensor(self):
        """
        Construct the magnetic permeability tensor scaled by mu_r.
        """
        base_tensor = Air().construct_tensor_singular()
        return base_tensor * self.mu_r


class MonoclinicMaterial:
    
    def __init__(self, frequency_length=400, run_on_device_decorator=run_on_device):
        """
        Initialize the AnisotropicMaterial class.

        Args:
            frequency_length (int): The length of the frequency range. Default is 410.
            run_on_device_decorator (function): Decorator function for device execution. Default is run_on_device.
        """
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator


class GalliumOxide(MonoclinicMaterial):

    def __init__(self, freq_min=350.0, freq_max=800.0):
        super().__init__()
        self.name = "GalliumOxide"
        self.frequency = tf.cast(
            tf.linspace(freq_min, freq_max, self.frequency_length), dtype=tf.complex128
        )


    @run_on_device
    def permittivity_parameters(self):
        parameters = {
            "Au": {
                "high_freq": tf.constant(3.71, dtype=tf.complex128),
                "amplitude": tf.constant(
                    [544.9, 727.1, 592.1, 78], dtype=tf.complex128
                )[tf.newaxis, :],
                "omega_tn": tf.constant(
                    [663.17, 448.66, 296.63, 154.84],
                    dtype=tf.complex128,
                )[tf.newaxis, :],
                "gamma_tn": tf.constant(
                    [3.2, 10.5, 14.9, 2.4], dtype=tf.complex128
                )[tf.newaxis, :],
                "alpha_tn": tf.constant(
                    [0., 0., 0., 0.], dtype=tf.complex128
                )[tf.newaxis, :],
                "omega_ln": tf.constant(
                    [781.3, 562.8, 345.9, 156.3],
                    dtype=tf.complex128,
                )[tf.newaxis, :],
                "gamma_ln": tf.constant(
                    [0., 0., 0., 0.], dtype=tf.complex128
                )[tf.newaxis, :],
                "alpha_ln": tf.constant(
                    [0., 0., 0., 0.], dtype=tf.complex128
                )[tf.newaxis, :],
            },
            "Bu": {
                "high_freq": 
                    {
                        "xx": tf.constant(3.75, dtype=tf.complex128),
                        "yy": tf.constant(3.21, dtype=tf.complex128),
                        "xy": tf.constant(-0.08, dtype=tf.complex128),
                },
                "amplitude": tf.constant(
                    [266.2, 406.5, 821.9, 795.7, 365.8, 164.2, 485.7, 520.7], dtype=tf.complex128
                )[tf.newaxis, :],
                "omega_tn": tf.constant(
                    [743.48, 692.44, 572.52, 432.57, 356.79, 279.15, 262.34, 213.79],
                    dtype=tf.complex128,
                )[tf.newaxis, :],
                "gamma_tn": tf.constant(
                    [11.0, 6.55, 12.36, 10.13, 3.83, 1.98, 1.75, 1.9], dtype=tf.complex128
                )[tf.newaxis, :],
                "alpha_tn": tf.constant(
                    [47.8, 5.1, 106., 21.0, 144., 4., 158.5, 80.9], dtype=tf.complex128
                )[tf.newaxis, :],
                "omega_ln": tf.constant(
                    [810., 770., 709., 595., 389., 305., 286., 269.],
                    dtype=tf.complex128,
                )[tf.newaxis, :],
                "gamma_ln": tf.constant(
                    [0., 0., 0., 0., 0., 0., 0., 0.], dtype=tf.complex128
                )[tf.newaxis, :],
                "alpha_ln": tf.constant(
                    [73., -30., 6., 73., -31., -42., 21., 27.], dtype=tf.complex128
                )[tf.newaxis, :],
            },
        }

        return parameters
    
    def permittivity_calc(self):
        parameters = self.permittivity_parameters()

        frequency = self.frequency[:,tf.newaxis]
        # Calculation for Bu modes
        partial_calc_tn_bu = parameters["Bu"]["amplitude"]**2. / (parameters["Bu"]["omega_tn"]**2.0 - frequency**2.0 - 1.j * frequency * parameters["Bu"]["gamma_tn"])

        eps_xx_bu = tf.reduce_sum(partial_calc_tn_bu * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0, axis=1)
        eps_xy_bu = tf.reduce_sum(partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0) * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0), axis=1)
        eps_yy_bu = tf.reduce_sum(partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0, axis=1)

        # Calculation for Au modes
        partial_calc_tn_au = parameters["Au"]["amplitude"]**2. / (parameters["Au"]["omega_tn"]**2.0 - frequency**2.0 - 1.j * frequency * parameters["Au"]["gamma_tn"])
        eps_zz_au = tf.reduce_sum(partial_calc_tn_au, axis=1)

        # Combine the results
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx, eps_yy, eps_zz, eps_xy
    
    @run_on_device
    def fetch_permittivity_tensor(self):
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
        parameters = self.permittivity_parameters()

        frequency = tf.constant([[frequency]], dtype=tf.complex128)

        # Calculation for Bu modes
        partial_calc_tn_bu = parameters["Bu"]["amplitude"]**2. / (parameters["Bu"]["omega_tn"]**2.0 - frequency**2.0 - 1.j * frequency * parameters["Bu"]["gamma_tn"])

        eps_xx_bu = tf.reduce_sum(partial_calc_tn_bu * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0, axis=1)
        eps_xy_bu = tf.reduce_sum(partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0) * tf.cos(parameters["Bu"]["alpha_tn"] * np.pi / 180.0), axis=1)
        eps_yy_bu = tf.reduce_sum(partial_calc_tn_bu * tf.sin(parameters["Bu"]["alpha_tn"] * np.pi / 180.0)**2.0, axis=1)

        # Calculation for Au modes
        partial_calc_tn_au = parameters["Au"]["amplitude"]**2. / (parameters["Au"]["omega_tn"]**2.0 - frequency**2.0 - 1.j * frequency * parameters["Au"]["gamma_tn"])
        eps_zz_au = tf.reduce_sum(partial_calc_tn_au, axis=1)

        # Combine the results
        eps_xx = parameters["Bu"]["high_freq"]["xx"] + eps_xx_bu
        eps_xy = parameters["Bu"]["high_freq"]["xy"] + eps_xy_bu
        eps_yy = parameters["Bu"]["high_freq"]["yy"] + eps_yy_bu
        eps_zz = parameters["Au"]["high_freq"] + eps_zz_au

        return eps_xx[0], eps_yy[0], eps_zz[0], eps_xy[0]

    @run_on_device
    def fetch_permittivity_tensor_for_freq(self, requested_frequency):
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

class Air:
    """
    Class representing air or an isotropic medium.
    
    Can now handle complex permittivity values.
    """

    def __init__(self, permittivity=1.0, run_on_device_decorator=run_on_device):
        """
        Initialize the Air/isotropic medium class.

        Args:
            permittivity (complex, dict, or float): The permittivity value. Can be:
                - Complex number
                - Dict with 'real' and 'imag' keys
                - Float (will be treated as real part with 0 imaginary)
            run_on_device_decorator (function): Decorator function for device execution.
        """
        self.run_on_device = run_on_device_decorator
        
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