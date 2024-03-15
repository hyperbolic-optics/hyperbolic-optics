"""
Material Parameters

This module defines the material parameters for various anisotropic materials, such as Quartz, Sapphire, and Calcite.
It includes permittivity and permeability tensors, recommended frequencies, and other relevant parameters.

The module also includes ambient incident and exit mediums.

Future plan is to integrate this better with Layers.py, to separate the parameters.
"""

import tensorflow as tf
from scipy import constants

from device_config import run_on_device


class AnisotropicMaterial:
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


class Quartz(AnisotropicMaterial):
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


class Calcite(AnisotropicMaterial):
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

    def __init__(self, freq_min=1300.0, freq_max=1600.0):
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


class Sapphire(AnisotropicMaterial):
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


class Antiferromagnet(object):
    def __init__(self, frequency_length):
        self.frequency_length = frequency_length
        self.frequency = tf.cast(
            tf.linspace(52.0, 54.0, self.frequency_length), dtype=tf.complex128
        )
        self.name = "FeF2"

        self.gamma = tf.cast(1.05, dtype=tf.complex128)  # cm^-1 / T
        self.B0 = tf.cast(0.3, dtype=tf.complex128)  # T
        self.Ba = tf.cast(19.745, dtype=tf.complex128)  # T
        self.Be = tf.cast(53.313, dtype=tf.complex128)  # T
        self.magnetisation = tf.cast(445633.84, dtype=tf.complex128)  # A/cm

        self.resonant_frequency_squared = self.gamma**2.0 * (
            2.0 * self.Ba * self.Be + self.Ba**2.0
        )
        self.damping_parameter = 1.27e-4 * tf.sqrt(self.resonant_frequency_squared)


    def fetch_epsilon_mu(self):
        X = 1.0 / (
            self.resonant_frequency_squared
            - (
                self.frequency
                + self.B0 * self.gamma
                + tf.constant(1j, dtype=tf.complex128) * self.damping_parameter
            )
            ** 2.0
        )
        Y = 1.0 / (
            self.resonant_frequency_squared
            - (
                self.frequency
                - self.B0 * self.gamma
                + tf.constant(1j, dtype=tf.complex128) * self.damping_parameter
            )
            ** 2.0
        )

        mu_3 = 1.0 + constants.mu_0 * self.gamma**2.0 * self.Ba * self.magnetisation * (
            X + Y
        )
        mu_t = (
            constants.mu_0 * self.gamma**2.0 * self.Ba * self.magnetisation * (X - Y)
        )

        magnet_permittivity = tf.constant(5.5 + 0.0j, dtype=tf.complex128)

        return mu_3, mu_t, magnet_permittivity

    def magnet_tensors(self):
        mu_3, mu_t, magnet_permittivity = self.fetch_epsilon_mu()

        permeability_tensor = tf.stack(
            [
                [
                    mu_3,
                    tf.zeros_like(mu_3),
                    -tf.constant(1.0j, dtype=tf.complex128) * mu_t,
                ],
                [tf.zeros_like(mu_3), tf.ones_like(mu_3), tf.zeros_like(mu_3)],
                [
                    tf.constant(1.0j, dtype=tf.complex128) * mu_t,
                    tf.zeros_like(mu_3),
                    mu_3,
                ],
            ],
            axis=-1,
        )
        permeability_tensor = tf.transpose(permeability_tensor, perm=[1, 0, 2])

        permittivity_tensor = tf.stack(
            [
                [magnet_permittivity, 0.0, 0.0],
                [0.0, magnet_permittivity, 0.0],
                [0.0, 0.0, magnet_permittivity],
            ],
            axis=-1,
        ) * tf.ones_like(permeability_tensor)

        return permeability_tensor, permittivity_tensor


class AmbientIncidentMedium(object):
    def __init__(self, permittivity, kx, run_on_device_decorator = run_on_device):
        self.permittivity = permittivity
        self.theta = tf.cast(tf.math.asin(kx / (permittivity**0.5)), dtype= tf.float64)
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def construct_tensor(self):
        n = tf.sqrt(self.permittivity)
        cos_theta = tf.cos(self.theta)
        n_cos_theta = n * cos_theta

        # Combine updates into a single tensor with shape [180, 4, 4]
        element1 = tf.stack(
            [
                tf.zeros_like(self.theta),
                tf.ones_like(self.theta),
                -1.0 / n_cos_theta,
                tf.zeros_like(self.theta),
            ],
            axis=-1,
        )
        element2 = tf.stack(
            [
                tf.zeros_like(self.theta),
                tf.ones_like(self.theta),
                1.0 / n_cos_theta,
                tf.zeros_like(self.theta),
            ],
            axis=-1,
        )
        element3 = tf.stack(
            [
                1.0 / cos_theta,
                tf.zeros_like(self.theta),
                tf.zeros_like(self.theta),
                1.0 / n * tf.ones_like(self.theta),
            ],
            axis=-1,
        )
        element4 = tf.stack(
            [
                -1.0 / cos_theta,
                tf.zeros_like(self.theta),
                tf.zeros_like(self.theta),
                1.0 / n * tf.ones_like(self.theta),
            ],
            axis=-1,
        )

        matrix = tf.stack([element1, element2, element3, element4], axis=1)

        return 0.5 * tf.cast(matrix, dtype=tf.complex128)

    @run_on_device
    def construct_tensor_singular(self):
        n = tf.sqrt(self.permittivity)
        cos_theta = tf.cos(self.theta)
        n_cos_theta = n * cos_theta

        # Combine updates into a single tensor with shape [180, 4, 4]
        element1 = tf.stack([0.0, 1.0, -1.0 / n_cos_theta, 0.0])
        element2 = tf.stack([0.0, 1.0, 1.0 / n_cos_theta, 0.0])
        element3 = tf.stack([1.0 / cos_theta, 0.0, 0.0, 1.0 / n])
        element4 = tf.stack([-1.0 / cos_theta, 0.0, 0.0, 1.0 / n])

        matrix = tf.stack([element1, element2, element3, element4], axis=0)

        return 0.5 * tf.cast(matrix, dtype=tf.complex128)


class AmbientExitMedium(object):
    def __init__(self, incident_angle, permittivity_incident, permittivity_exit, run_on_device_decorator = run_on_device):
        self.theta_incident = incident_angle
        self.N_exit = tf.sqrt(permittivity_exit)
        self.N_incident = tf.sqrt(permittivity_incident)
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def construct_tensor(self):
        sin_theta_incident = tf.sin(self.theta_incident)
        expr_inside_sqrt = 1.0 - ((self.N_incident / self.N_exit) * sin_theta_incident) ** 2.
        expr_inside_sqrt_complex = tf.cast(expr_inside_sqrt, dtype=tf.complex128)
        cos_theta_f = tf.sqrt(expr_inside_sqrt_complex)
        self.N_exit = tf.cast(self.N_exit, dtype=tf.complex128)
        Nf_cos_theta_f = self.N_exit * cos_theta_f

        element1 = tf.stack([
            tf.zeros_like(cos_theta_f),
            tf.zeros_like(cos_theta_f), 
            cos_theta_f, 
            -cos_theta_f], 
            axis=-1)
        
        element2 = tf.stack([
            tf.ones_like(cos_theta_f), 
            tf.ones_like(cos_theta_f), 
            tf.zeros_like(cos_theta_f), 
            tf.zeros_like(cos_theta_f)], 
            axis=-1)
        
        element3 = tf.stack([
            -Nf_cos_theta_f, 
            Nf_cos_theta_f, 
            tf.zeros_like(cos_theta_f), 
            tf.zeros_like(cos_theta_f)], 
            axis=-1)
        
        element4 = tf.stack([
            tf.zeros_like(cos_theta_f), 
            tf.zeros_like(cos_theta_f), 
            self.N_exit * tf.ones_like(cos_theta_f), 
            self.N_exit * tf.ones_like(cos_theta_f)], 
            axis=-1)

        # Stack the elements
        matrix = tf.stack([element1, element2, element3, element4], axis=1)


        return tf.cast(matrix, dtype=tf.complex128)



class Air(object):
    def __init__(self, run_on_device_decorator = run_on_device):
        self.run_on_device = run_on_device_decorator
        pass

    @run_on_device
    def construct_tensor_singular(self):
        tensor = tf.constant(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.complex128
        )
        return tensor
