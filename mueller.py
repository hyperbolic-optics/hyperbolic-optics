import tensorflow as tf
import numpy as np


class Mueller:
    def __init__(self):
        """
        Initialize the Mueller class.
        """
        self.mueller_matrix = None
        self.optical_components = []

    
    def anisotropic_sample(self, r_pp, r_ps, r_sp, r_ss):
        """
        Create a Mueller matrix for an anisotropic sample using the reflection coefficients.

        Args:
            r_pp, r_ps, r_sp, r_ss: Reflection coefficients (tf.Tensor).

        Returns:
            Mueller matrix for the anisotropic sample (tf.Tensor).
        """
        f_matrix = tf.convert_to_tensor([
            [
                r_pp * tf.math.conj(r_pp),
                r_pp * tf.math.conj(r_ps),
                r_ps * tf.math.conj(r_pp),
                r_ps * tf.math.conj(r_ps)
            ],
            [
                r_pp * tf.math.conj(r_sp),
                r_pp * tf.math.conj(r_ss),
                r_ps * tf.math.conj(r_sp),
                r_ps * tf.math.conj(r_ss)
            ],
            [
                r_sp * tf.math.conj(r_pp),
                r_sp * tf.math.conj(r_ps),
                r_ss * tf.math.conj(r_pp),
                r_ss * tf.math.conj(r_ps)
            ],
            [
                r_sp * tf.math.conj(r_sp),
                r_sp * tf.math.conj(r_ss),
                r_ss * tf.math.conj(r_sp),
                r_ss * tf.math.conj(r_ss)
            ],
        ], dtype=tf.complex128)

        f_matrix = tf.transpose(f_matrix, perm=[2, 3, 0, 1])

        a_matrix = tf.convert_to_tensor([
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 1, 1, 0],
            [0, 1j, -1j, 0]
        ], dtype=tf.complex128)[tf.newaxis, tf.newaxis, ...]

        self.mueller_matrix = True

        return tf.cast(a_matrix @ f_matrix @ tf.linalg.inv(a_matrix), dtype=tf.float64)


    def linear_polarizer(self, angle):
        """
        Create a Mueller matrix for a linear polarizer at a given angle.

        Args:
            angle: Polarizer angle in degrees (float).

        Returns:
            Mueller matrix for the linear polarizer (tf.Tensor).
        """
        angle_rad = tf.cast(np.radians(angle) * 2., dtype=tf.float64)

        cos_angle = tf.cos(angle_rad)
        sin_angle = tf.sin(angle_rad)

        return 0.5 * tf.convert_to_tensor([
            [1 , cos_angle, sin_angle, 0],
            [cos_angle, cos_angle**2., cos_angle * sin_angle, 0],
            [sin_angle, cos_angle * sin_angle, sin_angle**2., 0],
            [0, 0, 0, 0]
        ], dtype=tf.float64)


    def quarter_wave_plate(self, angle):
        """
        TODO: Create a Mueller matrix for a quarter-wave plate at a given angle.

        Args:
            angle: Fast axis angle in degrees (float).

        Returns:
            Mueller matrix for the quarter-wave plate (tf.Tensor).
        """
        pass


    def half_wave_plate(self, angle):
        """
        TODO: Create a Mueller matrix for a half-wave plate at a given angle.

        Args:
            angle: Fast axis angle in degrees (float).

        Returns:
            Mueller matrix for the half-wave plate (tf.Tensor).
        """
        pass


    def add_optical_component(self, component_type, *args):
        """
        Add an optical component to the system.

        Args:
            component_type: Type of the optical component (str).
            angle: Angle of the optical component in degrees (float).
        """
        if component_type == 'linear_polarizer':
            self.optical_components.append(self.linear_polarizer(*args))
        elif component_type == 'anisotropic_sample':
            self.optical_components.append(self.anisotropic_sample(*args))
        elif component_type == 'quarter_wave_plate':
            self.optical_components.append(self.quarter_wave_plate(*args))
        elif component_type == 'half_wave_plate':
            self.optical_components.append(self.half_wave_plate(*args))
        else:
            raise ValueError(f"Unsupported optical component type: {component_type}")

    def calculate_reflectivity(self):
        """
        Calculate the reflectivity of the system with the optical components and sample.

        Returns:
            Reflectivity of the system (tf.Tensor).
        """
        # Assuming unpolarized incident light (Stokes vector: [1, 0, 0, 0])
        incident_stokes = tf.constant([1, 0, 0, 0], dtype=tf.float64)
        incident_stokes = tf.reshape(incident_stokes, [1, 1, 4, 1])

        # Apply the optical components to the incident light
        light = incident_stokes
        for component in reversed(self.optical_components):
            light = component @ light

        if self.mueller_matrix is False:
            raise ValueError("Anisotropic sample Mueller matrix not set.")

        # Extract the reflectivity (first element of the Stokes vector)
        reflectivity = light[..., 0, 0]

        return reflectivity