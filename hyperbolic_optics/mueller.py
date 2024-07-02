import tensorflow as tf
import numpy as np

class Mueller:
    def __init__(self, structure):
        """
        Initialize the Mueller class.

        Args:
            structure: The structure object containing scenario information.
        """
        self.structure = structure
        self.mueller_matrix = None
        self.stokes_parameters = None
        self.incident_stokes = tf.constant([1, 0, 0, 0], dtype=tf.float64)  # Default to unpolarized light
        self.optical_components = []
        self.anisotropic_sample_added = False

    def set_incident_polarization(self, polarization_type, **kwargs):
        """
        Set the incident polarization state.

        Args:
            polarization_type (str): Type of polarization ('linear', 'circular', or 'elliptical')
            **kwargs: Additional arguments depending on the polarization type
                For 'linear': angle (in degrees)
                For 'circular': handedness ('right' or 'left')
                For 'elliptical': alpha (in degrees), ellipticity (between -45 and 45 degrees)
        """
        if polarization_type == 'linear':
            angle = kwargs.get('angle', 0)
            self.incident_stokes = self._linear_polarization(angle)
        elif polarization_type == 'circular':
            handedness = kwargs.get('handedness', 'right')
            self.incident_stokes = self._circular_polarization(handedness)
        elif polarization_type == 'elliptical':
            alpha = kwargs.get('alpha', 0)
            ellipticity = kwargs.get('ellipticity', 0)
            self.incident_stokes = self._elliptical_polarization(alpha, ellipticity)
        else:
            raise ValueError(f"Unsupported polarization type: {polarization_type}")

    def _linear_polarization(self, angle):
        """
        Create a Stokes vector for linear polarization.

        Args:
            angle (float): Angle of linear polarization in degrees (0° is p-polarized, 90° is s-polarized)

        Returns:
            tf.Tensor: Stokes vector for the specified linear polarization
        """
        angle_rad = np.radians(angle)
        return tf.constant([1, np.cos(2*angle_rad), np.sin(2*angle_rad), 0], dtype=tf.float64)

    def _circular_polarization(self, handedness):
        """
        Create a Stokes vector for circular polarization.

        Args:
            handedness (str): 'right' for right-handed, 'left' for left-handed

        Returns:
            tf.Tensor: Stokes vector for the specified circular polarization
        """
        s3 = 1 if handedness == 'right' else -1
        return tf.constant([1, 0, 0, s3], dtype=tf.float64)

    def _elliptical_polarization(self, alpha, ellipticity):
        """
        Create a Stokes vector for elliptical polarization.

        Args:
            alpha (float): Azimuth angle of the ellipse major axis in degrees
            ellipticity (float): Ellipticity angle in degrees (between -45° and 45°)

        Returns:
            tf.Tensor: Stokes vector for the specified elliptical polarization
        """
        alpha_rad = np.radians(alpha)
        ellipticity_rad = np.radians(ellipticity)
        return tf.constant([
            1,
            np.cos(2*ellipticity_rad) * np.cos(2*alpha_rad),
            np.cos(2*ellipticity_rad) * np.sin(2*alpha_rad),
            np.sin(2*ellipticity_rad)
        ], dtype=tf.float64)

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
        Create a Mueller matrix for a quarter-wave plate at a given angle.

        Args:
            angle: Fast axis angle in degrees (float).

        Returns:
            Mueller matrix for the quarter-wave plate (tf.Tensor).
        """
        angle_rad = tf.cast(np.radians(angle), dtype=tf.float64)
        cos_angle = tf.cos(2 * angle_rad)
        sin_angle = tf.sin(2 * angle_rad)

        return tf.convert_to_tensor([
            [1, 0, 0, 0],
            [0, cos_angle**2, cos_angle * sin_angle, -sin_angle],
            [0, cos_angle * sin_angle, sin_angle**2, cos_angle],
            [0, sin_angle, -cos_angle, 0]
        ], dtype=tf.float64)

    def half_wave_plate(self, angle):
        """
        Create a Mueller matrix for a half-wave plate at a given angle.

        Args:
            angle: Fast axis angle in degrees (float).

        Returns:
            Mueller matrix for the half-wave plate (tf.Tensor).
        """
        angle_rad = tf.cast(np.radians(angle), dtype=tf.float64)
        cos_angle = tf.cos(2 * angle_rad)
        sin_angle = tf.sin(2 * angle_rad)

        return tf.convert_to_tensor([
            [1, 0, 0, 0],
            [0, cos_angle**2 - sin_angle**2, 2 * cos_angle * sin_angle, 0],
            [0, 2 * cos_angle * sin_angle, sin_angle**2 - cos_angle**2, 0],
            [0, 0, 0, -1]
        ], dtype=tf.float64)

    def calculate_mueller_matrix(self):
        """
        Calculate the Mueller matrix for the anisotropic sample using the reflection coefficients.
        """
        r_pp = self.structure.r_pp
        r_ps = self.structure.r_ps
        r_sp = self.structure.r_sp
        r_ss = self.structure.r_ss

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

        self.mueller_matrix = tf.cast(a_matrix @ f_matrix @ tf.linalg.inv(a_matrix), dtype=tf.float64)

    def add_optical_component(self, component_type, *args):
        """
        Add an optical component to the system.

        Args:
            component_type: Type of the optical component (str).
            *args: Arguments for the optical component (e.g., angle).
        """
        if component_type == 'linear_polarizer':
            self.optical_components.append(self.linear_polarizer(*args))
        elif component_type == 'anisotropic_sample':
            if self.anisotropic_sample_added:
                raise ValueError("Anisotropic sample has already been added")
            self.calculate_mueller_matrix()
            self.optical_components.append(self.mueller_matrix)
            self.anisotropic_sample_added = True
        elif component_type == 'quarter_wave_plate':
            self.optical_components.append(self.quarter_wave_plate(*args))
        elif component_type == 'half_wave_plate':
            self.optical_components.append(self.half_wave_plate(*args))
        else:
            raise ValueError(f"Unsupported optical component type: {component_type}")

    def calculate_stokes_parameters(self):
        """
        Calculate the Stokes parameters of the system using the set incident polarization and optical components.

        Returns:
            Stokes parameters of the system (tf.Tensor).
        """
        # Start with the incident Stokes vector
        stokes_vector = tf.reshape(self.incident_stokes, [1, 1, 4, 1])

        # Apply optical components in order
        for component in self.optical_components:
            stokes_vector = tf.matmul(component, stokes_vector)

        self.stokes_parameters = stokes_vector[..., 0]
        return self.stokes_parameters

    def get_reflectivity(self):
        """
        Calculate the reflectivity of the system (S0 Stokes parameter).

        Returns:
            Reflectivity of the system (tf.Tensor).
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        return self.stokes_parameters[..., 0]

    def get_degree_of_polarisation(self):
        """
        Calculate the degree of polarization.

        Returns:
            Degree of polarization (tf.Tensor).
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s0, s1, s2, s3 = tf.unstack(self.stokes_parameters, axis=-1)
        return tf.sqrt(s1**2 + s2**2 + s3**2) / s0

    def get_ellipticity(self):
        """
        Calculate the ellipticity of the polarization.

        Returns:
            Ellipticity of the polarization (tf.Tensor).
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s3 = self.stokes_parameters[..., 3]
        s1 = self.stokes_parameters[..., 1]
        s2 = self.stokes_parameters[..., 2]

        return 0.5 * tf.math.atan2(s3, tf.sqrt(s1**2 + s2**2))

    def get_azimuth(self):
        """
        Calculate the azimuth of the polarization.

        Returns:
            Azimuth of the polarization (tf.Tensor).
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s1 = self.stokes_parameters[..., 1]
        s2 = self.stokes_parameters[..., 2]

        return 0.5 * tf.math.atan2(s2, s1)

    def get_stokes_parameters(self):
        """
        Get the Stokes parameters for plotting.

        Returns:
            Dictionary of Stokes parameters (S0, S1, S2, S3).
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        return {
            'S0': self.stokes_parameters[..., 0],
            'S1': self.stokes_parameters[..., 1],
            'S2': self.stokes_parameters[..., 2],
            'S3': self.stokes_parameters[..., 3]
        }

    def get_polarisation_parameters(self):
        """
        Get the polarization parameters for plotting.

        Returns:
            Dictionary of polarization parameters (DOP, Ellipticity, Azimuth).
        """
        return {
            'DOP': self.get_degree_of_polarisation(),
            'Ellipticity': self.get_ellipticity(),
            'Azimuth': self.get_azimuth()
        }

    def get_all_parameters(self):
        """
        Get all Stokes and polarization parameters for comprehensive plotting.

        Returns:
            Dictionary of all parameters (S0, S1, S2, S3, DOP, Ellipticity, Azimuth).
        """
        stokes = self.get_stokes_parameters()
        polarisation = self.get_polarisation_parameters()
        return {**stokes, **polarisation}