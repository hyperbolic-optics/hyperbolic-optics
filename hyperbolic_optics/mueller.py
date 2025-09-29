"""Mueller matrix calculus for polarization analysis.

This module provides the Mueller class for analyzing polarization transformations
in optical systems. Mueller matrices are 4×4 real matrices that describe how
Stokes parameters (polarization states) transform upon interaction with optical
elements.

The Mueller formalism handles:

- Arbitrary incident polarization states (linear, circular, elliptical)
- Polarization-dependent reflection from anisotropic samples
- Ideal optical components (polarizers, wave plates)
- Sequential component combinations

Key relationships:

- S_out = M · S_in (Stokes vector transformation)
- M = A · F · A⁻¹ (Mueller from Jones matrix F)

Reference:
    Chipman, Lam & Young, "Polarized Light and Optical Systems" (2018)
"""

from typing import Any

import numpy as np

from .structure import Structure


class Mueller:
    """Mueller matrix analyzer for polarization calculations.

    The Mueller class provides tools for analyzing polarization transformations
    in optical systems using the Mueller matrix formalism. It handles arbitrary
    incident polarization states and sequential optical components.

    Attributes:
        structure: The Structure object containing reflection coefficients
        mueller_matrix: 4×4 Mueller matrix for the sample
        stokes_parameters: Output Stokes parameters
        incident_stokes: Incident polarization state [S0, S1, S2, S3]
        optical_components: List of Mueller matrices for optical elements

    Note:
        Mueller matrices are real-valued 4×4 matrices that transform Stokes
        parameters: S_out = M · S_in

    Examples:
        Analyzing p-polarized reflection:

        >>> structure = Structure()
        >>> structure.execute(payload)
        >>> mueller = Mueller(structure)
        >>> mueller.set_incident_polarization('linear', angle=0)
        >>> mueller.add_optical_component('anisotropic_sample')
        >>> params = mueller.get_all_parameters()
        >>> print(f"Reflectance: {params['S0']:.4f}")
        >>> print(f"DOP: {params['DOP']:.4f}")

        Crossed polarizer configuration:

        >>> mueller = Mueller(structure)
        >>> mueller.set_incident_polarization('linear', angle=0)
        >>> mueller.add_optical_component('linear_polarizer', 0)
        >>> mueller.add_optical_component('anisotropic_sample')
        >>> mueller.add_optical_component('linear_polarizer', 90)
        >>> extinction_ratio = mueller.get_reflectivity()

        Converting linear to circular polarization:

        >>> mueller = Mueller(structure)
        >>> mueller.set_incident_polarization('linear', angle=45)
        >>> mueller.add_optical_component('quarter_wave_plate', 45)
        >>> mueller.add_optical_component('anisotropic_sample')
        >>> stokes = mueller.get_stokes_parameters()
        >>> circularity = abs(stokes['S3'] / stokes['S0'])
        >>> print(f"Circular component: {circularity:.2%}")
    """

    def __init__(self, structure: Structure, debug: bool = False) -> None:
        """Initialize Mueller matrix analyzer for polarization calculations.

        Args:
            structure: The Structure object containing reflection coefficients
            debug: Enable detailed debug output for troubleshooting

        Example:
            >>> structure = Structure()
            >>> structure.execute(payload)
            >>> mueller = Mueller(structure)
            >>> mueller.set_incident_polarization('linear', angle=45)
            >>> mueller.add_optical_component('anisotropic_sample')
        """
        self.structure = structure
        self.mueller_matrix = None
        self.stokes_parameters = None
        self.incident_stokes = np.array(
            [1, 0, 0, 0], dtype=np.float64
        )  # Default to unpolarized light
        self.optical_components = []
        self.anisotropic_sample_added = False

    def set_incident_polarization(self, polarization_type: str, **kwargs: Any) -> None:
        """Set the incident polarization state using Stokes parameters.

        Args:
            polarization_type: Type of polarization ('linear', 'circular', 'elliptical')
            **kwargs: Additional arguments depending on type:
                - linear: angle (float) - polarization angle in degrees
                - circular: handedness (str) - 'right' or 'left'
                - elliptical: alpha (float) - azimuth in degrees,
                            ellipticity (float) - ellipticity angle in degrees

        Raises:
            ValueError: If polarization_type is not recognized

        Example:
            >>> mueller.set_incident_polarization('linear', angle=0)  # p-polarized
            >>> mueller.set_incident_polarization('circular', handedness='right')
            >>> mueller.set_incident_polarization('elliptical', alpha=30, ellipticity=20)
        """
        if polarization_type == "linear":
            angle = kwargs.get("angle", 0)
            self.incident_stokes = self._linear_polarization(angle)
        elif polarization_type == "circular":
            handedness = kwargs.get("handedness", "right")
            self.incident_stokes = self._circular_polarization(handedness)
        elif polarization_type == "elliptical":
            alpha = kwargs.get("alpha", 0)
            ellipticity = kwargs.get("ellipticity", 0)
            self.incident_stokes = self._elliptical_polarization(alpha, ellipticity)
        else:
            raise ValueError(f"Unsupported polarization type: {polarization_type}")

    def _linear_polarization(self, angle: float) -> np.ndarray:
        """Create Stokes vector for linear polarization at specified angle.

        Args:
            angle: Polarization angle in degrees (0° = p-pol, 90° = s-pol)

        Returns:
            Stokes vector [S0, S1, S2, S3] = [1, cos(2θ), sin(2θ), 0]

        Note:
            Linear polarization has S3 = 0 (no circular component).
        """
        angle_rad = np.radians(angle)
        return np.array([1, np.cos(2 * angle_rad), np.sin(2 * angle_rad), 0], dtype=np.float64)

    def _circular_polarization(self, handedness: str) -> np.ndarray:
        """Create Stokes vector for circular polarization.

        Args:
            handedness: 'right' for right-handed or 'left' for left-handed

        Returns:
            Stokes vector [S0, S1, S2, S3] = [1, 0, 0, ±1]

        Note:
            Circular polarization has S1 = S2 = 0 and S3 = ±1.
        """
        s3 = 1 if handedness == "right" else -1
        return np.array([1, 0, 0, s3], dtype=np.float64)

    def _elliptical_polarization(self, alpha: float, ellipticity: float) -> np.ndarray:
        """Create Stokes vector for elliptical polarization.

        Args:
            alpha: Azimuth angle of ellipse major axis in degrees
            ellipticity: Ellipticity angle in degrees (-45° to 45°)

        Returns:
            Stokes vector [S0, S1, S2, S3]

        Note:
            Ellipticity = 0° gives linear, ±45° gives circular polarization.
        """
        alpha_rad = np.radians(alpha)
        ellipticity_rad = np.radians(ellipticity)
        return np.array(
            [
                1,
                np.cos(2 * ellipticity_rad) * np.cos(2 * alpha_rad),
                np.cos(2 * ellipticity_rad) * np.sin(2 * alpha_rad),
                np.sin(2 * ellipticity_rad),
            ],
            dtype=np.float64,
        )

    def linear_polarizer(self, angle: float) -> np.ndarray:
        """Create Mueller matrix for ideal linear polarizer.

        Args:
            angle: Polarizer transmission axis angle in degrees

        Returns:
            4×4 Mueller matrix for linear polarizer

        Note:
            Ideal polarizer fully transmits light along transmission axis
            and fully blocks light perpendicular to it.
        """
        angle_rad = np.float64(np.radians(angle) * 2.0)

        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        return 0.5 * np.array(
            [
                [1, cos_angle, sin_angle, 0],
                [cos_angle, cos_angle**2.0, cos_angle * sin_angle, 0],
                [sin_angle, cos_angle * sin_angle, sin_angle**2.0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

    def quarter_wave_plate(self, angle: float) -> np.ndarray:
        """Create Mueller matrix for quarter-wave plate (QWP).

        Args:
            angle: Fast axis orientation angle in degrees

        Returns:
            4×4 Mueller matrix for QWP

        Note:
            QWP introduces π/2 phase shift between fast and slow axes.
            Can convert linear to circular polarization and vice versa.
        """
        angle_rad = np.float64(np.radians(angle))
        cos_angle = np.cos(2 * angle_rad)
        sin_angle = np.sin(2 * angle_rad)

        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos_angle**2, cos_angle * sin_angle, -sin_angle],
                [0, cos_angle * sin_angle, sin_angle**2, cos_angle],
                [0, sin_angle, -cos_angle, 0],
            ],
            dtype=np.float64,
        )

    def half_wave_plate(self, angle: float) -> np.ndarray:
        """Create Mueller matrix for half-wave plate (HWP).

        Args:
            angle: Fast axis orientation angle in degrees

        Returns:
            4×4 Mueller matrix for HWP

        Note:
            HWP introduces π phase shift, effectively rotating linear
            polarization by 2θ where θ is the plate angle.
        """
        angle_rad = np.float64(np.radians(angle))
        cos_angle = np.cos(2 * angle_rad)
        sin_angle = np.sin(2 * angle_rad)

        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos_angle**2 - sin_angle**2, 2 * cos_angle * sin_angle, 0],
                [0, 2 * cos_angle * sin_angle, sin_angle**2 - cos_angle**2, 0],
                [0, 0, 0, -1],
            ],
            dtype=np.float64,
        )

    def calculate_mueller_matrix(self) -> None:
        """Calculate Mueller matrix from reflection coefficients.

        Converts the complex 2×2 Jones matrix to a real 4×4 Mueller matrix
        using the transformation: M = A·F·A⁻¹ where F is formed from
        r_pp, r_ss, r_ps, r_sp.

        Note:
            The Mueller matrix fully describes how the sample transforms
            arbitrary incident polarization states to reflected states.
        """
        r_pp = self.structure.r_pp
        r_ps = self.structure.r_ps
        r_sp = self.structure.r_sp
        r_ss = self.structure.r_ss

        f_matrix = np.array(
            [
                [
                    r_pp * np.conj(r_pp),
                    r_pp * np.conj(r_ps),
                    r_ps * np.conj(r_pp),
                    r_ps * np.conj(r_ps),
                ],
                [
                    r_pp * np.conj(r_sp),
                    r_pp * np.conj(r_ss),
                    r_ps * np.conj(r_sp),
                    r_ps * np.conj(r_ss),
                ],
                [
                    r_sp * np.conj(r_pp),
                    r_sp * np.conj(r_ps),
                    r_ss * np.conj(r_pp),
                    r_ss * np.conj(r_ps),
                ],
                [
                    r_sp * np.conj(r_sp),
                    r_sp * np.conj(r_ss),
                    r_ss * np.conj(r_sp),
                    r_ss * np.conj(r_ss),
                ],
            ],
            dtype=np.complex128,
        )

        # Handle different scenario types
        if self.structure.scenario.type == "Simple":
            # For Simple scenario, f_matrix is just [4, 4], no need to transpose
            pass
        else:
            # For other scenarios, transpose as before
            f_matrix = np.transpose(f_matrix, axes=[2, 3, 0, 1])

        a_matrix = np.array(
            [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]],
            dtype=np.complex128,
        )

        # Add batch dimensions if needed
        if self.structure.scenario.type == "Simple":
            # For Simple scenario, just compute matrix multiplication directly
            self.mueller_matrix = (a_matrix @ f_matrix @ np.linalg.inv(a_matrix)).astype(np.float64)
        else:
            # For other scenarios, add batch dimensions
            a_matrix = a_matrix[np.newaxis, np.newaxis, ...]
            self.mueller_matrix = (a_matrix @ f_matrix @ np.linalg.inv(a_matrix)).astype(np.float64)

    def add_optical_component(self, component_type: str, *args: Any) -> None:
        """Add optical component to the propagation path.

        Args:
            component_type: Type of component ('anisotropic_sample',
                        'linear_polarizer', 'quarter_wave_plate',
                        'half_wave_plate')
            *args: Component-specific parameters (e.g., angle for polarizers)

        Raises:
            ValueError: If component type is not recognized or anisotropic
                    sample is added more than once

        Example:
            >>> mueller.add_optical_component('linear_polarizer', 0)
            >>> mueller.add_optical_component('anisotropic_sample')
            >>> mueller.add_optical_component('linear_polarizer', 90)
        """
        if component_type == "linear_polarizer":
            self.optical_components.append(self.linear_polarizer(*args))
        elif component_type == "anisotropic_sample":
            if self.anisotropic_sample_added:
                raise ValueError("Anisotropic sample has already been added")
            self.calculate_mueller_matrix()
            self.optical_components.append(self.mueller_matrix)
            self.anisotropic_sample_added = True
        elif component_type == "quarter_wave_plate":
            self.optical_components.append(self.quarter_wave_plate(*args))
        elif component_type == "half_wave_plate":
            self.optical_components.append(self.half_wave_plate(*args))
        else:
            raise ValueError(f"Unsupported optical component type: {component_type}")

    def calculate_stokes_parameters(self) -> np.ndarray:
        """Calculate output Stokes parameters after all optical components.

        Propagates the incident Stokes vector through all added optical
        components by sequential Mueller matrix multiplication.

        Returns:
            Output Stokes vector [S0, S1, S2, S3] with shape matching scenario

        Note:
            S0 = total intensity (reflectance)
            S1 = horizontal vs vertical linear polarization
            S2 = +45° vs -45° linear polarization
            S3 = right vs left circular polarization
        """
        if self.structure.scenario.type == "Simple":
            # For Simple scenario, start with just the incident vector [4,]
            stokes_vector = self.incident_stokes.reshape([4, 1])
        else:
            # For other scenarios, add batch dimensions
            stokes_vector = self.incident_stokes.reshape([1, 1, 4, 1])

        for i, component in enumerate(self.optical_components):
            if self.structure.scenario.type == "Simple":
                # For Simple scenario, component should be [4, 4]
                stokes_vector = component @ stokes_vector
            else:
                # For other scenarios, component has batch dimensions
                stokes_vector = component @ stokes_vector

        if self.structure.scenario.type == "Simple":
            # For Simple scenario, remove the last dimension [4, 1] -> [4]
            self.stokes_parameters = stokes_vector[:, 0]
        else:
            # For other scenarios, remove the last dimension [..., 4, 1] -> [..., 4]
            self.stokes_parameters = stokes_vector[..., 0]

        return self.stokes_parameters

    def get_reflectivity(self) -> np.ndarray:
        """Get total reflectance (S0 Stokes parameter).

        Returns:
            Reflectance array with shape matching scenario type

        Note:
            Automatically calculates Stokes parameters if not already computed.
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        return self.stokes_parameters[..., 0]

    def get_degree_of_polarisation(self) -> np.ndarray:
        """Calculate degree of polarization (DOP).

        Returns:
            DOP array with values clipped to [0, 1]

        Note:
            DOP = √(S1² + S2² + S3²) / S0
            DOP = 1: fully polarized
            DOP = 0: unpolarized (random)
            0 < DOP < 1: partially polarized
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s0 = self.stokes_parameters[..., 0]
        s1 = self.stokes_parameters[..., 1]
        s2 = self.stokes_parameters[..., 2]
        s3 = self.stokes_parameters[..., 3]

        # Avoid division by zero
        epsilon = 1e-10
        s0_safe = np.maximum(s0, epsilon)

        dop = np.sqrt(s1**2 + s2**2 + s3**2) / s0_safe

        # Clip to ensure DOP is always between 0 and 1
        dop = np.clip(dop, 0.0, 1.0)

        return dop

    def get_ellipticity(self) -> np.ndarray:
        """Calculate ellipticity angle of polarization ellipse.

        Returns:
            Ellipticity angle in radians (-π/4 to π/4)

        Note:
            Ellipticity = 0: linear polarization
            Ellipticity = ±π/4: circular polarization
            Intermediate values: elliptical polarization
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s3 = self.stokes_parameters[..., 3]
        s1 = self.stokes_parameters[..., 1]
        s2 = self.stokes_parameters[..., 2]

        return 0.5 * np.arctan2(s3, np.sqrt(s1**2 + s2**2))

    def get_azimuth(self) -> np.ndarray:
        """Calculate azimuth angle of polarization ellipse major axis.

        Returns:
            Azimuth angle in radians

        Note:
            Defines orientation of the polarization ellipse in the plane
            perpendicular to propagation direction.
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        s1 = self.stokes_parameters[..., 1]
        s2 = self.stokes_parameters[..., 2]

        return 0.5 * np.arctan2(s2, s1)

    def get_stokes_parameters(self) -> dict[str, np.ndarray]:
        """Get all four Stokes parameters.

        Returns:
            Dictionary with keys 'S0', 'S1', 'S2', 'S3' containing parameter arrays

        Note:
            Automatically calculates if not already computed.
        """
        if self.stokes_parameters is None:
            self.calculate_stokes_parameters()

        return {
            "S0": self.stokes_parameters[..., 0],
            "S1": self.stokes_parameters[..., 1],
            "S2": self.stokes_parameters[..., 2],
            "S3": self.stokes_parameters[..., 3],
        }

    def get_polarisation_parameters(self) -> dict[str, np.ndarray]:
        """Get derived polarization properties.

        Returns:
            Dictionary with 'DOP', 'Ellipticity', 'Azimuth' arrays
        """
        return {
            "DOP": self.get_degree_of_polarisation(),
            "Ellipticity": self.get_ellipticity(),
            "Azimuth": self.get_azimuth(),
        }

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """Get comprehensive set of all Stokes and polarization parameters.

        Returns:
            Dictionary containing S0, S1, S2, S3, DOP, Ellipticity, Azimuth

        Example:
            >>> params = mueller.get_all_parameters()
            >>> print(f"Reflectance: {params['S0'].mean():.3f}")
            >>> print(f"Average DOP: {params['DOP'].mean():.3f}")
        """
        stokes = self.get_stokes_parameters()
        polarisation = self.get_polarisation_parameters()
        all_params = {**stokes, **polarisation}

        return all_params

    def reset(self) -> None:
        """Reset Mueller object to initial state.

        Clears all calculated matrices, Stokes parameters, optical components,
        and resets incident polarization to unpolarized state.

        Note:
            Call this before setting up a new calculation sequence.
        """
        self.mueller_matrix = None
        self.stokes_parameters = None
        self.incident_stokes = np.array([1, 0, 0, 0], dtype=np.float64)
        self.optical_components = []
        self.anisotropic_sample_added = False
