"""Wave propagation and electromagnetic field calculations.

This module implements the core 4×4 transfer matrix formalism for
wave propagation in anisotropic media. The key steps are:

1. Construct Berreman (delta) matrix from material tensors
2. Solve eigenvalue problem for propagation modes
3. Sort modes by propagation direction (forward/backward)
4. Calculate electric and magnetic field components
5. Compute Poynting vectors (energy flow)
6. Classify modes by polarization (p vs s)
7. Construct transfer matrix with phase propagation

The Berreman matrix Δ satisfies: dF/dz = iΔF where F = [Ex, Ey, Hx, Hy]ᵀ

Key physical insights:

- Anisotropic media support 4 partial waves (2 forward, 2 backward)
- Each mode has mixed s and p polarization character
- Energy flow (Poynting vector) may differ from phase velocity
- Evanescent modes have complex kz with Im(kz) > 0

References:

- Berreman, J. Opt. Soc. Am. 62, 502-510 (1972)
- Yeh, JOSA 69, 742-756 (1979)
- Passler & Paarmann, JOSA B 34, 2128-2139 (2017)
"""

import numpy as np
from scipy.linalg import expm


class WaveProfile:
    """Class representing the wave profile."""

    def __init__(self, profile: dict[str, dict[str, np.ndarray]]) -> None:
        """Initialize wave profile containing field components and propagation constants.

        Args:
            profile: Nested dictionary with 'transmitted' and 'reflected' keys,
                    each containing field components (Ex, Ey, Ez, Hx, Hy, Hz),
                    Poynting vectors (Px, Py, Pz), and propagation constants (k_z)

        Note:
            Stores electromagnetic field distributions and energy flow for
            the four partial waves (two transmitted, two reflected) in each layer.
        """
        self.transmitted_Ex = profile["transmitted"]["Ex"]
        self.transmitted_Ey = profile["transmitted"]["Ey"]
        self.transmitted_Ez = profile["transmitted"]["Ez"]
        self.transmitted_Hx = profile["transmitted"]["Hx"]
        self.transmitted_Hy = profile["transmitted"]["Hy"]
        self.transmitted_Hz = profile["transmitted"]["Hz"]
        self.transmitted_Px = profile["transmitted"]["Px_physical"]
        self.transmitted_Py = profile["transmitted"]["Py_physical"]
        self.transmitted_Pz = profile["transmitted"]["Pz_physical"]
        self.transmitted_k_z = profile["transmitted"]["propagation"]

        self.reflected_Ex = profile["reflected"]["Ex"]
        self.reflected_Ey = profile["reflected"]["Ey"]
        self.reflected_Ez = profile["reflected"]["Ez"]
        self.reflected_Hx = profile["reflected"]["Hx"]
        self.reflected_Hy = profile["reflected"]["Hy"]
        self.reflected_Hz = profile["reflected"]["Hz"]
        self.reflected_Px = profile["reflected"]["Px_physical"]
        self.reflected_Py = profile["reflected"]["Py_physical"]
        self.reflected_Pz = profile["reflected"]["Pz_physical"]
        self.reflected_k_z = profile["reflected"]["propagation"]


class Wave:
    """Class representing the four partial waves in a layer of the structure."""

    def __init__(
        self,
        kx: np.ndarray,
        eps_tensor: np.ndarray,
        mu_tensor: np.ndarray,
        mode: str,
        k_0: np.ndarray | None = None,
        thickness: float | None = None,
        semi_infinite: bool = False,
        magnet: bool = False,
    ) -> None:
        """Initialize wave calculation for a layer.

        Args:
            kx: Parallel wavevector component(s)
            eps_tensor: Permittivity tensor [3, 3] or [N, 3, 3]
            mu_tensor: Permeability tensor [3, 3] or [N, 3, 3]
            mode: Calculation mode ('Incident', 'Azimuthal', 'Dispersion', 'Simple',
                'airgap', 'simple_airgap', 'azimuthal_airgap', 'simple_scalar_airgap')
            k_0: Free-space wavenumber (required for finite layers)
            thickness: Layer thickness in cm (None for semi-infinite)
            semi_infinite: Whether layer is semi-infinite
            magnet: Whether material is magnetic (currently unused)

        Note:
            The mode parameter determines array broadcasting patterns for
            efficient batch processing of different scenario types.
        """
        self.k_x = kx.astype(np.complex128) if hasattr(kx, "astype") else np.complex128(kx)
        self.eps_tensor = eps_tensor  # Now pre-shaped from materials
        self.mu_tensor = mu_tensor  # Now pre-shaped from materials

        self.mode = mode
        self.batch_size = None

        self.k_0 = k_0
        self.thickness = thickness
        self.semi_infinite = semi_infinite
        self.magnet = magnet

        self.eigenvalues = None
        self.eigenvectors = None
        self.berreman_matrix = None

        # CHANGED: Ensure tensors have proper shapes and determine batch dimensions
        self._setup_tensor_shapes()

    def _setup_tensor_shapes(self) -> None:
        """Setup and validate tensor shapes based on mode.

        Ensures kx, eps_tensor, and mu_tensor have compatible shapes for
        the specified calculation mode and determines batch dimensions.

        Raises:
            NotImplementedError: If mode is not recognized
        """
        # Ensure k_x is complex
        self.k_x = (
            self.k_x.astype(np.complex128)
            if hasattr(self.k_x, "astype")
            else np.complex128(self.k_x)
        )

        # Ensure tensors are complex
        self.eps_tensor = self.eps_tensor.astype(np.complex128)
        self.mu_tensor = self.mu_tensor.astype(np.complex128)

        # Standardize tensor shapes based on mode
        if self.mode == "Simple":
            # Simple: scalar kx, [3,3] tensors
            if len(self.eps_tensor.shape) > 2:
                self.eps_tensor = np.squeeze(self.eps_tensor)
                self.mu_tensor = np.squeeze(self.mu_tensor)
        # For all other modes, keep natural shapes
        # kx reshaping will happen in delta_matrix_calc if needed

    def _get_tensor_shapes_for_mode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get properly shaped tensors for current calculation mode.

        Returns:
            Tuple of (kx, eps_tensor, mu_tensor) - now just returns as-is
            since shapes are already standardized in _setup_tensor_shapes
        """
        # After _setup_tensor_shapes, all arrays have consistent shapes for their mode
        return self.k_x, self.eps_tensor, self.mu_tensor

    def _get_poynting_tensor_shapes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get tensor shapes for Poynting vector calculation.

        Returns:
            Tuple of (kx, eps_tensor, mu_tensor) with shapes matching field arrays

        Note:
            Fields have shape [..., 4, 2] (4 components, 2 modes).
            Need to broadcast tensors to match [..., 3, 3] for Ez/Hz calculation.
        """
        # Just return as-is and let numpy broadcasting handle it
        # The calculation of Ez/Hz uses [..., 2, 2] indexing which broadcasts naturally
        return self.k_x, self.eps_tensor, self.mu_tensor

    def _get_matrix_calculation_shapes(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get tensor shapes for transfer matrix construction.

        Args:
            eigenvalues: Eigenvalues from Berreman matrix
            eigenvectors: Eigenvectors from Berreman matrix

        Returns:
            Tuple of (k_0, eigenvalues_diag, eigenvectors) with proper shapes
        """
        # Create diagonal matrix from eigenvalues - vectorized
        n = eigenvalues.shape[-1]
        eigenvalues_diag = np.zeros(eigenvalues.shape + (n,), dtype=eigenvalues.dtype)
        diagonal_indices = np.arange(n)
        eigenvalues_diag[..., diagonal_indices, diagonal_indices] = eigenvalues

        # Return k_0, eigenvalues_diag, eigenvectors - shapes already compatible
        return self.k_0, eigenvalues_diag, eigenvectors

    def mode_reshaping(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape tensors based on calculation mode.

        Returns:
            Tuple of (kx, eps_tensor, mu_tensor) with appropriate shapes

        Note:
            Wrapper around _get_tensor_shapes_for_mode for compatibility.
        """
        return self._get_tensor_shapes_for_mode()

    def delta_matrix_calc(self) -> None:
        """Construct 4×4 Berreman transfer matrix for anisotropic layer.

        Builds the Berreman matrix (also called delta matrix) that describes
        wave propagation in an anisotropic medium. The matrix couples electric
        and magnetic field components.

        Note:
            The Berreman formalism reduces Maxwell's equations to an eigenvalue
            problem: dF/dz = iΔF where F = [Ex, Ey, Hx, Hy]ᵀ

            Reference: N.C. Passler & A. Paarmann, JOSA B 34, 2128 (2017)
        """
        k_x, eps_tensor, mu_tensor = self.mode_reshaping()

        # Reshape kx for proper broadcasting with tensors
        if self.mode in ["Incident", "Dispersion"]:
            # Need kx as [N_angles, 1] to broadcast with [N_freq/N_azim, 3, 3] tensors
            if k_x.ndim == 1:
                k_x = k_x[:, np.newaxis]
        elif self.mode == "FullSweep":
            # Need kx as [N_angles, 1, 1] to broadcast with [N_angles, N_azim, N_freq, 3, 3] tensors
            if k_x.ndim == 1:
                k_x = k_x[:, np.newaxis, np.newaxis]

        # Extract relevant tensor components
        eps_20, eps_21, eps_22 = (
            eps_tensor[..., 2, 0],
            eps_tensor[..., 2, 1],
            eps_tensor[..., 2, 2],
        )
        eps_10, eps_11, eps_12 = (
            eps_tensor[..., 1, 0],
            eps_tensor[..., 1, 1],
            eps_tensor[..., 1, 2],
        )
        eps_00, eps_01, eps_02 = (
            eps_tensor[..., 0, 0],
            eps_tensor[..., 0, 1],
            eps_tensor[..., 0, 2],
        )
        mu_12, mu_22 = mu_tensor[..., 1, 2], mu_tensor[..., 2, 2]
        mu_10, mu_20 = mu_tensor[..., 1, 0], mu_tensor[..., 2, 0]
        mu_11, mu_21 = mu_tensor[..., 1, 1], mu_tensor[..., 2, 1]
        mu_02 = mu_tensor[..., 0, 2]
        mu_00, mu_01 = mu_tensor[..., 0, 0], mu_tensor[..., 0, 1]

        # Precompute common terms
        k_x_sq = k_x**2
        eps_22_inv = 1.0 / eps_22
        mu_22_inv = 1.0 / mu_22
        ones_like_k_x = np.ones_like(k_x)

        # Construct the matrix elements
        m00 = -k_x * eps_20 * eps_22_inv
        m01 = k_x * (mu_12 * mu_22_inv - eps_21 * eps_22_inv)
        m02 = (mu_10 - mu_12 * mu_20 * mu_22_inv) * ones_like_k_x
        m03 = mu_11 - mu_12 * mu_21 * mu_22_inv - k_x_sq * eps_22_inv

        m10 = np.zeros_like(m00)
        m11 = -k_x * mu_02 * mu_22_inv
        m12 = (mu_02 * mu_20 * mu_22_inv - mu_00) * ones_like_k_x
        m13 = (mu_02 * mu_21 * mu_22_inv - mu_01) * ones_like_k_x

        m20 = (eps_12 * eps_20 * eps_22_inv - eps_10) * ones_like_k_x
        m21 = k_x_sq * mu_22_inv - eps_11 + eps_12 * eps_21 * eps_22_inv
        m22 = -k_x * mu_20 * mu_22_inv
        m23 = k_x * (eps_12 * eps_22_inv - mu_21 * mu_22_inv)

        m30 = (eps_00 - eps_02 * eps_20 * eps_22_inv) * ones_like_k_x
        m31 = (eps_01 - eps_02 * eps_21 * eps_22_inv) * ones_like_k_x
        m32 = np.zeros_like(m00)
        m33 = -k_x * eps_02 * eps_22_inv

        # Stack the matrix elements into a 4x4 matrix
        # Always use the same stacking approach - shape [..., 4, 4]
        row0 = np.stack([m00, m01, m02, m03], axis=-1)
        row1 = np.stack([m10, m11, m12, m13], axis=-1)
        row2 = np.stack([m20, m21, m22, m23], axis=-1)
        row3 = np.stack([m30, m31, m32, m33], axis=-1)

        self.berreman_matrix = np.stack([row0, row1, row2, row3], axis=-2).astype(np.complex128)

    def delta_permutations(self) -> None:
        """Determine batch dimensions for mode - no longer does permutations.

        With unified shape convention, permutations are no longer needed.
        Just sets batch_dims for downstream operations.
        """
        # Set batch_dims based on mode
        if self.mode in ["Incident", "Azimuthal", "Dispersion", "FullSweep"]:
            self.batch_dims = self.berreman_matrix.ndim - 2
        elif self.mode == "Simple":
            self.batch_dims = 0
        elif self.mode in ["airgap", "simple_airgap", "full_sweep_airgap"]:
            self.batch_dims = self.berreman_matrix.ndim - 2
        elif self.mode in ["azimuthal_airgap", "simple_scalar_airgap"]:
            self.batch_dims = 0
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def wave_sorting(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sort wave modes into transmitted and reflected components.

        Solves the eigenvalue problem for the Berreman matrix and sorts
        eigenmodes based on their propagation direction (sign of Im(kz) or Re(kz)).

        Returns:
            Tuple of (transmitted_wavevectors, reflected_wavevectors,
                    transmitted_fields, reflected_fields)

        Note:
            Forward-propagating modes (Im(kz) > 0 or Re(kz) > 0) are transmitted.
            Backward-propagating modes are reflected.
        """
        wavevectors, fields = np.linalg.eig(self.berreman_matrix)

        # Vectorized sorting - prefer imaginary part if significant, else real part
        is_complex = np.abs(np.imag(wavevectors)) > 1e-9
        idx_real = np.argsort(np.real(wavevectors), axis=-1)[..., ::-1]  # DESCENDING
        idx_imag = np.argsort(np.imag(wavevectors), axis=-1)[..., ::-1]  # DESCENDING
        indices = np.where(is_complex, idx_imag, idx_real)

        # Gather sorted wavevectors and fields using take_along_axis
        sorted_waves = np.take_along_axis(wavevectors, indices, axis=-1)

        # For fields, need to expand indices to match the [..., 4, 4] shape
        field_indices = indices[..., np.newaxis, :]  # [..., 1, 4] for broadcasting
        sorted_fields = np.take_along_axis(fields, field_indices, axis=-1)

        # Split into transmitted (first 2 modes) and reflected (last 2 modes)
        transmitted_wavevectors = sorted_waves[..., :2]
        reflected_wavevectors = sorted_waves[..., 2:]
        transmitted_fields = sorted_fields[..., :2]
        reflected_fields = sorted_fields[..., 2:]

        return (
            transmitted_wavevectors,
            reflected_wavevectors,
            transmitted_fields,
            reflected_fields,
        )

    def get_matrix(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        """Construct transfer matrix from eigenvalues and eigenvectors.

        Args:
            eigenvalues: Propagation constants (kz) for the wave modes
            eigenvectors: Corresponding electromagnetic field distributions

        Returns:
            Transfer matrix for the layer (4×4 or batched)

        Note:
            For semi-infinite layers, returns eigenvector matrix directly.
            For finite layers, includes phase accumulation:
            M = V · exp(-i·diag(kz)·k0·d) · V⁻¹
            where d is thickness and V is the eigenvector matrix.
        """
        if self.semi_infinite:
            return eigenvectors

        # Get the mode shapes for matrix calculation
        k_0, eigenvalues_diag, eigenvectors = self._get_matrix_calculation_shapes(
            eigenvalues, eigenvectors
        )

        # For airgap modes, add dimensions to match k_0 frequency dimension
        if self.mode == "azimuthal_airgap" and k_0.ndim > 0:
            # k_0 is [N_freq], eigenvalues_diag is [4, 4]
            # Reshape eigenvalues_diag to [N_freq, 4, 4]
            eigenvalues_diag = eigenvalues_diag[np.newaxis, ...]
            eigenvectors = eigenvectors[np.newaxis, ...]
            # Reshape k_0 to [N_freq, 1, 1]
            k_0_broadcast = k_0[:, np.newaxis, np.newaxis]
        elif self.mode == "airgap" and k_0.ndim > 0:
            # k_0 is [N_freq], eigenvalues_diag is [N_angles, 4, 4]
            # Reshape eigenvalues_diag to [N_angles, N_freq, 4, 4]
            eigenvalues_diag = eigenvalues_diag[:, np.newaxis, ...]
            eigenvectors = eigenvectors[:, np.newaxis, ...]
            # Reshape k_0 to [1, N_freq, 1, 1]
            k_0_broadcast = k_0[np.newaxis, :, np.newaxis, np.newaxis]
        elif self.mode == "Incident" and k_0.ndim > 0:
            # k_0 is [N_freq], eigenvalues_diag is [N_angles, N_freq, 4, 4]
            # Reshape k_0 to [1, N_freq, 1, 1]
            k_0_broadcast = k_0[np.newaxis, :, np.newaxis, np.newaxis]
        elif self.mode == "FullSweep" and k_0.ndim > 0:
            # k_0 is [N_freq], eigenvalues_diag is [N_angles, N_azim, N_freq, 4, 4]
            # Reshape k_0 to [1, 1, N_freq, 1, 1]
            k_0_broadcast = k_0[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        elif self.mode == "full_sweep_airgap" and k_0.ndim > 0:
            # k_0 is [N_freq], eigenvalues_diag is [N_angles, 4, 4]
            # Reshape eigenvalues_diag to [N_angles, N_freq, 4, 4]
            eigenvalues_diag = eigenvalues_diag[:, np.newaxis, np.newaxis, ...]
            eigenvectors = eigenvectors[:, np.newaxis, np.newaxis, ...]
            # Reshape k_0 to [1, 1, N_freq, 1, 1]
            k_0_broadcast = k_0[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        else:
            # Reshape k_0 for proper broadcasting with eigenvalues_diag [..., 4, 4]
            k_0_broadcast = k_0 if isinstance(k_0, np.ndarray) else np.array(k_0)
            while k_0_broadcast.ndim < eigenvalues_diag.ndim:
                k_0_broadcast = k_0_broadcast[..., np.newaxis]

        # Calculate the partial matrix - optimized for diagonal matrices
        exponent = -1.0j * eigenvalues_diag * k_0_broadcast * self.thickness
        if exponent.ndim >= 2 and exponent.shape[-1] == exponent.shape[-2]:
            # For diagonal matrices, matrix exponential is exp() of diagonal elements
            partial = np.zeros_like(exponent)
            diagonal_indices = np.arange(exponent.shape[-1])
            # Check if it's actually diagonal by verifying off-diagonal elements are zero
            off_diag_mask = np.ones(exponent.shape[-2:], dtype=bool)
            off_diag_mask[diagonal_indices, diagonal_indices] = False
            if np.allclose(exponent[..., off_diag_mask], 0, atol=1e-12):
                # It's diagonal, use fast diagonal exponential
                exp_diag = np.exp(exponent[..., diagonal_indices, diagonal_indices])
                partial[..., diagonal_indices, diagonal_indices] = exp_diag
            else:
                # Not actually diagonal, fall back to general matrix exponential
                partial = expm(exponent)
        else:
            partial = expm(exponent)

        # Calculate the transfer matrix
        eigenvectors_inv = np.linalg.inv(eigenvectors)
        transfer_matrix = np.matmul(np.matmul(eigenvectors, partial), eigenvectors_inv)

        return transfer_matrix

    def poynting_reshaping(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape tensors for Poynting vector calculation.

        Returns:
            Tuple of (kx, eps_tensor, mu_tensor) with appropriate shapes
            for energy flow calculations
        """
        return self._get_poynting_tensor_shapes()

    def get_poynting(
        self,
        transmitted_waves: np.ndarray,
        reflected_waves: np.ndarray,
        transmitted_fields: np.ndarray,
        reflected_fields: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Calculate electromagnetic fields and Poynting vectors for all modes.

        Args:
            transmitted_waves: Transmitted mode propagation constants
            reflected_waves: Reflected mode propagation constants
            transmitted_fields: Transmitted mode field components [Ex, Ey, Hx, Hy]
            reflected_fields: Reflected mode field components

        Returns:
            Tuple of (transmitted_profile, reflected_profile) dictionaries,
            each containing all field components (E, H), Poynting vectors (P),
            and propagation constants

        Note:
            The Poynting vector P = (1/2) Re(E × H*) gives time-averaged
            energy flux density. Ez and Hz are computed from Maxwell's
            equations using the material tensors.
        """
        k_x, eps_tensor, mu_tensor = self.poynting_reshaping()

        def calculate_fields(fields):
            """
            Extract the field components from the input tensor.

            Args:
                fields (np.ndarray): The input tensor containing the field components.

            Returns:
                tuple: A tuple containing the extracted field components (Ex, Ey, Hx, Hy).
            """
            Ex, Ey = fields[..., 0, :], fields[..., 1, :]
            Hx, Hy = fields[..., 2, :], fields[..., 3, :]
            return Ex, Ey, Hx, Hy

        transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy = calculate_fields(
            transmitted_fields
        )
        reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy = calculate_fields(reflected_fields)

        def calculate_Ez_Hz(Ex, Ey, Hx, Hy):
            """
            Calculate the Ez and Hz components based on the input field components.

            Args:
                Ex (np.ndarray): The Ex field component.
                Ey (np.ndarray): The Ey field component.
                Hx (np.ndarray): The Hx field component.
                Hy (np.ndarray): The Hy field component.

            Returns:
                tuple: A tuple containing the calculated Ez and Hz components.
            """
            # Broadcast k_x to match field dimensions
            # Fields have shape [..., 2] where 2 is the number of modes
            k_x_broadcast = k_x
            while k_x_broadcast.ndim < Hy.ndim:
                k_x_broadcast = k_x_broadcast[..., np.newaxis]

            Ez = (-1.0 / eps_tensor[..., 2, 2, np.newaxis]) * (
                k_x_broadcast * Hy
                + eps_tensor[..., 2, 0, np.newaxis] * Ex
                + eps_tensor[..., 2, 1, np.newaxis] * Ey
            )
            Hz = (1.0 / mu_tensor[..., 2, 2, np.newaxis]) * (
                k_x_broadcast * Ey
                - mu_tensor[..., 2, 0, np.newaxis] * Hx
                - mu_tensor[..., 2, 1, np.newaxis] * Hy
            )
            return Ez, Hz

        transmitted_Ez, transmitted_Hz = calculate_Ez_Hz(
            transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy
        )
        reflected_Ez, reflected_Hz = calculate_Ez_Hz(
            reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy
        )

        def calculate_poynting(Ex, Ey, Ez, Hx, Hy, Hz):
            """
            Calculate the Poynting vector components based on the input field components.

            Args:
                Ex (np.ndarray): The Ex field component.
                Ey (np.ndarray): The Ey field component.
                Ez (np.ndarray): The Ez field component.
                Hx (np.ndarray): The Hx field component.
                Hy (np.ndarray): The Hy field component.
                Hz (np.ndarray): The Hz field component.

            Returns:
                tuple: A tuple containing the calculated Poynting vector components
                    (Px, Py, Pz, physical_Px, physical_Py, physical_Pz).
            """
            Px = Ey * Hz - Ez * Hy
            Py = Ez * Hx - Ex * Hz
            Pz = Ex * Hy - Ey * Hx
            physical_Px = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
            physical_Py = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
            physical_Pz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
            return Px, Py, Pz, physical_Px, physical_Py, physical_Pz

        transmitted_poynting = calculate_poynting(
            transmitted_Ex,
            transmitted_Ey,
            transmitted_Ez,
            transmitted_Hx,
            transmitted_Hy,
            transmitted_Hz,
        )
        reflected_poynting = calculate_poynting(
            reflected_Ex,
            reflected_Ey,
            reflected_Ez,
            reflected_Hx,
            reflected_Hy,
            reflected_Hz,
        )

        def create_wave_profile(fields, poynting, waves):
            """
            Create a wave profile dictionary based on the input field components,
            Poynting vector components,
            and wavevectors.

            Args:
                fields (tuple): A tuple containing the field components (Ex, Ey, Ez, Hx, Hy, Hz).
                poynting (tuple): A tuple containing the Poynting vector components
                    (Px, Py, Pz, physical_Px, physical_Py, physical_Pz).
                waves (np.ndarray): The wavevectors.

            Returns:
                dict: A dictionary representing the wave profile.
            """
            Ex, Ey, Ez, Hx, Hy, Hz = fields
            Px, Py, Pz, physical_Px, physical_Py, physical_Pz = poynting
            return {
                "Ex": Ex,
                "Ey": Ey,
                "Ez": Ez,
                "Hx": Hx,
                "Hy": Hy,
                "Hz": Hz,
                "Px": Px,
                "Py": Py,
                "Pz": Pz,
                "Px_physical": physical_Px,
                "Py_physical": physical_Py,
                "Pz_physical": physical_Pz,
                "propagation": waves,
            }

        transmitted_fields = (
            transmitted_Ex,
            transmitted_Ey,
            transmitted_Ez,
            transmitted_Hx,
            transmitted_Hy,
            transmitted_Hz,
        )
        reflected_fields = (
            reflected_Ex,
            reflected_Ey,
            reflected_Ez,
            reflected_Hx,
            reflected_Hy,
            reflected_Hz,
        )

        transmitted_wave_profile = create_wave_profile(
            transmitted_fields, transmitted_poynting, transmitted_waves
        )
        reflected_wave_profile = create_wave_profile(
            reflected_fields, reflected_poynting, reflected_waves
        )

        return transmitted_wave_profile, reflected_wave_profile

    def sort_poynting_indices(self, profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Sort wave modes by polarization character (p vs s).

        Args:
            profile: Dictionary containing field and Poynting components

        Returns:
            Sorted profile dictionary with modes ordered by polarization

        Note:
            Uses both Poynting vector and electric field ratios to classify
            modes as primarily p-polarized (more x-component) or s-polarized
            (more y-component). This classification is crucial for properly
            assigning r_pp, r_ss, r_ps, r_sp coefficients.
        """
        # Calculate polarization ratios
        poynting_x = np.abs(profile["Px"]) ** 2
        poynting_y = np.abs(profile["Py"]) ** 2
        electric_x = np.abs(profile["Ex"]) ** 2
        electric_y = np.abs(profile["Ey"]) ** 2

        Cp_E = electric_x / (electric_x + electric_y)
        Cp_P = poynting_x / (poynting_x + poynting_y)

        # Sort by Poynting (descending) or E-field (ascending) based on difference threshold
        indices_P = np.argsort(Cp_P, axis=-1)[..., ::-1]  # DESCENDING
        indices_E = np.argsort(Cp_E, axis=-1)  # ASCENDING

        # Choose sorting method based on distinctiveness of modes
        condition_P = np.abs(Cp_P[..., 1] - Cp_P[..., 0])[..., np.newaxis]
        sorting_indices = np.where(condition_P > 1e-6, indices_P, indices_E)

        # Apply sorting to all profile elements using vectorized take_along_axis
        for key in profile:
            profile[key] = np.take_along_axis(profile[key], sorting_indices, axis=-1)

        return profile

    def sort_profile_back_to_matrix(self) -> np.ndarray:
        """Reconstruct transfer matrix from sorted wave profile.

        Returns:
            Transfer matrix with properly ordered wave modes

        Note:
            For semi-infinite layers, constructs matrix with only transmitted
            modes (zeros for reflected components). For finite layers, combines
            transmitted and reflected modes to form complete transfer matrix.
        """
        transmitted_new_profile = np.stack(
            [
                self.profile.transmitted_Ex,
                self.profile.transmitted_Ey,
                self.profile.transmitted_Hx,
                self.profile.transmitted_Hy,
            ],
            axis=-2,
        )

        if self.semi_infinite:
            transfer_matrix = np.stack(
                [
                    transmitted_new_profile[..., 0],
                    np.zeros_like(transmitted_new_profile[..., 1]),
                    transmitted_new_profile[..., 1],
                    np.zeros_like(transmitted_new_profile[..., 1]),
                ],
                axis=-1,
            )
            return transfer_matrix
        else:
            reflected_new_profile = np.stack(
                [
                    self.profile.reflected_Ex,
                    self.profile.reflected_Ey,
                    self.profile.reflected_Hx,
                    self.profile.reflected_Hy,
                ],
                axis=-2,
            )

            eigenvalues = np.concatenate(
                [self.profile.transmitted_k_z, self.profile.reflected_k_z], axis=-1
            )
            eigenvectors = np.concatenate([transmitted_new_profile, reflected_new_profile], axis=-1)

            transfer_matrix = self.get_matrix(eigenvalues, eigenvectors)

            return transfer_matrix

    def execute(self) -> tuple[WaveProfile, np.ndarray]:
        """Execute complete wave calculation pipeline.

        Returns:
            Tuple of (wave_profile, transfer_matrix)

        Process:
            1. Calculate Berreman matrix
            2. Apply axis permutations
            3. Solve eigenvalue problem and sort modes
            4. Calculate fields and Poynting vectors
            5. Sort modes by polarization
            6. Construct transfer matrix

        Example:
            >>> wave = Wave(kx, eps_tensor, mu_tensor, mode='Simple',
            ...            k_0=k0, thickness=0.5e-4)
            >>> profile, matrix = wave.execute()
        """
        self.delta_matrix_calc()
        self.delta_permutations()

        transmitted_waves, reflected_waves, transmitted_fields, reflected_fields = (
            self.wave_sorting()
        )
        transmitted_wave_profile, reflected_wave_profile = self.get_poynting(
            transmitted_waves, reflected_waves, transmitted_fields, reflected_fields
        )
        transmitted_wave_profile = self.sort_poynting_indices(transmitted_wave_profile)
        reflected_wave_profile = self.sort_poynting_indices(reflected_wave_profile)

        profile = {
            "transmitted": transmitted_wave_profile,
            "reflected": reflected_wave_profile,
        }

        self.profile = WaveProfile(profile)

        matrix = self.sort_profile_back_to_matrix()

        return self.profile, matrix
