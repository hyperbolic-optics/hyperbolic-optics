import tensorflow as tf


class WaveProfile:
    """Class representing the wave profile."""

    def __init__(self, profile):
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
        kx,
        eps_tensor,
        mu_tensor,
        mode,
        k_0=None,
        thickness=None,
        semi_infinite=False,
        magnet=False,
    ):
        self.k_x = tf.cast(kx, dtype=tf.complex128)
        self.eps_tensor = eps_tensor  # Now pre-shaped from materials
        self.mu_tensor = mu_tensor    # Now pre-shaped from materials

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

    def _setup_tensor_shapes(self):
        """Setup and validate tensor shapes, determine batch dimensions."""
        # Ensure k_x is complex
        self.k_x = tf.cast(self.k_x, dtype=tf.complex128)
        
        # Ensure tensors are complex
        self.eps_tensor = tf.cast(self.eps_tensor, dtype=tf.complex128)
        self.mu_tensor = tf.cast(self.mu_tensor, dtype=tf.complex128)
        
        # Determine batch dimensions based on mode
        if self.mode in ["Incident", "Azimuthal", "Dispersion"]:
            self.batch_dims = 2
        elif self.mode in ["airgap", "simple_airgap"]:
            self.batch_dims = 1
        elif self.mode == "azimuthal_airgap":
            self.batch_dims = 0
        elif self.mode == "simple_scalar_airgap":
            self.batch_dims = 0  # No batch dimensions for simple scalar
        elif self.mode == "Simple":
            self.batch_dims = 0  # No batch dimensions for simple scenario
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        # For Simple mode, ensure tensors are just [3, 3] without batch dimensions
        if self.mode == "Simple":
            if len(self.eps_tensor.shape) > 2:
                # If tensors have extra dimensions, squeeze them out
                self.eps_tensor = tf.squeeze(self.eps_tensor, axis=list(range(len(self.eps_tensor.shape) - 2)))
                self.mu_tensor = tf.squeeze(self.mu_tensor, axis=list(range(len(self.mu_tensor.shape) - 2)))
        
        # Validate tensor shapes match (skip for Simple mode which has different validation)
        if self.mode != "Simple":
            eps_shape = tf.shape(self.eps_tensor)
            mu_shape = tf.shape(self.mu_tensor)
            
            # Check that last two dimensions are 3x3 for both tensors
            tf.debugging.assert_equal(eps_shape[-2:], [3, 3], 
                                    message="eps_tensor must have shape [..., 3, 3]")
            tf.debugging.assert_equal(mu_shape[-2:], [3, 3], 
                                    message="mu_tensor must have shape [..., 3, 3]")
            
            # Check that batch dimensions match
            tf.debugging.assert_equal(eps_shape[:-2], mu_shape[:-2], 
                                    message="eps_tensor and mu_tensor must have matching batch dimensions")

    def _get_tensor_shapes_for_mode(self):
        """Get properly shaped tensors for the current mode.
        
        Returns:
            tuple: (k_x, eps_tensor, mu_tensor) with proper shapes for the mode
        """
        if self.mode == "Incident":
            # k_x: [180] -> [180, 1] for broadcasting with tensors [1, 3, 3]
            k_x = self.k_x[:, tf.newaxis] if len(self.k_x.shape) == 1 else self.k_x
            return k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "Azimuthal":
            # k_x: scalar or [1], tensors should be [360, 3, 3]
            k_x = self.k_x
            return k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "Dispersion":
            # k_x: [180] -> [180, 1] for broadcasting with tensors [1, 480, 3, 3]
            k_x = self.k_x[:, tf.newaxis] if len(self.k_x.shape) == 1 else self.k_x
            return k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "Simple":
            # For simple mode, everything should be scalar/simple shapes
            # k_x: scalar, tensors: [3, 3]
            return self.k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "airgap":
            # For airgap modes, tensors should already match k_x shape
            return self.k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "simple_airgap":
            # Similar to airgap but different batch structure
            return self.k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "azimuthal_airgap":
            # Different broadcasting pattern
            return self.k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "simple_scalar_airgap":
            # Simple scalar airgap - no batch dimensions
            return self.k_x, self.eps_tensor, self.mu_tensor
            
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def _get_poynting_tensor_shapes(self):
        """Get tensor shapes for Poynting vector calculation.
        
        Returns:
            tuple: (k_x, eps_tensor, mu_tensor) shaped for Poynting calculation
        """
        # For Poynting calculations, we often need different reshaping
        if self.mode == "Incident":
            k_x = self.k_x[tf.newaxis, :, tf.newaxis]
            eps_tensor = self.eps_tensor[:, tf.newaxis, tf.newaxis, ...]
            mu_tensor = self.mu_tensor[:, tf.newaxis, tf.newaxis, ...]
            return k_x, eps_tensor, mu_tensor
            
        elif self.mode == "airgap":
            k_x = self.k_x[:, tf.newaxis]
            return k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "simple_airgap":
            k_x = self.k_x[:, tf.newaxis]
            return k_x, self.eps_tensor, self.mu_tensor
            
        elif self.mode == "Azimuthal":
            eps_tensor = self.eps_tensor[:, :, tf.newaxis, ...]
            mu_tensor = self.mu_tensor[:, :, tf.newaxis, ...]
            return self.k_x, eps_tensor, mu_tensor
            
        elif self.mode == "azimuthal_airgap":
            eps_tensor = self.eps_tensor[tf.newaxis, ...]
            mu_tensor = self.mu_tensor[tf.newaxis, ...]
            return self.k_x, eps_tensor, mu_tensor
            
        elif self.mode == "simple_scalar_airgap":
            # For simple scalar airgap, add minimal dimensions for consistency
            k_x = self.k_x[tf.newaxis] if tf.rank(self.k_x) == 0 else self.k_x
            eps_tensor = self.eps_tensor[tf.newaxis, ...]
            mu_tensor = self.mu_tensor[tf.newaxis, ...]
            return k_x, eps_tensor, mu_tensor
            
        elif self.mode == "Simple":
            # For simple mode, add minimal dimensions for Poynting calculation
            k_x = self.k_x[tf.newaxis] if tf.rank(self.k_x) == 0 else self.k_x
            eps_tensor = self.eps_tensor[tf.newaxis, ...]
            mu_tensor = self.mu_tensor[tf.newaxis, ...]
            return k_x, eps_tensor, mu_tensor
            
        elif self.mode == "Dispersion":
            k_x = self.k_x[:, tf.newaxis, tf.newaxis]
            eps_tensor = self.eps_tensor[tf.newaxis, :, tf.newaxis, ...]
            mu_tensor = self.mu_tensor[tf.newaxis, :, tf.newaxis, ...]
            return k_x, eps_tensor, mu_tensor
            
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def _get_matrix_calculation_shapes(self, eigenvalues, eigenvectors):
        """Get tensor shapes for matrix calculations with eigenvalues/eigenvectors.
        
        Args:
            eigenvalues: Eigenvalues tensor
            eigenvectors: Eigenvectors tensor
            
        Returns:
            tuple: (k_0, eigenvalues_diag, eigenvectors) with proper shapes
        """
        eigenvalues_diag = tf.linalg.diag(eigenvalues)
        
        if self.mode == "Incident":
            k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
            return k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "airgap":
            k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis] if tf.is_tensor(self.k_0) else self.k_0
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
            eigenvectors = eigenvectors[tf.newaxis, ...]
            return k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "simple_airgap":
            eigenvalues_diag = eigenvalues_diag[:, tf.newaxis, ...]
            eigenvectors = eigenvectors[:, tf.newaxis, ...]
            return self.k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "Azimuthal":
            k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
            return k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "azimuthal_airgap":
            k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, tf.newaxis, ...]
            eigenvectors = eigenvectors[tf.newaxis, tf.newaxis, ...]
            return k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "simple_scalar_airgap":
            # For simple scalar airgap, k_0 is scalar, no extra dimensions needed
            return self.k_0, eigenvalues_diag, eigenvectors

        elif self.mode == "Simple":
            # For simple mode, k_0 is scalar, eigenvalues/vectors are [4, 4]
            return self.k_0, eigenvalues_diag, eigenvectors
            
        elif self.mode == "Dispersion":
            return self.k_0, eigenvalues_diag, eigenvectors
            
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def mode_reshaping(self):
        """Reshape the k_x, eps_tensor, and mu_tensor based on the mode."""
        return self._get_tensor_shapes_for_mode()

    def delta_matrix_calc(self):
        """
        Construct the 4x4 Berreman transfer matrix.
        """
        k_x, eps_tensor, mu_tensor = self.mode_reshaping()

        # Extract relevant tensor components
        eps_20, eps_21, eps_22 = eps_tensor[..., 2, 0], eps_tensor[..., 2, 1], eps_tensor[..., 2, 2]
        eps_10, eps_11, eps_12 = eps_tensor[..., 1, 0], eps_tensor[..., 1, 1], eps_tensor[..., 1, 2]
        eps_00, eps_01, eps_02 = eps_tensor[..., 0, 0], eps_tensor[..., 0, 1], eps_tensor[..., 0, 2]
        mu_12, mu_22 = mu_tensor[..., 1, 2], mu_tensor[..., 2, 2]
        mu_10, mu_20 = mu_tensor[..., 1, 0], mu_tensor[..., 2, 0]
        mu_11, mu_21 = mu_tensor[..., 1, 1], mu_tensor[..., 2, 1]
        mu_02 = mu_tensor[..., 0, 2]
        mu_00, mu_01 = mu_tensor[..., 0, 0], mu_tensor[..., 0, 1]

        # Precompute common terms
        k_x_sq = k_x ** 2
        eps_22_inv = 1.0 / eps_22
        mu_22_inv = 1.0 / mu_22
        ones_like_k_x = tf.ones_like(k_x)

        # Construct the matrix elements
        m00 = -k_x * eps_20 * eps_22_inv
        m01 = k_x * (mu_12 * mu_22_inv - eps_21 * eps_22_inv)
        m02 = (mu_10 - mu_12 * mu_20 * mu_22_inv) * ones_like_k_x
        m03 = mu_11 - mu_12 * mu_21 * mu_22_inv - k_x_sq * eps_22_inv

        m10 = tf.zeros_like(m00)
        m11 = -k_x * mu_02 * mu_22_inv
        m12 = (mu_02 * mu_20 * mu_22_inv - mu_00) * ones_like_k_x
        m13 = (mu_02 * mu_21 * mu_22_inv - mu_01) * ones_like_k_x

        m20 = (eps_12 * eps_20 * eps_22_inv - eps_10) * ones_like_k_x
        m21 = k_x_sq * mu_22_inv - eps_11 + eps_12 * eps_21 * eps_22_inv
        m22 = -k_x * mu_20 * mu_22_inv
        m23 = k_x * (eps_12 * eps_22_inv - mu_21 * mu_22_inv)

        m30 = (eps_00 - eps_02 * eps_20 * eps_22_inv) * ones_like_k_x
        m31 = (eps_01 - eps_02 * eps_21 * eps_22_inv) * ones_like_k_x
        m32 = tf.zeros_like(m00)
        m33 = -k_x * eps_02 * eps_22_inv

        # Stack the matrix elements into a 4x4 matrix
        if self.mode == "Simple":
            # For Simple mode, create matrix directly from scalars
            self.berreman_matrix = tf.convert_to_tensor([
                [m00, m01, m02, m03],
                [m10, m11, m12, m13],
                [m20, m21, m22, m23],
                [m30, m31, m32, m33]
            ], dtype=tf.complex128)
        else:
            # For other modes, use the original stacking method
            self.berreman_matrix = tf.stack([
                [m00, m01, m02, m03],
                [m10, m11, m12, m13],
                [m20, m21, m22, m23],
                [m30, m31, m32, m33]
            ], axis=-1)

    def delta_permutations(self):
        """Perform permutations on the Berreman matrix based on the mode."""

        mode_permutations = {
            "Incident": ([2, 1, 3, 0], 2),
            "Azimuthal": ([1, 2, 3, 0], 2),
            "Dispersion": ([1, 2, 3, 0], 2),
            "Simple": ([0, 1], 0),  # Identity permutation for [4,4] matrix - NO transpose
            "airgap": ([1, 2, 0], 1),
            "simple_airgap": ([1, 2, 0], 1),
            "azimuthal_airgap": ([1, 0], 0),
            "simple_scalar_airgap": ([1,0], 0),
        }

        if self.mode not in mode_permutations:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        permutation, self.batch_dims = mode_permutations[self.mode]
        
        if permutation is not None:
            self.berreman_matrix = tf.transpose(self.berreman_matrix, perm=permutation)

    def wave_sorting(self):
        """
        Sort the wavevectors and fields based on the eigenvalues.
        """

        wavevectors, fields = tf.linalg.eig(self.berreman_matrix)

        def sort_vector(waves):
            """Sort the wavevectors based on their real and imaginary parts."""

            is_complex = tf.math.abs(tf.math.imag(waves)) > 1e-9
            idx_real = tf.argsort(tf.math.real(waves), axis=-1, direction="DESCENDING")
            idx_imag = tf.argsort(tf.math.imag(waves), axis=-1, direction="DESCENDING")
            indices = tf.where(is_complex, idx_imag, idx_real)
    
            return indices

        # Sort the wavevectors based on their rank
        if self.mode == "Simple":
            # For simple mode, no mapping needed, just direct sorting
            indices = sort_vector(wavevectors)
        elif tf.rank(wavevectors) > 1:
            indices = tf.map_fn(sort_vector, wavevectors, dtype=tf.int32)
        else:
            indices = sort_vector(wavevectors)

        # Gather the sorted wavevectors and fields
        sorted_waves = tf.gather(wavevectors, indices, axis=-1, batch_dims=self.batch_dims)
        sorted_fields = tf.gather(fields, indices, axis=-1, batch_dims=self.batch_dims)

        # Split the sorted wavevectors and fields into transmitted and reflected components
        transmitted_wavevectors = tf.stack([sorted_waves[..., 0], sorted_waves[..., 1]], axis=-1)
        reflected_wavevectors = tf.stack([sorted_waves[..., 2], sorted_waves[..., 3]], axis=-1)
        transmitted_fields = tf.stack([sorted_fields[..., 0], sorted_fields[..., 1]], axis=-1)
        reflected_fields = tf.stack([sorted_fields[..., 2], sorted_fields[..., 3]], axis=-1)

        return transmitted_wavevectors, reflected_wavevectors, transmitted_fields, reflected_fields

    def get_matrix(self, eigenvalues, eigenvectors):
        """
        Get the transfer matrix based on the mode.

        Args:
            eigenvalues (tf.Tensor): The eigenvalues of the Berreman matrix.
            eigenvectors (tf.Tensor): The eigenvectors of the Berreman matrix.

        Returns:
            tf.Tensor: The transfer matrix.
        """
        if self.semi_infinite:
            return eigenvectors

        # Get the mode shapes for matrix calculation
        k_0, eigenvalues_diag, eigenvectors = self._get_matrix_calculation_shapes(eigenvalues, eigenvectors)

        # Calculate the partial matrix
        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * self.thickness)

        # Calculate the transfer matrix
        transfer_matrix = tf.matmul(eigenvectors, tf.matmul(partial, tf.linalg.inv(eigenvectors)))

        return transfer_matrix

    def poynting_reshaping(self):
        """
        Reshape the tensors for Poynting vector calculation based on the mode.

        Returns:
            tuple: A tuple containing the reshaped k_x, eps_tensor, and mu_tensor.
        """
        return self._get_poynting_tensor_shapes()

    def get_poynting(self, transmitted_waves, reflected_waves, transmitted_fields, reflected_fields):
        """
        Calculate the Poynting vector components for transmitted and reflected waves.

        Args:
            transmitted_waves (tf.Tensor): The transmitted wavevectors.
            reflected_waves (tf.Tensor): The reflected wavevectors.
            transmitted_fields (tf.Tensor): The transmitted field components.
            reflected_fields (tf.Tensor): The reflected field components.

        Returns:
            tuple: A tuple containing the transmitted and reflected wave profiles.
        """
        k_x, eps_tensor, mu_tensor = self.poynting_reshaping()

        def calculate_fields(fields):
            """
            Extract the field components from the input tensor.

            Args:
                fields (tf.Tensor): The input tensor containing the field components.

            Returns:
                tuple: A tuple containing the extracted field components (Ex, Ey, Hx, Hy).
            """
            Ex, Ey = fields[..., 0, :], fields[..., 1, :]
            Hx, Hy = fields[..., 2, :], fields[..., 3, :]
            return Ex, Ey, Hx, Hy

        transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy = calculate_fields(transmitted_fields)
        reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy = calculate_fields(reflected_fields)

        def calculate_Ez_Hz(Ex, Ey, Hx, Hy):
            """
            Calculate the Ez and Hz components based on the input field components.

            Args:
                Ex (tf.Tensor): The Ex field component.
                Ey (tf.Tensor): The Ey field component.
                Hx (tf.Tensor): The Hx field component.
                Hy (tf.Tensor): The Hy field component.

            Returns:
                tuple: A tuple containing the calculated Ez and Hz components.
            """
            Ez = (-1.0 / eps_tensor[..., 2, 2]) * (k_x * Hy + eps_tensor[..., 2, 0] * Ex + eps_tensor[..., 2, 1] * Ey)
            Hz = (1.0 / mu_tensor[..., 2, 2]) * (k_x * Ey - mu_tensor[..., 2, 0] * Hx - mu_tensor[..., 2, 1] * Hy)
            return Ez, Hz

        transmitted_Ez, transmitted_Hz = calculate_Ez_Hz(transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy)
        reflected_Ez, reflected_Hz = calculate_Ez_Hz(reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy)

        def calculate_poynting(Ex, Ey, Ez, Hx, Hy, Hz):
            """
            Calculate the Poynting vector components based on the input field components.

            Args:
                Ex (tf.Tensor): The Ex field component.
                Ey (tf.Tensor): The Ey field component.
                Ez (tf.Tensor): The Ez field component.
                Hx (tf.Tensor): The Hx field component.
                Hy (tf.Tensor): The Hy field component.
                Hz (tf.Tensor): The Hz field component.

            Returns:
                tuple: A tuple containing the calculated Poynting vector components
                    (Px, Py, Pz, physical_Px, physical_Py, physical_Pz).
            """
            Px = Ey * Hz - Ez * Hy
            Py = Ez * Hx - Ex * Hz
            Pz = Ex * Hy - Ey * Hx
            physical_Px = 0.5 * tf.math.real(Ey * tf.math.conj(Hz) - Ez * tf.math.conj(Hy))
            physical_Py = 0.5 * tf.math.real(Ez * tf.math.conj(Hx) - Ex * tf.math.conj(Hz))
            physical_Pz = 0.5 * tf.math.real(Ex * tf.math.conj(Hy) - Ey * tf.math.conj(Hx))
            return Px, Py, Pz, physical_Px, physical_Py, physical_Pz

        transmitted_poynting = calculate_poynting(
            transmitted_Ex, transmitted_Ey, transmitted_Ez, transmitted_Hx, transmitted_Hy, transmitted_Hz
        )
        reflected_poynting = calculate_poynting(
            reflected_Ex, reflected_Ey, reflected_Ez, reflected_Hx, reflected_Hy, reflected_Hz
        )

        def create_wave_profile(fields, poynting, waves):
            """
            Create a wave profile dictionary based on the input field components, Poynting vector components,
            and wavevectors.

            Args:
                fields (tuple): A tuple containing the field components (Ex, Ey, Ez, Hx, Hy, Hz).
                poynting (tuple): A tuple containing the Poynting vector components
                    (Px, Py, Pz, physical_Px, physical_Py, physical_Pz).
                waves (tf.Tensor): The wavevectors.

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

        transmitted_fields = (transmitted_Ex, transmitted_Ey, transmitted_Ez, transmitted_Hx, transmitted_Hy, transmitted_Hz)
        reflected_fields = (reflected_Ex, reflected_Ey, reflected_Ez, reflected_Hx, reflected_Hy, reflected_Hz)

        transmitted_wave_profile = create_wave_profile(transmitted_fields, transmitted_poynting, transmitted_waves)
        reflected_wave_profile = create_wave_profile(reflected_fields, reflected_poynting, reflected_waves)

        return transmitted_wave_profile, reflected_wave_profile

    def sort_poynting_indices(self, profile):
        """
        Sort the Poynting vector by the z-component.

        Args:
            profile (dict): A dictionary containing the wave profile.

        Returns:
            dict: The updated wave profile with sorted indices.
        """
        poynting_x = tf.math.abs(profile["Px"]) ** 2
        poynting_y = tf.math.abs(profile["Py"]) ** 2

        electric_x = tf.math.abs(profile["Ex"]) ** 2
        electric_y = tf.math.abs(profile["Ey"]) ** 2

        denominator_E_field = electric_x + electric_y
        Cp_E = electric_x / denominator_E_field

        denominator_poynting = poynting_x + poynting_y
        Cp_P = poynting_x / denominator_poynting

        indices_P = tf.argsort(Cp_P, axis=-1, direction="DESCENDING")
        indices_E = tf.argsort(Cp_E, axis=-1, direction="ASCENDING")

        condition_P = tf.abs(Cp_P[..., 1] - Cp_P[..., 0])[..., tf.newaxis]
        thresh = 1e-6
        overall_condition = condition_P > thresh

        sorting_indices = tf.where(overall_condition, indices_P, indices_E)

        for element in profile:
            profile[element] = tf.gather(profile[element], sorting_indices, axis=-1, batch_dims=self.batch_dims)

        return profile

    def sort_profile_back_to_matrix(self):
        """Sort the wave profile back to the transfer matrix."""
        transmitted_new_profile = tf.stack(
            [self.profile.transmitted_Ex, self.profile.transmitted_Ey, self.profile.transmitted_Hx, self.profile.transmitted_Hy],
            axis=-2,
        )

        if self.semi_infinite:
            transfer_matrix = tf.stack(
                [
                    transmitted_new_profile[..., 0],
                    tf.zeros_like(transmitted_new_profile[..., 1]),
                    transmitted_new_profile[..., 1],
                    tf.zeros_like(transmitted_new_profile[..., 1]),
                ],
                axis=-1,
            )
            return transfer_matrix
        else:
            reflected_new_profile = tf.stack(
                [self.profile.reflected_Ex, self.profile.reflected_Ey, self.profile.reflected_Hx, self.profile.reflected_Hy],
                axis=-2,
            )

            eigenvalues = tf.concat([self.profile.transmitted_k_z, self.profile.reflected_k_z], axis=-1)
            eigenvectors = tf.concat([transmitted_new_profile, reflected_new_profile], axis=-1)

            transfer_matrix = self.get_matrix(eigenvalues, eigenvectors)

            return transfer_matrix

    def execute(self):
        """Execute the wave calculations."""
        self.delta_matrix_calc()
        self.delta_permutations()

        transmitted_waves, reflected_waves, transmitted_fields, reflected_fields = self.wave_sorting()
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