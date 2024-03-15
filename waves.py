import tensorflow as tf

class WaveProfile:

    def __init__(self, profile):
        
        self.transmitted_Ex = profile['transmitted']['Ex']
        self.transmitted_Ey = profile['transmitted']['Ey']
        self.transmitted_Ez = profile['transmitted']['Ez']
        self.transmitted_Hx = profile['transmitted']['Hx']
        self.transmitted_Hy = profile['transmitted']['Hy']
        self.transmitted_Hz = profile['transmitted']['Hz']
        self.transmitted_Px = profile['transmitted']['Px_physical']
        self.transmitted_Py = profile['transmitted']['Py_physical']
        self.transmitted_Pz = profile['transmitted']['Pz_physical']
        self.transmitted_k_z = profile['transmitted']['propagation']

        self.reflected_Ex = profile['reflected']['Ex']
        self.reflected_Ey = profile['reflected']['Ey']
        self.reflected_Ez = profile['reflected']['Ez']
        self.reflected_Hx = profile['reflected']['Hx']
        self.reflected_Hy = profile['reflected']['Hy']
        self.reflected_Hz = profile['reflected']['Hz']
        self.reflected_Px = profile['reflected']['Px_physical']
        self.reflected_Py = profile['reflected']['Py_physical']
        self.reflected_Pz = profile['reflected']['Pz_physical']
        self.reflected_k_z = profile['reflected']['propagation']

class Wave:
    """
    This class will be used to represent the four partial waves in a layer of the structure
    
    This will include the following:
    - Electric Field
    - Magnetic Field
    - Poynting Vector
    - Tangential Component of its wavevector
    """

    def __init__(self, kx, eps_tensor, mu_tensor, mode, k_0 = None, thickness = None, semi_infinite = False, magnet = False):
        
        self.k_x = tf.cast(kx, dtype=tf.complex128)
        self.eps_tensor = eps_tensor
        self.mu_tensor = mu_tensor

        self.mode = mode
        self.batch_size = None

        self.k_0 = k_0
        self.thickness = thickness
        self.semi_infinite = semi_infinite
        self.magnet = magnet        
        
        self.eigenvalues = None
        self.eigenvectors = None
        self.berreman_matrix = None

    def _get_mode_shapes(self, mode, k_x, eps_tensor, mu_tensor, eigenvalues=None, eigenvectors=None, poynting = False):
        """
        Get the reshaped tensors based on the mode.

        Args:
            mode (str): The mode of the operation.
            k_x (tf.Tensor): The wavevector in the x direction.
            eps_tensor (tf.Tensor): The epsilon tensor.
            mu_tensor (tf.Tensor): The mu tensor.
            eigenvalues (tf.Tensor, optional): The eigenvalues tensor. Defaults to None.
            eigenvectors (tf.Tensor, optional): The eigenvectors tensor. Defaults to None.

        Returns:
            tuple: A tuple containing the reshaped tensors.

        Raises:
            NotImplementedError: If the mode is not recognized.
        """
        mode_shape_map = {
            'Incident': (lambda: (k_x[:, tf.newaxis], eps_tensor[tf.newaxis, ...], mu_tensor * tf.ones_like(eps_tensor))),
            'Azimuthal': (lambda: (k_x, eps_tensor, mu_tensor * tf.ones_like(eps_tensor))),
            'Dispersion': (lambda: (k_x[:, tf.newaxis], eps_tensor[tf.newaxis, ...], mu_tensor * tf.ones_like(eps_tensor))),
            'airgap': (lambda: (k_x, eps_tensor, mu_tensor)),
            'simple_airgap': (lambda: (k_x, eps_tensor, mu_tensor)),
            'azimuthal_airgap': (lambda: (k_x, eps_tensor, mu_tensor)),
        }

        if mode not in mode_shape_map:
            raise NotImplementedError(f"Mode {mode} not implemented")

        reshaped_tensors = mode_shape_map[mode]()

        if poynting:
            mode_poynting_map = {
                'Incident': (lambda: (self.k_x[tf.newaxis, :, tf.newaxis], self.eps_tensor[:, tf.newaxis, tf.newaxis, ...], tf.ones_like(self.eps_tensor[:, tf.newaxis, tf.newaxis, ...]) * self.mu_tensor)),
                'airgap': (lambda: (self.k_x[:, tf.newaxis], self.eps_tensor, self.mu_tensor * tf.ones_like(self.eps_tensor))),
                'simple_airgap': (lambda: (self.k_x[:, tf.newaxis], self.eps_tensor, self.mu_tensor * tf.ones_like(self.eps_tensor))),
                'Azimuthal': (lambda: (self.k_x, self.eps_tensor[:, :, tf.newaxis, ...], self.mu_tensor * tf.ones_like(self.eps_tensor[:, :, tf.newaxis, ...]))),
                'azimuthal_airgap': (lambda: (self.k_x, self.eps_tensor[tf.newaxis, ...], self.mu_tensor * tf.ones_like(self.eps_tensor[tf.newaxis, ...]))),
                'Dispersion': (lambda: (self.k_x[:, tf.newaxis, tf.newaxis], self.eps_tensor[tf.newaxis, :, tf.newaxis, ...], self.mu_tensor * tf.ones_like(self.eps_tensor[tf.newaxis, :, tf.newaxis, ...]))),
            }

            reshaped_tensors = mode_poynting_map[mode]()

        if eigenvalues is not None and eigenvectors is not None:
            eigenvalues_diag = tf.linalg.diag(eigenvalues)
            mode_matrix_map = {
                'Incident': (lambda: (self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis], eigenvalues_diag, eigenvectors)),
                'airgap': (lambda: (self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis] if tf.is_tensor(self.k_0) else self.k_0, eigenvalues_diag[tf.newaxis, ...], eigenvectors[tf.newaxis, ...])),
                'simple_airgap': (lambda: (self.k_0, eigenvalues_diag[:, tf.newaxis, ...], eigenvectors[:, tf.newaxis, ...])),
                'Azimuthal': (lambda: (self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis], eigenvalues_diag, eigenvectors)),
                'azimuthal_airgap': (lambda: (self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis], eigenvalues_diag[tf.newaxis, tf.newaxis, ...], eigenvectors[tf.newaxis, tf.newaxis, ...])),
                'Dispersion': (lambda: (self.k_0, eigenvalues_diag, eigenvectors)),
            }
            k_0, eigenvalues_diag, eigenvectors = mode_matrix_map[mode]()
            reshaped_tensors += (k_0, eigenvalues_diag, eigenvectors)

        return reshaped_tensors
    

    def mode_reshaping(self):
        """
        Reshape the k_x, eps_tensor, and mu_tensor based on the mode.

        Returns:
            tuple: A tuple containing the reshaped k_x, eps_tensor, and mu_tensor.
        """
        k_x, eps_tensor, mu_tensor = self._get_mode_shapes(self.mode, self.k_x, self.eps_tensor, self.mu_tensor)
        return k_x, eps_tensor, mu_tensor

    
    def delta_matrix_calc(self):
        """
        Calculates the Berreman 4x4 Matrix for the layer

        k_x: wavevector in x direction
        eps_tensor: tensor of shape (..., 3, 3) with epsilon tensor
        mu_tensor: tensor of shape (..., 3, 3) with mu tensor
        Computes general berreman matrix, assuming that shapes will broadcast correctly.
        Does not handle any permutations for different axes.

        Returns matrix of shape (..., 4, 4) with the berreman matrix
        """
        k_x, eps_tensor, mu_tensor = self.mode_reshaping()
        
        self.berreman_matrix = tf.stack(
        [
            [
                -k_x * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2],
                k_x
                * (
                    (mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2])
                    - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
                ),
                (
                    mu_tensor[..., 1, 0]
                    - (
                        mu_tensor[..., 1, 2]
                        * mu_tensor[..., 2, 0]
                        / mu_tensor[..., 2, 2]
                    )
                )
                * tf.ones_like(k_x),
                mu_tensor[..., 1, 1]
                - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2])
                - (k_x**2) / eps_tensor[..., 2, 2],
            ],
            [
                tf.zeros_like(-k_x * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
                -k_x * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2],
                (
                    (mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2])
                    - mu_tensor[..., 0, 0]
                )
                * tf.ones_like(k_x),
                (
                    (mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2])
                    - mu_tensor[..., 0, 1]
                )
                * tf.ones_like(k_x),
            ],
            [
                (
                    (
                        eps_tensor[..., 1, 2]
                        * eps_tensor[..., 2, 0]
                        / eps_tensor[..., 2, 2]
                    )
                    - eps_tensor[..., 1, 0]
                )
                * tf.ones_like(k_x),
                (k_x**2) / mu_tensor[..., 2, 2]
                - eps_tensor[..., 1, 1]
                + (
                    eps_tensor[..., 1, 2]
                    * eps_tensor[..., 2, 1]
                    / eps_tensor[..., 2, 2]
                ),
                -k_x * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2],
                k_x
                * (
                    (eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2])
                    - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2])
                ),
            ],
            [
                (
                    eps_tensor[..., 0, 0]
                    - (
                        eps_tensor[..., 0, 2]
                        * eps_tensor[..., 2, 0]
                        / eps_tensor[..., 2, 2]
                    )
                )
                * tf.ones_like(k_x),
                (
                    eps_tensor[..., 0, 1]
                    - (
                        eps_tensor[..., 0, 2]
                        * eps_tensor[..., 2, 1]
                        / eps_tensor[..., 2, 2]
                    )
                )
                * tf.ones_like(k_x),
                tf.zeros_like(-k_x * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
                -k_x * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2],
            ],
        ],
        axis=-1,
        )


    def delta_permutations(self):
        """
        Perform permutations on the Berreman matrix based on the mode.

        This method transposes the Berreman matrix according to the specified
        permutation for each mode. It also sets the `batch_dims` attribute
        based on the mode.

        Raises:
            NotImplementedError: If the mode is not recognized.
        """
        mode_permutations = {
            'Incident': ([2, 1, 3, 0], 2),
            'Azimuthal': ([1, 2, 3, 0], 2),
            'Dispersion': ([1, 2, 3, 0], 2),
            'airgap': ([1, 2, 0], 1),
            'simple_airgap': ([1, 2, 0], 1),
            'azimuthal_airgap': ([1, 0], 0),
        }

        if self.mode not in mode_permutations:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        permutation, self.batch_dims = mode_permutations[self.mode]
        self.berreman_matrix = tf.transpose(self.berreman_matrix, perm=permutation)


    def wave_sorting(self):
        wavevectors, fields = tf.linalg.eig(self.berreman_matrix)
        
        def sort_vector(waves):
            # Check if the vector contains complex numbers
            is_complex = tf.math.abs(tf.math.imag(waves)) > 0
            # Indices for sorting real and imaginary parts
            idx_real = tf.argsort(tf.math.real(waves), axis=-1, direction='DESCENDING')
            idx_imag = tf.argsort(tf.math.imag(waves), axis=-1, direction='DESCENDING')
            # Create new indices based on the condition
            indices = tf.where(is_complex, idx_imag, idx_real)
            return indices
        
        def stack_indices(tensor, indices):
            return tf.stack([tensor[..., idx] for idx in indices], axis=-1)
        
        if tf.rank(wavevectors)>1:
            # Apply sort_vector on the last axis of wavevectors
            indices = tf.map_fn(sort_vector, wavevectors, dtype=tf.int32)
        else:
            indices = sort_vector(wavevectors)

        sorted_waves = tf.gather(wavevectors, indices, axis=-1, batch_dims=self.batch_dims)
        sorted_fields = tf.gather(fields, indices, axis=-1, batch_dims=self.batch_dims)
        # sorted_fields, _ = tf.linalg.normalize(sorted_fields, axis=-2)
        

        transmitted_wavevectors = stack_indices(sorted_waves, [0, 1])
        reflected_wavevectors = stack_indices(sorted_waves, [2, 3])
        transmitted_fields = stack_indices(sorted_fields, [0, 1])
        reflected_fields = stack_indices(sorted_fields, [2, 3])

        return transmitted_wavevectors, reflected_wavevectors, transmitted_fields, reflected_fields
    

    def get_matrix(self, eigenvalues, eigenvectors):
        """
        Get the transfer matrix based on the mode.

        Args:
            eigenvalues (tf.Tensor): The eigenvalues tensor.
            eigenvectors (tf.Tensor): The eigenvectors tensor.

        Returns:
            tf.Tensor: The transfer matrix.
        """
        if self.semi_infinite:
            return eigenvectors

        _, _, _, k_0, eigenvalues_diag, eigenvectors = self._get_mode_shapes(
            self.mode, self.k_x, self.eps_tensor, self.mu_tensor, eigenvalues, eigenvectors
        )

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * self.thickness)

        transfer_matrix = tf.matmul(eigenvectors, tf.matmul(partial, tf.linalg.inv(eigenvectors)))

        return transfer_matrix
    
    def poynting_reshaping(self):
        """
        Reshape the tensors for Poynting vector calculation based on the mode.

        Returns:
            tuple: A tuple containing the reshaped k_x, eps_tensor, and mu_tensor.
        """
        k_x, eps_tensor, mu_tensor = self._get_mode_shapes(self.mode, self.k_x, self.eps_tensor, self.mu_tensor, poynting=True)
        return k_x, eps_tensor, mu_tensor
            

    def get_poynting(self, transmitted_waves, reflected_waves, transmitted_fields, reflected_fields):
        """
        Calculate the Poynting vector components for transmitted and reflected waves.

        Args:
            transmitted_waves (tf.Tensor): Transmitted wavevectors.
            reflected_waves (tf.Tensor): Reflected wavevectors.
            transmitted_fields (tf.Tensor): Transmitted field components (Ex, Ey, Hx, Hy).
            reflected_fields (tf.Tensor): Reflected field components (Ex, Ey, Hx, Hy).

        Returns:
            tuple: A tuple containing the transmitted and reflected wave profiles.
        """
        k_x, eps_tensor, mu_tensor = self.poynting_reshaping()

        def calculate_fields(fields):
            Ex, Ey = fields[..., 0, :], fields[..., 1, :]
            Hx, Hy = fields[..., 2, :], fields[..., 3, :]
            return Ex, Ey, Hx, Hy

        transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy = calculate_fields(transmitted_fields)
        reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy = calculate_fields(reflected_fields)

        def calculate_Ez_Hz(Ex, Ey, Hx, Hy):
            Ez = (-1. / eps_tensor[..., 2, 2]) * (k_x * Hy + eps_tensor[..., 2, 0] * Ex + eps_tensor[..., 2, 1] * Ey)
            Hz = (1. / mu_tensor[..., 2, 2]) * (k_x * Ey - mu_tensor[..., 2, 0] * Hx - mu_tensor[..., 2, 1] * Hy)
            return Ez, Hz

        transmitted_Ez, transmitted_Hz = calculate_Ez_Hz(transmitted_Ex, transmitted_Ey, transmitted_Hx, transmitted_Hy)
        reflected_Ez, reflected_Hz = calculate_Ez_Hz(reflected_Ex, reflected_Ey, reflected_Hx, reflected_Hy)

        def calculate_poynting(Ex, Ey, Ez, Hx, Hy, Hz):
            Px = Ey * Hz - Ez * Hy
            Py = Ez * Hx - Ex * Hz
            Pz = Ex * Hy - Ey * Hx
            physical_Px = 0.5 * tf.math.real(Ey * tf.math.conj(Hz) - Ez * tf.math.conj(Hy))
            physical_Py = 0.5 * tf.math.real(Ez * tf.math.conj(Hx) - Ex * tf.math.conj(Hz))
            physical_Pz = 0.5 * tf.math.real(Ex * tf.math.conj(Hy) - Ey * tf.math.conj(Hx))
            return Px, Py, Pz, physical_Px, physical_Py, physical_Pz

        transmitted_poynting = calculate_poynting(transmitted_Ex, transmitted_Ey, transmitted_Ez,
                                                transmitted_Hx, transmitted_Hy, transmitted_Hz)
        reflected_poynting = calculate_poynting(reflected_Ex, reflected_Ey, reflected_Ez,
                                                reflected_Hx, reflected_Hy, reflected_Hz)

        def create_wave_profile(fields, poynting, waves):
            Ex, Ey, Ez, Hx, Hy, Hz = fields
            Px, Py, Pz, physical_Px, physical_Py, physical_Pz = poynting
            return {
                'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
                'Px': Px, 'Py': Py, 'Pz': Pz,
                'Px_physical': physical_Px, 'Py_physical': physical_Py, 'Pz_physical': physical_Pz,
                'propagation': waves
            }

        transmitted_fields = (transmitted_Ex, transmitted_Ey, transmitted_Ez, transmitted_Hx, transmitted_Hy, transmitted_Hz)
        reflected_fields = (reflected_Ex, reflected_Ey, reflected_Ez, reflected_Hx, reflected_Hy, reflected_Hz)

        transmitted_wave_profile = create_wave_profile(transmitted_fields, transmitted_poynting, transmitted_waves)
        reflected_wave_profile = create_wave_profile(reflected_fields, reflected_poynting, reflected_waves)

        return transmitted_wave_profile, reflected_wave_profile
    

    def sort_poynting_indices(self, profile):
        """Sorts the poynting vector by z component"""

        poynting_x = tf.math.abs(profile['Px'])**2.
        poynting_y = tf.math.abs(profile['Py'])**2.

        electric_x = tf.math.abs(profile['Ex'])**2.
        electric_y = tf.math.abs(profile['Ey'])**2.

        # calculate cross-polarisation components for electric field
        denominator_E_field = electric_x + electric_y
        Cp_E = electric_x / denominator_E_field

        # calculate cross-polarisation components for Poynting Vector
        denominator_poynting = poynting_x + poynting_y
        Cp_P = poynting_x / denominator_poynting
        
        indices_P = tf.argsort(Cp_P, axis=-1, direction='DESCENDING')
        indices_E = tf.argsort(Cp_E, axis=-1, direction='ASCENDING')

        condition_P = tf.abs(Cp_P[...,1] - Cp_P[...,0])[..., tf.newaxis]
        thresh = 1.e-6
        overall_condition = (condition_P > thresh)

        sorting_indices = tf.where(overall_condition, indices_P, indices_E)            

        for element in profile:
            profile[element] = tf.gather(profile[element], sorting_indices, axis=-1, batch_dims=self.batch_dims)

        return profile
    
    
    def sort_profile_back_to_matrix(self):
        transmitted_new_profile = tf.stack(
            [self.profile.transmitted_Ex,
            self.profile.transmitted_Ey,
            self.profile.transmitted_Hx,
            self.profile.transmitted_Hy],
            axis=-2
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
                [self.profile.reflected_Ex,
                self.profile.reflected_Ey,
                self.profile.reflected_Hx,
                self.profile.reflected_Hy],
                axis=-2
            )

            eigenvalues = tf.concat(
                [self.profile.transmitted_k_z, self.profile.reflected_k_z], axis=-1
            )

            eigenvectors = tf.concat(
                [transmitted_new_profile, reflected_new_profile], axis=-1
            )

            transfer_matrix = self.get_matrix(eigenvalues, eigenvectors)

            return transfer_matrix
        

    def execute(self):
        self.delta_matrix_calc()
        self.delta_permutations()
        
        transmitted_waves, reflected_waves, transmitted_fields, reflected_fields = self.wave_sorting()
        transmitted_wave_profile, reflected_wave_profile = self.get_poynting(transmitted_waves, reflected_waves, transmitted_fields, reflected_fields)
        transmitted_wave_profile = self.sort_poynting_indices(transmitted_wave_profile)
        reflected_wave_profile = self.sort_poynting_indices(reflected_wave_profile)

        profile = {
            'transmitted': transmitted_wave_profile,
            'reflected': reflected_wave_profile
        }

        self.profile = WaveProfile(profile)

        matrix = self.sort_profile_back_to_matrix()

        return self.profile, matrix