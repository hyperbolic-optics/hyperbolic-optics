import tensorflow as tf

class WaveProfile:

    def __init__(self, profile):
        
        self.transmitted_Ex = profile['transmitted']['Ex']
        self.transmitted_Ey = profile['transmitted']['Ey']
        self.transmitted_Ez = profile['transmitted']['Ez']
        self.transmitted_Hx = profile['transmitted']['Hx']
        self.transmitted_Hy = profile['transmitted']['Hy']
        self.transmitted_Hz = profile['transmitted']['Hz']
        self.transmitted_Px = profile['transmitted']['Px']
        self.transmitted_Py = profile['transmitted']['Py']
        self.transmitted_Pz = profile['transmitted']['Pz']
        self.transmitted_k_z = profile['transmitted']['propagation']

        self.reflected_Ex = profile['reflected']['Ex']
        self.reflected_Ey = profile['reflected']['Ey']
        self.reflected_Ez = profile['reflected']['Ez']
        self.reflected_Hx = profile['reflected']['Hx']
        self.reflected_Hy = profile['reflected']['Hy']
        self.reflected_Hz = profile['reflected']['Hz']
        self.reflected_Px = profile['reflected']['Px']
        self.reflected_Py = profile['reflected']['Py']
        self.reflected_Pz = profile['reflected']['Pz']
        self.reflected_k_z = profile['reflected']['propagation']


        pass


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

    
    def mode_reshaping(self):

        k_x = self.k_x
        eps_tensor = self.eps_tensor
        mu_tensor = self.mu_tensor

        match self.mode:
            case 'Incident':
                k_x = self.k_x[:, tf.newaxis]
                eps_tensor = eps_tensor[tf.newaxis, ...]
                mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

            case 'Azimuthal':
                mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

            case 'Dispersion':
                k_x = k_x[:, tf.newaxis]
                eps_tensor = eps_tensor[tf.newaxis, ...]
                mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

            case 'airgap':
                pass

            case 'simple_airgap':
                pass

            case 'azimuthal_airgap':
                pass

            case _:
                raise NotImplementedError(f"Mode {self.mode} not implemented")
        
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
        match self.mode:
            case 'Incident':
                permutation = [2, 1, 3, 0]
                self.batch_dims = 2

            case 'Azimuthal':
                permutation = [1, 2, 3, 0]
                self.batch_dims = 2

            case 'Dispersion':
                permutation = [1, 2, 3, 0]
                self.batch_dims = 2

            case 'airgap':
                permutation = [1, 2, 0]
                self.batch_dims = 1

            case 'simple_airgap':
                permutation = [1, 2, 0]
                self.batch_dims = 1

            case 'azimuthal_airgap':
                permutation = [1, 0]
                self.batch_dims = 0

            case _:
                raise NotImplementedError(f"Mode {self.mode} not implemented")
        
        self.berreman_matrix = tf.transpose(self.berreman_matrix, perm = permutation)


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

        transmitted_wavevectors = stack_indices(sorted_waves, [0, 1])
        reflected_wavevectors = stack_indices(sorted_waves, [2, 3])
        transmitted_fields = stack_indices(sorted_fields, [0, 1])
        reflected_fields = stack_indices(sorted_fields, [2, 3])

        return transmitted_wavevectors, reflected_wavevectors, transmitted_fields, reflected_fields
    

    def get_matrix(self, eigenvalues, eigenvectors):
        """
        Short term replacement of berreman
        TODO: add azimuthal and dispersion modes for anisotropic non semi infinite.
        Azimuthal: k_0 = k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        Dispersion: Nothing needed
        """
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        match self.mode:

            case 'Incident':
                k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

            case 'airgap':
                eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
                k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
                eigenvectors = eigenvectors[tf.newaxis, ...]

            case 'azimuthal_airgap':
                eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
                k_0 = self.k_0[:, tf.newaxis, tf.newaxis]
                eigenvectors = eigenvectors[tf.newaxis, ...]

            case 'Azimuthal':
                k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

            case 'Dispersion':
                k_0 = self.k_0

            case 'simple_airgap':
                k_0 = self.k_0

            case _:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * self.thickness)
        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix
    
    def poynting_reshaping(self):

        match self.mode:
            case 'Dispersion':
                k_x = self.k_x[:, tf.newaxis,  tf.newaxis]
                eps_tensor = self.eps_tensor[tf.newaxis, :, tf.newaxis, ...]
                mu_tensor = tf.ones_like(eps_tensor) * self.mu_tensor

            case 'Incident':
                k_x = self.k_x[tf.newaxis, :, tf.newaxis]
                eps_tensor = self.eps_tensor[:,tf.newaxis, tf.newaxis, ...]
                mu_tensor = tf.ones_like(eps_tensor) * self.mu_tensor

            case 'Azimuthal':
                k_x = self.k_x
                eps_tensor = self.eps_tensor[:, :, tf.newaxis, ...]
                mu_tensor = tf.ones_like(eps_tensor) * self.mu_tensor

            case 'airgap':
                k_x = self.k_x[:, tf.newaxis]
                eps_tensor = self.eps_tensor[tf.newaxis, ...]
                mu_tensor = self.mu_tensor * tf.ones_like(eps_tensor)

            case 'simple_airgap':
                k_x = self.k_x[:,tf.newaxis]
                eps_tensor = self.eps_tensor[tf.newaxis, ...]
                mu_tensor = self.mu_tensor * tf.ones_like(eps_tensor)

            case 'azimuthal_airgap':
                k_x = self.k_x[tf.newaxis]
                eps_tensor = self.eps_tensor[tf.newaxis, ...]
                mu_tensor = self.mu_tensor * tf.ones_like(eps_tensor)

            case _:
                raise NotImplementedError(f"Mode {self.mode} not implemented for poynting vector")

        
        return k_x, eps_tensor, mu_tensor

            

    def get_poynting(self, *args):
        transmitted_waves, reflected_waves, transmitted_fields, reflected_fields = args

        k_x, eps_tensor, mu_tensor = self.poynting_reshaping()

        transmitted_Ex = transmitted_fields[..., 0, :]
        transmitted_Ey = transmitted_fields[..., 1, :]
        reflected_Ex = reflected_fields[..., 0, :]
        reflected_Ey = reflected_fields[..., 1, :]

        transmitted_Hx = transmitted_fields[..., 2, :]
        transmitted_Hy = transmitted_fields[..., 3, :]
        reflected_Hx = reflected_fields[..., 2, :]
        reflected_Hy = reflected_fields[..., 3, :]

        transmitted_Ez = (-1./eps_tensor[...,2,2]) * (k_x * transmitted_Hy + eps_tensor[...,2,0] * transmitted_Ex + eps_tensor[...,2,1] * transmitted_Ey)
        reflected_Ez = (-1./eps_tensor[...,2,2]) * (k_x * reflected_Hy + eps_tensor[...,2,0] * reflected_Ex + eps_tensor[...,2,1] * reflected_Ey)


        transmitted_Hz = (1./mu_tensor[...,2,2]) * (k_x * transmitted_Ey - mu_tensor[...,2,0] * transmitted_Hx - mu_tensor[...,2,1] * transmitted_Hy)
        reflected_Hz = (1./mu_tensor[...,2,2]) * (k_x * reflected_Ey - mu_tensor[...,2,0] * reflected_Hx - mu_tensor[...,2,1] * reflected_Hy)

        transmitted_Px = transmitted_Ey * transmitted_Hz - transmitted_Ez * transmitted_Hy
        transmitted_Py = transmitted_Ez * transmitted_Hx - transmitted_Ex * transmitted_Hz
        transmitted_Pz = transmitted_Ex * transmitted_Hy - transmitted_Ey * transmitted_Hx

        reflected_Px = reflected_Ey * reflected_Hz - reflected_Ez * reflected_Hy
        reflected_Py = reflected_Ez * reflected_Hx - reflected_Ex * reflected_Hz
        reflected_Pz = reflected_Ex * reflected_Hy - reflected_Ey * reflected_Hx
    
        transmitted_wave_profile = {
            'Ex': transmitted_Ex,
            'Ey': transmitted_Ey,
            'Ez': transmitted_Ez,
            'Hx': transmitted_Hx,
            'Hy': transmitted_Hy,
            'Hz': transmitted_Hz,
            'Px': transmitted_Px,
            'Py': transmitted_Py,
            'Pz': transmitted_Pz,
            'propagation': transmitted_waves
        }

        reflected_wave_profile = {
            'Ex': reflected_Ex,
            'Ey': reflected_Ey,
            'Ez': reflected_Ez,
            'Hx': reflected_Hx,
            'Hy': reflected_Hy,
            'Hz': reflected_Hz,
            'Px': reflected_Px,
            'Py': reflected_Py,
            'Pz': reflected_Pz,
            'propagation': reflected_waves
        }

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
        thresh = 1.e-14
        overall_condition = (condition_P > thresh)

        sorting_indices = tf.where(overall_condition, indices_P, indices_E)

        for element in profile:
            profile[element] = tf.gather(profile[element], sorting_indices, axis=-1, batch_dims=self.batch_dims)

        return profile
        

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

        profile_to_return = WaveProfile(profile)

        return profile