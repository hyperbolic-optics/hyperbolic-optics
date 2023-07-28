import tensorflow as tf

class Wave:
    """
    This class will be used to represent the four partial waves in a layer of the structure
    
    This will include the following:
    - Electric Field
    - Magnetic Field
    - Poynting Vector
    - Tangential Component of its wavevector

    Params:
    - Eigenvalues of the layer
    - Eigenmodes of the layer
    """

    def __init__(self, kx, eps_tensor, mu_tensor, mode, k_0 = None, thickness = None, semi_infinite = False, magnet = False):
        
        self.k_x = kx
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
        self.electric_field = None
        self.magnetic_field = None
        self.poynting_vector = None
        self.tangential_wavevector = None

        self.berreman_matrix = None

    
    def mode_reshaping(self):

        k_x = self.k_x
        eps_tensor = self.eps_tensor
        mu_tensor = self.mu_tensor

        match self.mode:
            case 'incidence':
                k_x = self.k_x[:, tf.newaxis]
                eps_tensor = eps_tensor[tf.newaxis, ...]
                mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

            case 'azimuthal':
                mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

            case 'dispersion':
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
            case 'incidence':
                permutation = [2, 1, 3, 0]
                self.batch_dims = 2

            case 'azimuthal':
                permutation = [1, 2, 3, 0]
                self.batch_dims = 2

            case 'dispersion':
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
                self.batch_dims = 1

            case _:
                raise NotImplementedError(f"Mode {self.mode} not implemented")
        
        self.berreman_matrix = tf.transpose(self.berreman_matrix, perm = permutation)

    def get_wave(self):
        """
        Short term replacement of berreman
        TODO: add azimuthal and dispersion modes for anisotropic non semi infinite.
        Azimuthal: k_0 = k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        Dispersion: Nothing needed
        """
        eigenvalues, eigenvectors = tf.linalg.eig(self.berreman_matrix)
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        if not self.semi_infinite:

            match self.mode:

                case 'airgap':
                    eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
                    k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
                    eigenvectors = eigenvectors[tf.newaxis, ...]

                case 'azimuthal_airgap':
                    eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
                    k_0 = self.k_0[:, tf.newaxis, tf.newaxis]
                    eigenvectors = eigenvectors[tf.newaxis, ...]
        
                case'simple_airgap':
                    k_0 = self.k_0

                case 'azimuthal':
                    k_0 = self.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

                case 'dispersion':
                    k_0 = self.k_0

                case _:
                    raise NotImplementedError(f"Mode {self.mode} not implemented")

            partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * self.thickness)
            transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

            return transfer_matrix

        else:
            return
    
    
    def passler_sorting(self):
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
        
        # Apply sort_vector on the last axis of wavevectors
        indices = tf.map_fn(sort_vector, wavevectors, dtype=tf.int32)

        sorted_waves = tf.gather(wavevectors, indices, axis=-1, batch_dims=self.batch_dims)
        sorted_fields = tf.gather(fields, indices, axis=-1, batch_dims=self.batch_dims)

        transmitted_wavevectors = stack_indices(sorted_waves, [0, 1])
        reflected_wavevectors = stack_indices(sorted_waves, [2, 3])
        transmitted_fields = stack_indices(sorted_fields, [0, 1])
        reflected_fields = stack_indices(sorted_fields, [2, 3])


        eigenvectors = tf.stack(
            [
                transmitted_fields[..., 0],
                tf.zeros_like(transmitted_fields[..., 1]),
                transmitted_fields[..., 1],
                tf.zeros_like(transmitted_fields[..., 1]),
            ],
            axis=-1,
        )

        return eigenvectors



    def execute(self):
        self.delta_matrix_calc()
        self.delta_permutations()

        if self.semi_infinite:
            eigenvectors = self.passler_sorting()
            return eigenvectors
        else:
            return self.get_wave()