import tensorflow as tf

from device_config import run_on_device


def reflection_coefficients(T):
    """
    T: tensor of shape (..., 4, 4) with berreman matrix
    Computes reflection coefficients for p and s polarizations.
    """
    bottom_line = T[..., 0, 0] * T[..., 2, 2] - T[..., 0, 2] * T[..., 2, 0]
    r_pp = (T[..., 0, 0] * T[..., 3, 2] - T[..., 3, 0] * T[..., 0, 2]) / bottom_line
    r_ps = (T[..., 0, 0] * T[..., 1, 2] - (T[..., 1, 0] * T[..., 0, 2])) / bottom_line
    r_sp = (T[..., 3, 0] * T[..., 2, 2] - T[..., 3, 2] * T[..., 2, 0]) / bottom_line
    r_ss = (T[..., 1, 0] * T[..., 2, 2] - T[..., 1, 2] * T[..., 2, 0]) / bottom_line

    return tf.stack([r_pp, r_ps, r_sp, r_ss])

def modify_eigenvectors(ordered_eigenvectors):
    """
    ordered_eigenvectors: tensor of shape (..., 4, 4, 2) with ordered eigenvectors
    Modifies the eigenvectors to be in the correct order for a semi-infinite layer.
    """
    # Get the first column of the eigenvectors
    col1 = ordered_eigenvectors[...,0]
    # Create a zero-filled tensor shaped like the eigenvectors
    col2_zeros = tf.zeros_like(ordered_eigenvectors[...,1])
    # Get the third column of the eigenvectors, which is the current second column
    col3 = ordered_eigenvectors[...,1]
    # Create a zero-filled tensor shaped like the eigenvectors
    col4_zeros = tf.zeros_like(ordered_eigenvectors[...,3])
    # Stack the modified eigenvectors along the last axis
    result = tf.stack([col1, col2_zeros, col3, col4_zeros], axis=-1)
    
    return result

def modify_eigenvalues_eigenvectors_non_semi_infinite(ordered_eigenvalues, ordered_eigenvectors):
    col1 = ordered_eigenvectors[..., 0]
    col2 = ordered_eigenvectors[...,2]
    col3 = ordered_eigenvectors[...,1]
    col4 = ordered_eigenvectors[...,3]
    eigenvector_result = tf.stack([col1, col2, col3, col4], axis=-1)
    
    eigenvalue_result = tf.stack([ordered_eigenvalues[...,0],
                          ordered_eigenvalues[...,2],
                          ordered_eigenvalues[...,1],
                          ordered_eigenvalues[...,3]], axis=-1)
    
    return eigenvalue_result, eigenvector_result


def berreman_method_general(
    kx,
    eps_tensor,
    mu_tensor):
    """
    kx: wavevector in x direction
    eps_tensor: tensor of shape (..., 3, 3) with epsilon tensor
    mu_tensor: tensor of shape (..., 3, 3) with mu tensor
    Computes general berreman matrix, assuming that shapes will broadcast correctly.
    Does not handle any permutations for different axes.
    """

    berreman_matrix = tf.stack(
        [
                [
                    -kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2],
                    kx
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
                    * tf.ones_like(kx),
                    mu_tensor[..., 1, 1]
                    - (
                        mu_tensor[..., 1, 2]
                        * mu_tensor[..., 2, 1]
                        / mu_tensor[..., 2, 2]
                    )
                    - (kx**2) / eps_tensor[..., 2, 2],
                ],
                [
                    tf.zeros_like(-kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
                    -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2],
                    (
                        (
                            mu_tensor[..., 0, 2]
                            * mu_tensor[..., 2, 0]
                            / mu_tensor[..., 2, 2]
                        )
                        - mu_tensor[..., 0, 0]
                    )
                    * tf.ones_like(kx),
                    (
                        (
                            mu_tensor[..., 0, 2]
                            * mu_tensor[..., 2, 1]
                            / mu_tensor[..., 2, 2]
                        )
                        - mu_tensor[..., 0, 1]
                    )
                    * tf.ones_like(kx),
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
                    * tf.ones_like(kx),
                    (kx**2) / mu_tensor[..., 2, 2]
                    - eps_tensor[..., 1, 1]
                    + (
                        eps_tensor[..., 1, 2]
                        * eps_tensor[..., 2, 1]
                        / eps_tensor[..., 2, 2]
                    ),
                    -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2],
                    kx
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
                    * tf.ones_like(kx),
                    (
                        eps_tensor[..., 0, 1]
                        - (
                            eps_tensor[..., 0, 2]
                            * eps_tensor[..., 2, 1]
                            / eps_tensor[..., 2, 2]
                        )
                    )
                    * tf.ones_like(kx),
                    tf.zeros_like(-kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
                    -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2],
                ],
            ],
            axis=-1,
        )

    return berreman_matrix


def berreman_simple(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
):
    """
    kx: wavevector in x direction
    eps_tensor: tensor of shape (..., 3, 3) with epsilon tensor
    mu_tensor: tensor of shape (..., 3, 3) with mu tensor
    k0: wavevector
    thickness: thickness of layer in cm
    semi_infinite: whether the layer is semi-infinite
    """

    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor), perm=[1, 2, 0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(
            tf.math.imag(eigenvalues), axis=-1, direction="DESCENDING"
        )

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(
            eigenvectors, sorted_indices, axis=-1, batch_dims=2
        )

        # Replace the third column with the second column and set columns 2 and 4 to 0
        ordered_eigenvectors = tf.stack(
            [
                ordered_eigenvectors[:, :, :, 0],
                tf.zeros_like(ordered_eigenvectors[:, :, :, 1]),
                ordered_eigenvectors[:, :, :, 1],
                tf.zeros_like(ordered_eigenvectors[:, :, :, 3]),
            ],
            axis=3,
        )

        partial = ordered_eigenvectors

    else:
        eigenvectors = eigenvectors[:, tf.newaxis, :, :]

        # Propagation part
        # Create diagonal matrix with eigenvalues
        eye_matrix = tf.eye(4, batch_shape=[eigenvalues.shape[0]], dtype=tf.complex64)

        eigenvalues_diag = (eye_matrix * eigenvalues[:, :, tf.newaxis])[
            :, tf.newaxis, :, :
        ]

        # Compute partial using the exponential function
        k0_expanded = k0[tf.newaxis, :, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = tf.transpose(
            eigenvectors @ partial @ tf.linalg.inv(eigenvectors), perm=[1, 0, 2, 3]
        )

    return partial


def berreman_incidence(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
):
    """Constructs the Berreman matrix for a range of kx values, for a given frequency."""

    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor), perm=[1, 2, 0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(
            tf.math.imag(eigenvalues), axis=-1, direction="DESCENDING"
        )

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(
            eigenvectors, sorted_indices, axis=-1, batch_dims=2
        )

        # Replace the third column with the second column and set columns 2 and 4 to 0
        ordered_eigenvectors = tf.stack(
            [
                ordered_eigenvectors[:, :, :, 0],
                tf.zeros_like(ordered_eigenvectors[:, :, :, 1]),
                ordered_eigenvectors[:, :, :, 1],
                tf.zeros_like(ordered_eigenvectors[:, :, :, 3]),
            ],
            axis=3,
        )

        partial = ordered_eigenvectors

    else:
        eigenvectors = eigenvectors[:, tf.newaxis, :, :]

        # Propagation part
        # Create diagonal matrix with eigenvalues
        eye_matrix = tf.eye(4, batch_shape=[eigenvalues.shape[0]], dtype=tf.complex64)

        eigenvalues_diag = (eye_matrix * eigenvalues[:, :, tf.newaxis])[
            tf.newaxis, :, tf.newaxis, :, :
        ]

        # Compute partial using the exponential function
        k0_expanded = k0[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
        thickness = thickness[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = tf.transpose(
            eigenvectors @ partial @ tf.linalg.inv(eigenvectors), perm=[0, 2, 1, 3, 4]
        )

    return partial


def berreman_all_anisotropy(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    magnet=False,
):
    """
    Constructs the Berreman matrix for a range of kx values, for a range of frequencies and anisotropy rotations.
    Axes of berreman_matrix after permutation:
    0: kx
    1: frequency
    2: anisotropy rotation X
    3: anisotropy rotation Y
    4: anisotropy rotation Z
    5 + 6: 4x4 berreman matrix
    """

    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)
    
    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
        perm=[2, 1, 3, 4, 5, 6, 0],
    )

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    del berreman_matrix

    if semi_infinite:
        if magnet:
            # Sort indices of eigenvalues in descending order
            sorted_indices = tf.argsort(
                tf.math.real(eigenvalues), axis=-1, direction="DESCENDING"
            )
        else:
            sorted_indices = tf.argsort(
                tf.math.imag(eigenvalues), axis=-1, direction="DESCENDING"
            )

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(
            eigenvectors, sorted_indices, axis=-1, batch_dims=5
        )

        # Replace the third column with the second column and set columns 2 and 4 to 0
        modified_eigenvectors = modify_eigenvectors(ordered_eigenvectors)

        return modified_eigenvectors

    else:
        # # Propagation part
        # # Sort the indices of eigenvalues in descending order by their absolute value
        # sorted_indices = tf.argsort(tf.math.imag(eigenvalues), axis=-1, direction="DESCENDING")

        # # Reorder eigenvalues and eigenvectors using sorted_indices
        # ordered_eigenvalues = tf.gather(eigenvalues, sorted_indices, axis=-1, batch_dims=5)
        # ordered_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=-1, batch_dims=5)

        # modified_eigenvalues, modified_eigenvectors = modify_eigenvalues_eigenvectors_non_semi_infinite(ordered_eigenvalues, 
        #                                                                                                 ordered_eigenvectors)

        # Create diagonal matrix with eigenvalues
        eigenvalues = tf.linalg.diag(eigenvalues)
        k0_expanded = k0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1j * eigenvalues * k0_expanded * thickness)
        partial = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return partial



def transfer_matrix_wrapper(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=None,
    semi_infinite=False,
    magnet=False,
    mode="simple",
):

    if mode == "simple":
        berreman_matrix_simple = berreman_simple(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
        )
        return berreman_matrix_simple

    elif mode == "incidence":
        berreman_matrix_incidence = berreman_incidence(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
        )
        return berreman_matrix_incidence

    elif mode == "all_anisotropy":
        berreman_matrix_all_anisotropy = berreman_all_anisotropy(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet
            )
        return berreman_matrix_all_anisotropy

    else:
        raise ValueError("Invalid mode specified. Choose from 'simple', 'incidence', or 'all_anisotropy'.")


