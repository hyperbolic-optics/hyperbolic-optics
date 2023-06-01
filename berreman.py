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


def _modify_eigenvalues_eigenvectors(eigenvalues, eigenvectors, semi_infinite=False):

    col1 = eigenvectors[..., 0]
    col2 = eigenvectors[..., 2]
    col3 = eigenvectors[..., 1]
    col4 = eigenvectors[..., 3]

    # If we're dealing with a semi-infinite layer, there are no backward waves
    if semi_infinite:
        col2 = tf.zeros_like(eigenvectors[..., 1])
        col4 = tf.zeros_like(eigenvectors[..., 3])

    # Stack the columns of the eigenvectors and eigenvalues
    eigenvector_result = tf.stack([col1, col2, col3, col4], axis=-1)
    eigenvalue_result = tf.stack(
        [
            eigenvalues[..., 0],
            eigenvalues[..., 2],
            eigenvalues[..., 1],
            eigenvalues[..., 3],
        ],
        axis=-1,
    )

    return eigenvalue_result, eigenvector_result


def eigenvalue_vector_sorting(
    eigenvalues, eigenvectors, batch_dims=5, magnet=False, semi_infinite=False
):
    if magnet:
        sorted_indices = tf.argsort(
            tf.math.real(eigenvalues), axis=-1, direction="DESCENDING"
        )
    else:
        sorted_indices = tf.argsort(
            tf.math.imag(eigenvalues), axis=-1, direction="DESCENDING"
        )

    # Reorder eigenvectors using sorted_indices
    ordered_eigenvalues = tf.gather(
        eigenvalues, sorted_indices, axis=-1, batch_dims=batch_dims
    )

    ordered_eigenvectors = tf.gather(
        eigenvectors, sorted_indices, axis=-1, batch_dims=batch_dims
    )

    modified_eigenvalues, modified_eigenvectors = _modify_eigenvalues_eigenvectors(
        ordered_eigenvalues, ordered_eigenvectors, semi_infinite=semi_infinite
    )

    return modified_eigenvalues, modified_eigenvectors


@run_on_device
def berreman_method_general(kx, eps_tensor, mu_tensor):
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
                - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2])
                - (kx**2) / eps_tensor[..., 2, 2],
            ],
            [
                tf.zeros_like(-kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
                -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2],
                (
                    (mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2])
                    - mu_tensor[..., 0, 0]
                )
                * tf.ones_like(kx),
                (
                    (mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2])
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


@run_on_device
def berreman_air_gap(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    magnet=False,
):
    berreman_matrix = tf.transpose(
        berreman_method_general(kx, eps_tensor, mu_tensor), perm=[1, 2, 0]
    )

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=1,
        magnet=magnet,
        semi_infinite=semi_infinite,
    )

    del berreman_matrix

    if semi_infinite:
        return eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
        k0 = k0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        eigenvectors = eigenvectors[tf.newaxis, ...]

        if tf.is_tensor(thickness):
            thickness = thickness[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
            k0 = k0[tf.newaxis, ...]
            eigenvectors = eigenvectors[tf.newaxis, ...]


        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix


@run_on_device
def berreman_simple_air_gap(kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    magnet=False):
    """
    Function used to calculate transfer matrix for air gap at only one incident angle.
    """
    if tf.is_tensor(k0):
        berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                    perm = [1, 0])
        batch_size = 0
    
    else:
        berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                    perm = [1, 2, 0])
        batch_size = 1

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=batch_size,
        magnet=magnet,
        semi_infinite=semi_infinite,
    )

    if semi_infinite:
        return eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        if tf.is_tensor(k0):
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
            k0 = k0[:, tf.newaxis, tf.newaxis]
            eigenvectors = eigenvectors[tf.newaxis, ...]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)
        
        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix



def berreman_simple_azimuthal(kx,
    eps_tensor,
    mu_tensor,
    k0,
    thickness,
    semi_infinite=True,
    magnet=False):
    """
    Function used to calculate transfer matrix for azimuthal rotation at only one incident angle.
    Includes a range of frequencies with permittivity and permeability tensors.
    Should be semi-infinite.
    """

    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                   perm=[1,2,3,0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=2,
        magnet=False,
        semi_infinite=semi_infinite,
    )

    if semi_infinite:
        return eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        if tf.is_tensor(k0):
            k0 = k0[:, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)
        
        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix


def berreman_simple_dispersion(
        kx,
        eps_tensor,
        mu_tensor,
        k0,
        thickness,
        semi_infinite=True,
        magnet=False
        ):
    
    kx = kx[:, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                   perm=[1,2,3,0])
    
    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=2,
        magnet=False,
        semi_infinite=semi_infinite,
    )

    if semi_infinite:
        return eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)
        
        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix
    

@run_on_device
def berreman_single_rotation(
        kx,
        eps_tensor,
        mu_tensor,
        k0=None,
        thickness=tf.constant(0.5e-4, dtype=tf.complex64),
        semi_infinite=False,
        magnet=False,
):
    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)
    
    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                   perm = [4,1,5,0,2,3])
    berreman_matrix = tf.squeeze(berreman_matrix, axis = [-1,-2])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=2,
        magnet=magnet,
        semi_infinite=semi_infinite,
    )

    del berreman_matrix

    if semi_infinite:
        return eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)
        k0 = k0[:, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix



@run_on_device
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

    berreman_matrix = tf.transpose(
        berreman_method_general(kx, eps_tensor, mu_tensor),
        perm=[2, 1, 3, 4, 5, 6, 0],
    )

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)
    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues, eigenvectors, magnet=magnet, semi_infinite=semi_infinite
    )

    del berreman_matrix

    if semi_infinite:
        return eigenvectors

    else:
        k0 = k0[
            :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis
        ]
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k0 * thickness)
        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        return transfer_matrix



@run_on_device
def berreman_dispersion(
    kx,
    eps_tensor,
    mu_tensor,
    k0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    magnet=False,
):
    
    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)
    
    berreman_matrix = tf.transpose(berreman_method_general(kx, eps_tensor, mu_tensor),
                                   perm = [4,1,5,0,2,3])
    berreman_matrix = tf.squeeze(berreman_matrix, axis = [-1,-2])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=2,
        magnet=magnet,
        semi_infinite=semi_infinite,
    )

    return eigenvalues




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
    if mode == "airgap":
        berreman_matrix_incidence = berreman_air_gap(
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
            magnet=magnet,
        )
        return berreman_matrix_all_anisotropy
    
    elif mode == "single_rotation":
        berreman_matrix_single_rotation = berreman_single_rotation(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return berreman_matrix_single_rotation
    
    elif mode == "dispersion":
        dispersion_values = berreman_dispersion(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return dispersion_values
    
    elif mode == "simple_airgap":
        airgapsimple = berreman_simple_air_gap(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return airgapsimple
    
    elif mode == "simple_azimuthal":
        azimuthalsimple = berreman_simple_azimuthal(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return azimuthalsimple
    
    elif mode == "simple_dispersion":
        dispersionsimple = berreman_simple_dispersion(
            kx,
            eps_tensor,
            mu_tensor,
            k0=k0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return dispersionsimple

    else:
        raise ValueError(
            "Invalid mode specified for 4x4 Matrix."
        )
