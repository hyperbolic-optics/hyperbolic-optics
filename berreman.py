"""
Berreman.py

Responsible for calculating the transfer matrix of a given layer, 
and for sorting of eigenvectors accordingly.

Needs to be refactored significantly to fully utilise GPU."""


import tensorflow as tf
from device_config import run_on_device


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
    """
    Responsible for sorting the eigenvector components such that they multiply
    correctly later on for the transfer matrix field components.
    TODO: Refactor to understand batch_dims more to make more general
    """
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
def berreman_method_general(k_x, eps_tensor, mu_tensor):
    """
    k_x: wavevector in x direction
    eps_tensor: tensor of shape (..., 3, 3) with epsilon tensor
    mu_tensor: tensor of shape (..., 3, 3) with mu tensor
    Computes general berreman matrix, assuming that shapes will broadcast correctly.
    Does not handle any permutations for different axes.
    """

    berreman_matrix = tf.stack(
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

    return berreman_matrix


@run_on_device
def berreman_air_gap(
    k_x,
    eps_tensor,
    mu_tensor,
    k_0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    ):
    """
    Used to calculate layer for air gap within transfer matrix
    """
    berreman_matrix = tf.transpose(
        berreman_method_general(k_x, eps_tensor, mu_tensor), perm=[1, 2, 0]
    )

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    eigenvalues, eigenvectors = eigenvalue_vector_sorting(
        eigenvalues,
        eigenvectors,
        batch_dims=1,
        semi_infinite=semi_infinite,
    )

    del berreman_matrix

    if semi_infinite:
        transfer_matrix = eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
        k_0 = k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]
        eigenvectors = eigenvectors[tf.newaxis, ...]

        if tf.is_tensor(thickness):
            thickness = thickness[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
            k_0 = k_0[tf.newaxis, ...]
            eigenvectors = eigenvectors[tf.newaxis, ...]


        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return transfer_matrix


@run_on_device
def berreman_simple_air_gap(k_x,
    eps_tensor,
    mu_tensor,
    k_0=None,
    thickness=tf.constant(0.5e-4, dtype=tf.complex64),
    semi_infinite=False,
    magnet=False):
    """
    Function used to calculate transfer matrix for air gap at only one incident angle.
    """
    if tf.is_tensor(k_0):
        berreman_matrix = tf.transpose(berreman_method_general(k_x, eps_tensor, mu_tensor),
                                    perm = [1, 0])
        batch_size = 0


    else:
        berreman_matrix = tf.transpose(berreman_method_general(k_x, eps_tensor, mu_tensor),
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
        transfer_matrix = eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        if tf.is_tensor(k_0):
            eigenvalues_diag = eigenvalues_diag[tf.newaxis, ...]
            k_0 = k_0[:, tf.newaxis, tf.newaxis]
            eigenvectors = eigenvectors[tf.newaxis, ...]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return transfer_matrix



def berreman_azimuthal(k_x,
    eps_tensor,
    mu_tensor,
    k_0,
    thickness,
    semi_infinite=True):
    """
    Function used to calculate transfer matrix for azimuthal rotation at only one incident angle.
    Includes a range of frequencies with permittivity and permeability tensors.
    Should be semi-infinite.
    """

    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

    berreman_matrix = tf.transpose(berreman_method_general(k_x, eps_tensor, mu_tensor),
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
        transfer_matrix = eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        if tf.is_tensor(k_0):
            k_0 = k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return transfer_matrix


def berreman_dispersion(
        k_x,
        eps_tensor,
        mu_tensor,
        k_0,
        thickness,
        semi_infinite=True,
        ):
    """
    Used to calculate transfer matrix for the case of dispersion
    (Fixed Frequency, varying incident angle and rotation)
    """

    k_x = k_x[:, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

    berreman_matrix = tf.transpose(berreman_method_general(k_x, eps_tensor, mu_tensor),
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
        transfer_matrix = eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return transfer_matrix


@run_on_device
def berreman_incidence(
        k_x,
        eps_tensor,
        mu_tensor,
        k_0=None,
        thickness=tf.constant(0.5e-4, dtype=tf.complex64),
        semi_infinite=False,
        magnet=False,
        ):
    """
    Used to calculate transfer matrix for the case of varying incident angle
    """
    k_x = k_x[:, tf.newaxis, tf.newaxis, tf.newaxis]
    eps_tensor = eps_tensor[tf.newaxis, ...]
    mu_tensor = mu_tensor * tf.ones_like(eps_tensor)

    berreman_matrix = tf.transpose(berreman_method_general(k_x, eps_tensor, mu_tensor),
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
        transfer_matrix = eigenvectors

    else:
        eigenvalues_diag = tf.linalg.diag(eigenvalues)
        k_0 = k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(-1.0j * eigenvalues_diag * k_0 * thickness)

        transfer_matrix = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return transfer_matrix



def transfer_matrix_wrapper(
    k_x,
    eps_tensor,
    mu_tensor,
    mode,
    k_0=None,
    thickness=None,
    semi_infinite=False,
    magnet=False,
    ):
    """
    Wrapper function for the different transfer matrix functions.
    This is what is interfaced with in other files.
    """
    if mode == "airgap":
        berreman_matrix_incidence = berreman_air_gap(
            k_x,
            eps_tensor,
            mu_tensor,
            k_0=k_0,
            thickness=thickness,
            semi_infinite=semi_infinite,
        )
        return berreman_matrix_incidence

    elif mode == "incidence":
        berreman_matrix_single_rotation = berreman_incidence(
            k_x,
            eps_tensor,
            mu_tensor,
            k_0=k_0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return berreman_matrix_single_rotation

    elif mode == "simple_airgap":

        airgapsimple = berreman_simple_air_gap(
            k_x,
            eps_tensor,
            mu_tensor,
            k_0=k_0,
            thickness=thickness,
            semi_infinite=semi_infinite,
            magnet=magnet,
        )
        return airgapsimple

    elif mode == "azimuthal":
        azimuthalsimple = berreman_azimuthal(
            k_x,
            eps_tensor,
            mu_tensor,
            k_0=k_0,
            thickness=thickness,
            semi_infinite=semi_infinite,
        )
        return azimuthalsimple

    elif mode == "dispersion":
        dispersionsimple = berreman_dispersion(
            k_x,
            eps_tensor,
            mu_tensor,
            k_0=k_0,
            thickness=thickness,
            semi_infinite=semi_infinite,
        )
        return dispersionsimple

    else:
        raise ValueError(
            "Invalid mode specified for 4x4 Matrix."
        )
