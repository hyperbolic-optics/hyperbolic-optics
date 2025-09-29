import numpy as np


def anisotropy_rotation_one_value(matrix, theta, phi, beta):

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    total_rotation = (rotation_z @ rotation_y @ rotation_x).astype(np.complex128)
    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result


def anisotropy_rotation_one_axis(matrix, theta, phi, beta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    total_rotation = (rotation_z @ rotation_y @ rotation_x).astype(np.complex128)

    matrix = matrix[:, np.newaxis, :, :]
    total_rotation = total_rotation[np.newaxis, :, :, :]

    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result


def anisotropy_rotation_all_axes(matrix, theta, phi, beta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    rotation_x = rotation_x[:, np.newaxis, np.newaxis, :, :]
    rotation_y = rotation_y[np.newaxis, :, np.newaxis, :, :]
    rotation_z = rotation_z[np.newaxis, np.newaxis, :, :, :]

    total_rotation = rotation_z @ rotation_y @ rotation_x

    matrix = matrix[:, np.newaxis, np.newaxis, np.newaxis, :, :]
    total_rotation = total_rotation[np.newaxis, ...]

    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result
