import tensorflow as tf

from hyperbolic_optics.device_config import run_on_device


@run_on_device
def anisotropy_rotation_one_value(matrix, theta, phi, beta):

    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    rotation_x = tf.stack(
        [
            tf.stack(
                [tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)],
                axis=-1,
            ),
            tf.stack([tf.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            tf.stack([tf.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = tf.math.cos(phi)
    sin_phi = tf.math.sin(phi)
    rotation_y = tf.stack(
        [
            tf.stack([cos_phi, tf.zeros_like(phi), sin_phi], axis=-1),
            tf.stack(
                [tf.zeros_like(phi), tf.ones_like(phi), tf.zeros_like(phi)], axis=-1
            ),
            tf.stack([-sin_phi, tf.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = tf.math.cos(beta)
    sin_beta = tf.math.sin(beta)
    rotation_z = tf.stack(
        [
            tf.stack([cos_beta, -sin_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack([sin_beta, cos_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack(
                [tf.zeros_like(beta), tf.zeros_like(beta), tf.ones_like(beta)], axis=-1
            ),
        ],
        axis=-2,
    )

    total_rotation = tf.cast(rotation_z @ rotation_y @ rotation_x, dtype=tf.complex128)
    result = total_rotation @ matrix @ tf.linalg.matrix_transpose(total_rotation)

    return result


@run_on_device
def anisotropy_rotation_one_axis(matrix, theta, phi, beta):
    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    rotation_x = tf.stack(
        [
            tf.stack(
                [tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)],
                axis=-1,
            ),
            tf.stack([tf.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            tf.stack([tf.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = tf.math.cos(phi)
    sin_phi = tf.math.sin(phi)
    rotation_y = tf.stack(
        [
            tf.stack([cos_phi, tf.zeros_like(phi), sin_phi], axis=-1),
            tf.stack(
                [tf.zeros_like(phi), tf.ones_like(phi), tf.zeros_like(phi)], axis=-1
            ),
            tf.stack([-sin_phi, tf.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = tf.math.cos(beta)
    sin_beta = tf.math.sin(beta)
    rotation_z = tf.stack(
        [
            tf.stack([cos_beta, -sin_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack([sin_beta, cos_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack(
                [tf.zeros_like(beta), tf.zeros_like(beta), tf.ones_like(beta)], axis=-1
            ),
        ],
        axis=-2,
    )

    total_rotation = tf.cast(rotation_z @ rotation_y @ rotation_x, dtype=tf.complex128)

    matrix = matrix[:, tf.newaxis, :, :]
    total_rotation = total_rotation[tf.newaxis, :, :, :]

    result = total_rotation @ matrix @ tf.linalg.matrix_transpose(total_rotation)

    return result


def anisotropy_rotation_all_axes(matrix, theta, phi, beta):
    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    rotation_x = tf.stack(
        [
            tf.stack(
                [tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)],
                axis=-1,
            ),
            tf.stack([tf.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            tf.stack([tf.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = tf.math.cos(phi)
    sin_phi = tf.math.sin(phi)
    rotation_y = tf.stack(
        [
            tf.stack([cos_phi, tf.zeros_like(phi), sin_phi], axis=-1),
            tf.stack(
                [tf.zeros_like(phi), tf.ones_like(phi), tf.zeros_like(phi)], axis=-1
            ),
            tf.stack([-sin_phi, tf.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = tf.math.cos(beta)
    sin_beta = tf.math.sin(beta)
    rotation_z = tf.stack(
        [
            tf.stack([cos_beta, -sin_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack([sin_beta, cos_beta, tf.zeros_like(beta)], axis=-1),
            tf.stack(
                [tf.zeros_like(beta), tf.zeros_like(beta), tf.ones_like(beta)], axis=-1
            ),
        ],
        axis=-2,
    )

    rotation_x = rotation_x[:, tf.newaxis, tf.newaxis, :, :]
    rotation_y = rotation_y[tf.newaxis, :, tf.newaxis, :, :]
    rotation_z = rotation_z[tf.newaxis, tf.newaxis, :, :, :]

    total_rotation = rotation_z @ rotation_y @ rotation_x

    matrix = matrix[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    total_rotation = total_rotation[tf.newaxis, ...]

    result = total_rotation @ matrix @ tf.linalg.matrix_transpose(total_rotation)

    return result
