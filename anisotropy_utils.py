import numpy as np


def anisotropy_rotation(matrix, phi, beta, theta):

    rotation_x = np.zeros((theta.size, 3, 3))
    rotation_x[:, 0, 0] = 1.
    rotation_x[:, 1, 1] = np.cos(theta)
    rotation_x[:, 1, 2] = -np.sin(theta)
    rotation_x[:, 2, 1] = np.sin(theta)
    rotation_x[:, 2, 2] = np.cos(theta)
    

    rotation_y = np.zeros((phi.size, 3, 3))
    rotation_y[:, 0, 0] = np.cos(phi)
    rotation_y[:, 0, 2] = np.sin(phi)
    rotation_y[:, 1, 1] = 1.
    rotation_y[:, 2, 0] = -np.sin(phi)
    rotation_y[:, 2, 2] = np.cos(phi)
    

    rotation_z = np.zeros((beta.size, 3, 3))
    rotation_z[:, 0, 0] = np.cos(beta)
    rotation_z[:, 0, 1] = -np.sin(beta)
    rotation_z[:, 1, 0] = np.sin(beta)
    rotation_z[:, 1, 1] = np.cos(beta)
    rotation_z[:, 2, 2] = 1.

    total_rotation = rotation_z @ rotation_y @ rotation_x

    matrix = matrix[..., np.newaxis, :, :]
    total_rotation = total_rotation[np.newaxis, ...]

    result = total_rotation @ matrix @ np.transpose(total_rotation, (0, 1, 3, 2))

    return result