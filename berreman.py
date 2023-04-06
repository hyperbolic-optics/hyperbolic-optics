import numpy as np
from scipy.linalg import expm
from device_config import run_on_device

import tensorflow as tf

@run_on_device
def tensorflow_berreman_matrix(kx, eps_tensor, mu_tensor):
    """Constructs the Berreman matrix for a given kx and a given rotation for a range of frequencies."""

    element11 = - kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]
    element12 = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element13 = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    element14 = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]
    
    element21 = tf.zeros(eps_tensor.shape[0], dtype= tf.complex128)
    element22 = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    element23 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    element24 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    element31 = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    element32 = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    element33 = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    element34 = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    element41 = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    element42 = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element43 = tf.zeros(eps_tensor.shape[0],dtype= tf.complex128)
    element44 = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]


    first_row = tf.stack([element11, element12, element13, element14], axis=1)
    second_row = tf.stack([element21, element22, element23, element24], axis=1)
    third_row = tf.stack([element31, element32, element33, element34], axis=1)
    fourth_row = tf.stack([element41, element42, element43, element44], axis=1)

    overall_matrix = tf.stack([first_row, second_row, third_row, fourth_row], axis=2)


    return overall_matrix




def layer_matrix_incidence(eps_tensor, mu_tensor, kx, k0, thickness, quartz = False):

    delta = np.zeros((len(eps_tensor), len(kx), 4, 4), dtype=np.complex128)

    kx = kx[:, np.newaxis]
    eps_tensor = eps_tensor[np.newaxis, ...]
    mu_tensor = mu_tensor[np.newaxis, ...]

    delta[..., 0, 0] = -kx * (eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])
    delta[..., 0, 1] = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 0, 2] = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    delta[..., 0, 3] = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]

    delta[..., 1, 0] = np.zeros_like(kx)
    delta[..., 1, 1] = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    delta[..., 1, 2] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    delta[..., 1, 3] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    delta[..., 2, 0] = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    delta[..., 2, 1] = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    delta[..., 2, 2] = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    delta[..., 2, 3] = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    delta[..., 3, 0] = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    delta[..., 3, 1] = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 3, 2] = np.zeros_like(kx)
    delta[..., 3, 3] = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    delta = np.transpose(delta, (1,0,2,3))

    eigenvalues, vector = np.linalg.eig(delta)

    if quartz:
        
        order = (1.j * eigenvalues).argsort(axis=-1)[..., np.newaxis]
        vector = np.transpose(vector, (0, 1, 3, 2))
        vector = np.take_along_axis(vector, order, axis=-2)

        vector[..., 2:4, :] = 0
        vector[..., 2, :] = vector[..., 1, :]
        vector[..., 1, :] = 0

        vector = np.transpose(vector, (0, 1, 3, 2))

        return vector
    
    eye_matrix = np.eye(4, dtype=np.complex128)
    eigenvalues_diag = eye_matrix * eigenvalues[..., np.newaxis]

    k0_expanded = k0[:, np.newaxis, np.newaxis]
    partial = expm(1.j * eigenvalues_diag * k0_expanded * thickness)
    partial_complete = vector @ partial @ np.linalg.inv(vector)

    return partial_complete
   

def layer_matrix_incidence_azimuth(eps_tensor, mu_tensor, kx, k0, thickness, quartz=False):

    delta = np.zeros((eps_tensor.shape[0], eps_tensor.shape[1], 4, 4), dtype=np.complex128)

    delta[..., 0, 0] = -kx * (eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])
    delta[..., 0, 1] = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 0, 2] = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    delta[..., 0, 3] = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]

    delta[..., 1, 0] = np.zeros_like(kx)
    delta[..., 1, 1] = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    delta[..., 1, 2] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    delta[..., 1, 3] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    delta[..., 2, 0] = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    delta[..., 2, 1] = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    delta[..., 2, 2] = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    delta[..., 2, 3] = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    delta[..., 3, 0] = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    delta[..., 3, 1] = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 3, 2] = np.zeros_like(kx)
    delta[..., 3, 3] = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    eigenvalues, vector = np.linalg.eig(delta)

    if quartz:
        
        order = (1.j * eigenvalues).argsort(axis=-1)[..., np.newaxis]
        vector = np.transpose(vector, (0, 1, 3, 2))
        vector = np.take_along_axis(vector, order, axis=-2)

        vector[..., 2:4, :] = 0
        vector[..., 2, :] = vector[..., 1, :]
        vector[..., 1, :] = 0

        vector = np.transpose(vector, (0, 1, 3, 2))

        return vector
    
    eye_matrix = np.eye(4, dtype=np.complex128)
    eigenvalues_diag = eye_matrix * eigenvalues[..., np.newaxis]

    k0_expanded = k0[:, np.newaxis, np.newaxis, np.newaxis]
    partial = expm(1.j * eigenvalues_diag * k0_expanded * thickness)
    partial_complete = vector @ partial @ np.linalg.inv(vector)

    return partial_complete

if __name__ == '__main__':
    import math as m
    from device_config import run_on_device
    from material_params import Quartz, Ambient_Incident_Prism, Air

    eps_prism = 5.5
    incident_angle = m.pi/4.
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype = tf.complex128)

    quartz = Quartz(frequency_length=300, run_on_device_decorator=run_on_device)
    ext, ord = quartz.permittivity_fetch()
    eps_tensor = quartz.fetch_permittivity_tensor()
    mu_tensor = Air(run_on_device_decorator=run_on_device).construct_tensor_singular() * tf.ones_like(eps_tensor)
    quartz_layer = tensorflow_berreman_matrix(kx, eps_tensor, mu_tensor)

    print(quartz_layer.shape)