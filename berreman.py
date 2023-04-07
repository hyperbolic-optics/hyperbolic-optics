import numpy as np
from scipy.linalg import expm
from device_config import run_on_device
import time
import tensorflow as tf

import math as m
from device_config import run_on_device
from material_params import Quartz, Ambient_Incident_Prism, Air
from anisotropy_utils import anisotropy_rotation_one_axis

from plots import contour_plot
 
@run_on_device
def reflection_coefficients(T):

    bottom_line = (T[...,0,0] * T[...,2,2] - T[...,0,2] * T[...,2,0])
    r_pp = (T[...,0,0] * T[...,3,2] - T[...,3,0] * T[...,0,2]) / bottom_line
    r_ps = (T[...,0,0] * T[...,1,2] - (T[...,1,0] * T[...,0,2])) / bottom_line
    r_sp = (T[...,3,0] * T[...,2,2] - T[...,3,2] * T[...,2,0]) / bottom_line
    r_ss = (T[...,1,0] * T[...,2,2] - T[...,1,2] * T[...,2,0]) / bottom_line  
    
    return tf.stack([r_pp, r_ps, r_sp, r_ss])

@run_on_device
def layer_matrix_one_theta_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex128), semi_infinite = False):
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

    berreman_matrix = tf.stack([
        [element11, element12, element13, element14],
        [element21, element22, element23, element24],
        [element31, element32, element33, element34],
        [element41, element42, element43, element44]
    ], axis=-1)
    berreman_matrix = tf.transpose(berreman_matrix, perm=[1, 2, 0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)
    
    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(tf.math.imag(eigenvalues), axis=-1, direction='DESCENDING')

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=-1, batch_dims=1)

        # Replace the third column with the second column and set columns 2 and 4 to 0
        ordered_eigenvectors = tf.stack([ordered_eigenvectors[:, :, 0], tf.zeros_like(ordered_eigenvectors[:, :, 1]),
                                    ordered_eigenvectors[:, :, 1], tf.zeros_like(ordered_eigenvectors[:, :, 3])], axis=2)
        
        partial = ordered_eigenvectors
    
    else:
        # Propagation part
        # Create diagonal matrix with eigenvalues
        eye_matrix = tf.eye(4, batch_shape=[eps_tensor.shape[0]], dtype=tf.complex128)
        eigenvalues_diag = eye_matrix * tf.expand_dims(eigenvalues, axis=-1)

        # Compute partial using the exponential function
        k0_expanded = tf.expand_dims(tf.expand_dims(k0, axis=-1), axis=-1)
        partial = tf.linalg.expm(1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return partial


@run_on_device
def layer_matrix_incidence_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex128), semi_infinite = False):
    """Constructs the Berreman matrix for a range of kx values and a range of frequencies"""

    element11 = - kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]
    element12 = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element13 = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2])) * tf.ones_like(kx)
    element14 = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]
    
    element21 = tf.zeros_like(element11)
    element22 = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    element23 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0]) * tf.ones_like(kx)
    element24 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1]) * tf.ones_like(kx)

    element31 = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0]) * tf.ones_like(kx)
    element32 = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    element33 = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    element34 = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    element41 = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])) * tf.ones_like(kx)
    element42 = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])) * tf.ones_like(kx)
    element43 = tf.zeros_like(element11)
    element44 = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    berreman_matrix = tf.stack([
        [element11, element12, element13, element14],
        [element21, element22, element23, element24],
        [element31, element32, element33, element34],
        [element41, element42, element43, element44]
    ], axis=-1)
    berreman_matrix = tf.transpose(berreman_matrix, perm=[2, 1, 3, 0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)
    
    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(tf.math.imag(eigenvalues), axis=-1, direction='DESCENDING')

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=-1, batch_dims=2)

        # Replace the third column with the second column and set columns 2 and 4 to 0
        ordered_eigenvectors = tf.stack([ordered_eigenvectors[:, :,:, 0], tf.zeros_like(ordered_eigenvectors[:, :, :,1]),
                                    ordered_eigenvectors[:, :, :,1], tf.zeros_like(ordered_eigenvectors[:, :, :, 3])], axis=3)
        
        partial = ordered_eigenvectors
    
    return partial


@run_on_device
def layer_matrix_anisotropy_one_axis_rotation_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex128), semi_infinite = False):

    element11 = - kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]
    element12 = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element13 = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2])) * tf.ones_like(kx)
    element14 = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]
    
    element21 = tf.zeros_like(element11)
    element22 = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    element23 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0]) * tf.ones_like(kx)
    element24 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1]) * tf.ones_like(kx)

    element31 = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0]) * tf.ones_like(kx)
    element32 = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    element33 = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    element34 = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    element41 = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])) * tf.ones_like(kx)
    element42 = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])) * tf.ones_like(kx)
    element43 = tf.zeros_like(element11)
    element44 = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]



    berreman_matrix = tf.stack([
        [element11, element12, element13, element14],
        [element21, element22, element23, element24],
        [element31, element32, element33, element34],
        [element41, element42, element43, element44]
    ], axis=1)
    berreman_matrix = tf.transpose(berreman_matrix, perm=[3, 2, 4, 1, 0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)
    
    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(tf.math.imag(eigenvalues), axis=-1, direction='DESCENDING')

        print(sorted_indices.shape)

        ordered_eigenvalues = tf.gather(eigenvalues, sorted_indices, axis=-1, batch_dims=3)

        print(ordered_eigenvalues.shape)

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=-1, batch_dims=3)

        # Replace the third column with the second column and set columns 2 and 4 to 0
        ordered_eigenvectors = tf.stack([ordered_eigenvectors[:,:,:,:,0], tf.zeros_like(ordered_eigenvectors[:,:,:,:,1]),
                                    ordered_eigenvectors[:,:,:,:,1], tf.zeros_like(ordered_eigenvectors[:,:,:,:,3])], axis=-1)
        
        
        partial = ordered_eigenvectors
    
    return partial




def layer_matrix_incidence(eps_tensor, mu_tensor, kx, k0, thickness = 0., quartz = False):

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

@run_on_device
def main_incident():

    eps_prism = 5.5
    incident_angle = tf.linspace(-tf.constant(m.pi, dtype=tf.float32) / 2, tf.constant(m.pi, dtype=tf.float32) / 2, 180)
    
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype = tf.complex128)

    quartz = Quartz(frequency_length=200, run_on_device_decorator=run_on_device)
    k0 = quartz.frequency * 2. * m.pi
    
    ext, ord = quartz.permittivity_fetch()
    eps_tensor = quartz.fetch_permittivity_tensor()

    x_rotation = tf.constant(0., dtype = tf.complex128)
    y_rotation = tf.cast(tf.linspace(0.,m.pi/2.,100), dtype = tf.complex128)
    z_rotation = tf.constant(0., dtype = tf.complex128)

    eps_tensor = anisotropy_rotation_one_axis(eps_tensor, x_rotation, y_rotation, z_rotation)
    non_magnetic_tensor = Air(run_on_device_decorator=run_on_device).construct_tensor_singular() * tf.ones_like(eps_tensor)

    prism_layer = Ambient_Incident_Prism(eps_prism, incident_angle, run_on_device_decorator=run_on_device).construct_tensor()
    # # air_layer = tf.linalg.inv(layer_matrix_one_theta_tensorflow(kx, non_magnetic_tensor, non_magnetic_tensor, k0, thickness = 1.5e-4))
    kx = kx[:, tf.newaxis, tf.newaxis]
    eps_tensor = tf.expand_dims(eps_tensor, axis=0)
    non_magnetic_tensor = tf.expand_dims(non_magnetic_tensor, axis=0)

    print(kx.shape)
    print(eps_tensor.shape)
    print(non_magnetic_tensor.shape)

    quartz_layer = layer_matrix_anisotropy_one_axis_rotation_tensorflow(kx, eps_tensor, non_magnetic_tensor, semi_infinite=True)
    prism_layer = tf.expand_dims(prism_layer, axis=1)
    prism_layer = tf.expand_dims(prism_layer, axis=0)

    T = tf.matmul(prism_layer, quartz_layer)
    r = reflection_coefficients(T)
    print(r.shape)

    contour_plot('theta', r[:,:,:,99].numpy(),  quartz.frequency.numpy().real, incident_angle.numpy().real, 0., None, rotation_x = 0., rotation_y = 0., rotation_z = 0.)





if __name__ == '__main__':
    main_incident()

