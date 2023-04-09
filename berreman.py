import numpy as np
from device_config import run_on_device
import tensorflow as tf

import math as m
from device_config import run_on_device
from material_params import Quartz, Ambient_Incident_Prism, Air
from anisotropy_utils import anisotropy_rotation_one_axis, anisotropy_rotation_all_axes

from plots import contour_plot, all_axis_plot, azimuthal_slider_plot

@run_on_device
def compute_kx(eps_prism, incident_angle):
    return tf.sqrt(eps_prism) * tf.sin(incident_angle)
 

@run_on_device
def reflection_coefficients(T):

    bottom_line = (T[...,0,0] * T[...,2,2] - T[...,0,2] * T[...,2,0])
    r_pp = (T[...,0,0] * T[...,3,2] - T[...,3,0] * T[...,0,2]) / bottom_line
    r_ps = (T[...,0,0] * T[...,1,2] - (T[...,1,0] * T[...,0,2])) / bottom_line
    r_sp = (T[...,3,0] * T[...,2,2] - T[...,3,2] * T[...,2,0]) / bottom_line
    r_ss = (T[...,1,0] * T[...,2,2] - T[...,1,2] * T[...,2,0]) / bottom_line  
    
    return tf.stack([r_pp, r_ps, r_sp, r_ss])

@run_on_device
def layer_matrix_one_theta_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex64), semi_infinite = False):
    """Constructs the Berreman matrix for a given kx and a given rotation for a range of frequencies."""

    print("hello")

    element11 = - kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]
    element12 = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element13 = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    element14 = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]
    
    element21 = tf.zeros(eps_tensor.shape[0], dtype= tf.complex64)
    element22 = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    element23 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    element24 = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    element31 = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    element32 = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    element33 = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    element34 = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    element41 = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    element42 = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    element43 = tf.zeros(eps_tensor.shape[0],dtype= tf.complex64)
    element44 = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    for element in [element11, element12, element13, element14, element21, element22, element23, element24, element31, element32, element33, element34, element41, element42, element43, element44]:
        print(element.shape)

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
        eye_matrix = tf.eye(4, batch_shape=[eps_tensor.shape[0]], dtype=tf.complex64)
        eigenvalues_diag = eye_matrix * tf.expand_dims(eigenvalues, axis=-1)

        # Compute partial using the exponential function
        k0_expanded = tf.expand_dims(tf.expand_dims(k0, axis=-1), axis=-1)
        partial = tf.linalg.expm(1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

    return partial


@run_on_device
def layer_matrix_incidence_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex64), semi_infinite = False):
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

    berreman_matrix = tf.transpose(berreman_matrix, perm=[1, 2, 0])
    # berreman_matrix = tf.transpose(berreman_matrix, perm=[2, 1, 3, 0])


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

    else:
        eigenvectors = eigenvectors[:, tf.newaxis, :, :]

        # Propagation part
        # Create diagonal matrix with eigenvalues
        eye_matrix = tf.eye(4, batch_shape=[eigenvalues.shape[0]], dtype=tf.complex64)

        eigenvalues_diag = (eye_matrix * eigenvalues[:, :, tf.newaxis])[:,tf.newaxis,:,:]
        
        # Compute partial using the exponential function
        k0_expanded = k0[tf.newaxis, :, tf.newaxis, tf.newaxis]

        partial = tf.linalg.expm(1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = tf.transpose(eigenvectors @ partial @ tf.linalg.inv(eigenvectors), perm = [1,0,2,3])
    
    return partial


@run_on_device
def layer_matrix_anisotropy_one_axis_rotation_tensorflow(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex64), semi_infinite = False):

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


@run_on_device
def berreman_all_anisotropy(kx, eps_tensor, mu_tensor, k0 = None, thickness = tf.constant(0.5e-4, dtype=tf.complex64), semi_infinite = False):

    berreman_matrix = tf.transpose(tf.stack([
        [
        - kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2],
        kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])),
        (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2])) * tf.ones_like(kx),
        mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]
        ],
        [
        tf.zeros_like(- kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
        -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2],
        ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0]) * tf.ones_like(kx),
        ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1]) * tf.ones_like(kx)
        ],
        [
        ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0]) * tf.ones_like(kx),
        (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]),
        -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2],
        kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))
        ],
        [
        (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])) * tf.ones_like(kx),
        (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])) * tf.ones_like(kx),
        tf.zeros_like(- kx * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]),
        -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]
        ]
    ], axis=-1), perm = [2,1,3,4,5,6,0])

    eigenvalues, eigenvectors = tf.linalg.eig(berreman_matrix)

    del berreman_matrix
    
    if semi_infinite:
        # Sort indices of eigenvalues in descending order
        sorted_indices = tf.argsort(tf.math.imag(eigenvalues), axis=-1, direction='DESCENDING')

        # Reorder eigenvectors using sorted_indices
        ordered_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=-1, batch_dims=5)

        # Replace the third column with the second column and set columns 2 and 4 to 0
        return tf.stack([ordered_eigenvectors[:,:,:,:,:, :,0], tf.zeros_like(ordered_eigenvectors[:,:,:,:,:,:,1]),
                                    ordered_eigenvectors[:,:,:,:,:,:,1], tf.zeros_like(ordered_eigenvectors[:,:,:,:,:,:,3])], axis=-1)
        
    else:
        # Propagation part
        # Create diagonal matrix with eigenvalues
        eye_matrix = tf.eye(4, batch_shape=[eps_tensor.shape[0]], dtype=tf.complex64)
        eigenvalues_diag = eye_matrix * tf.expand_dims(eigenvalues, axis=-1)

        # Compute partial using the exponential function
        k0_expanded = tf.expand_dims(tf.expand_dims(k0, axis=-1), axis=-1)
        partial = tf.linalg.expm(1j * eigenvalues_diag * k0_expanded * thickness)

        # Compute partial_complete using the @ symbol for matrix multiplication
        partial = eigenvectors @ partial @ tf.linalg.inv(eigenvectors)

        exit()

    return 0.

@run_on_device
def main_incident_one_axis_anisotropy():

    eps_prism = 5.5
    incident_angle = tf.linspace(-tf.constant(m.pi, dtype=tf.float32) / 2, tf.constant(m.pi, dtype=tf.float32) / 2, 180)
    
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype = tf.complex64)

    quartz = Quartz(frequency_length=200, run_on_device_decorator=run_on_device)
    k0 = quartz.frequency * 2. * m.pi
    
    ext, ord = quartz.permittivity_fetch()
    eps_tensor = quartz.fetch_permittivity_tensor()

    x_rotation = tf.constant(0., dtype = tf.complex64)
    y_rotation = tf.cast(tf.linspace(0.,m.pi/2.,100), dtype = tf.complex64)
    z_rotation = tf.constant(m.radians(340), dtype = tf.complex64)

    eps_tensor = anisotropy_rotation_one_axis(eps_tensor, x_rotation, y_rotation, z_rotation)
    non_magnetic_tensor = Air(run_on_device_decorator=run_on_device).construct_tensor_singular() * tf.ones_like(eps_tensor)

    prism_layer = Ambient_Incident_Prism(eps_prism, incident_angle, run_on_device_decorator=run_on_device).construct_tensor()
    # # air_layer = tf.linalg.inv(layer_matrix_one_theta_tensorflow(kx, non_magnetic_tensor, non_magnetic_tensor, k0, thickness = 1.5e-4))
    kx = kx[:, tf.newaxis, tf.newaxis]
    eps_tensor = tf.expand_dims(eps_tensor, axis=0)
    non_magnetic_tensor = tf.expand_dims(non_magnetic_tensor, axis=0)

    quartz_layer = layer_matrix_anisotropy_one_axis_rotation_tensorflow(kx, eps_tensor, non_magnetic_tensor, semi_infinite=True)
    prism_layer = tf.expand_dims(prism_layer, axis=1)
    prism_layer = tf.expand_dims(prism_layer, axis=0)

    T = tf.matmul(prism_layer, quartz_layer)
    r = reflection_coefficients(T)
    print(r.shape)

    contour_plot('theta', r[:,:,:,99].numpy(),  quartz.frequency.numpy().real, incident_angle.numpy().real, 0., None, rotation_x = 0., rotation_y = 0., rotation_z = 0.)


@run_on_device
def main_all_anisotropy_axes():

    eps_prism = 5.5
    incident_angle = tf.linspace(-tf.constant(m.pi, dtype=tf.float32) / 2, tf.constant(m.pi, dtype=tf.float32) / 2, 110)
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype = tf.complex64)
    distance = 1.5e-4

    quartz = Quartz(frequency_length=140, run_on_device_decorator=run_on_device)
    
    k0 = quartz.frequency * 2. * m.pi
    eps_tensor = quartz.fetch_permittivity_tensor()

    x_rotation = tf.cast(tf.linspace(0.,np.pi/2.,2), dtype = tf.complex64)
    y_rotation = tf.cast(tf.linspace(0.,np.pi/2.,45), dtype = tf.complex64)
    z_rotation = tf.cast(tf.linspace(0.,2 * np.pi,60), dtype = tf.complex64)

    eps_tensor = anisotropy_rotation_all_axes(eps_tensor, x_rotation, y_rotation, z_rotation)[tf.newaxis, ...]

    non_magnetic_tensor = (Air(run_on_device_decorator=run_on_device).construct_tensor_singular())
    
    air_layer = (tf.linalg.inv(layer_matrix_incidence_tensorflow(kx, non_magnetic_tensor, non_magnetic_tensor, k0, thickness = distance)))[:,:,tf.newaxis, tf.newaxis, tf.newaxis, :, :]


    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

    quartz_layer = berreman_all_anisotropy(kx, eps_tensor, non_magnetic_tensor * tf.ones_like(eps_tensor), semi_infinite=True)

    prism_layer = Ambient_Incident_Prism(eps_prism, incident_angle, run_on_device_decorator=run_on_device).construct_tensor()[tf.newaxis,:, tf.newaxis,tf.newaxis, tf.newaxis, :, :]
    

    T = prism_layer @ air_layer @ quartz_layer

    del quartz_layer, air_layer, prism_layer, kx, eps_tensor, non_magnetic_tensor
    
    r = reflection_coefficients(T)

    all_axis_plot(r.numpy(), incident_angle.numpy().real, quartz.frequency.numpy().real, x_rotation.numpy().real, y_rotation.numpy().real, z_rotation.numpy().real, distance)

    # azimuthal_slider_plot(r.numpy(), incident_angle.numpy().real, quartz.frequency.numpy().real, x_rotation.numpy().real, y_rotation.numpy().real, z_rotation.numpy().real, distance)

if __name__ == '__main__':
    # main_incident_one_axis_anisotropy()
    main_all_anisotropy_axes()

