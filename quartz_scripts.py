import tensorflow as tf
import math as m

from device_config import run_on_device
from material_params import Quartz, Ambient_Incident_Prism, Air
from anisotropy_utils import anisotropy_rotation_one_axis, anisotropy_rotation_all_axes
from plots import contour_plot, all_axis_plot, azimuthal_slider_plot
from berreman import reflection_coefficients, berreman_all_anisotropy, berreman_incidence_one_anisotropy, berreman_one_angle, berreman_incidence


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

    quartz_layer = berreman_incidence_one_anisotropy(kx, eps_tensor, non_magnetic_tensor, semi_infinite=True)
    prism_layer = tf.expand_dims(prism_layer, axis=1)
    prism_layer = tf.expand_dims(prism_layer, axis=0)

    T = tf.matmul(prism_layer, quartz_layer)
    r = reflection_coefficients(T)
    print(r.shape)

    contour_plot('theta', r[:,:,:,99].numpy(),  quartz.frequency.numpy().real, incident_angle.numpy().real, 0., None, rotation_x = 0., rotation_y = 0., rotation_z = 0.)

@run_on_device
def main_all_anisotropy_axes():

    eps_prism = 5.5
    incident_angle = tf.linspace(-tf.constant(m.pi, dtype=tf.float32) / 2, tf.constant(m.pi, dtype=tf.float32) / 2, 40)
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype = tf.complex64)

    quartz = Quartz(frequency_length=90, run_on_device_decorator=run_on_device)
    
    k0 = quartz.frequency * 2. * m.pi
    eps_tensor = quartz.fetch_permittivity_tensor()

    air_gap_thickness = tf.cast(tf.linspace(0.,3.e-4, 10), dtype = tf.complex64)
    x_rotation = tf.cast(tf.linspace(0.,2 * m.pi,2), dtype = tf.complex64)
    y_rotation = tf.cast(tf.linspace(0.,2 * m.pi,20), dtype = tf.complex64)
    z_rotation = tf.cast(tf.linspace(0.,2 * m.pi,100), dtype = tf.complex64)

    eps_tensor = anisotropy_rotation_all_axes(eps_tensor, x_rotation, y_rotation, z_rotation)[tf.newaxis, ...]

    non_magnetic_tensor = (Air(run_on_device_decorator=run_on_device).construct_tensor_singular())
    
    air_layer = (tf.linalg.inv(berreman_incidence(kx, non_magnetic_tensor, non_magnetic_tensor, k0, thickness = air_gap_thickness)))[:,:,:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

    quartz_layer = berreman_all_anisotropy(kx, eps_tensor, non_magnetic_tensor * tf.ones_like(eps_tensor), semi_infinite=True)[tf.newaxis, ...]

    prism_layer = Ambient_Incident_Prism(eps_prism, incident_angle, run_on_device_decorator=run_on_device).construct_tensor()[tf.newaxis, tf.newaxis,:, tf.newaxis,tf.newaxis, tf.newaxis, :, :]
    

    T = prism_layer @ air_layer @ quartz_layer
    del quartz_layer, air_layer, prism_layer, kx, eps_tensor, non_magnetic_tensor
    
    r = reflection_coefficients(T)

    # all_axis_plot(r.numpy(), incident_angle.numpy().real, quartz.frequency.numpy().real, x_rotation.numpy().real, y_rotation.numpy().real, z_rotation.numpy().real, air_gap_thickness.numpy())

    azimuthal_slider_plot(r.numpy(), incident_angle.numpy().real, quartz.frequency.numpy().real, x_rotation.numpy().real, y_rotation.numpy().real, z_rotation.numpy().real, air_gap_thickness.numpy())

if __name__ == '__main__':
    # main_incident_one_axis_anisotropy()
    main_all_anisotropy_axes()
