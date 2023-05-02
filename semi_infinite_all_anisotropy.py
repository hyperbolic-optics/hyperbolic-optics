"""
Computing the transfer matrix for a semi infinite anisotropic material
"""
import math as m
import tensorflow as tf
from anisotropy_utils import (anisotropy_rotation_all_axes)
from berreman import (transfer_matrix_wrapper,
                      reflection_coefficients)
from device_config import run_on_device
from material_params import (Air, Ambient_Incident_Prism, CalciteLower,
                             CalciteUpper, Quartz, Sapphire)
from plots import (all_axis_plot, azimuthal_slider_plot)

@run_on_device
def main_all_anisotropy_axes(material_type):
    """
    Computing the transfer matrix for a semi infinite anisotropic material,
      for all possible anisotropy rotations.
    """
    eps_prism = 5.5
    incident_angle = tf.linspace(
        -tf.constant(m.pi, dtype=tf.float32) / 2,
        tf.constant(m.pi, dtype=tf.float32) / 2,
        45,
    )
    k_x = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype=tf.complex64)

    material = material_type(frequency_length=100, run_on_device_decorator=run_on_device)

    k_0 = material.frequency * 2.0 * m.pi
    eps_tensor = material.fetch_permittivity_tensor()

    air_gap_thickness = tf.cast(tf.linspace(0.0e-4, 2.5e-4, 6), dtype=tf.complex64)
    x_rotation = tf.cast(tf.linspace(0.0, m.pi/2., 2), dtype=tf.complex64)
    y_rotation = tf.cast(tf.linspace(0.0, m.pi / 2.0, 15), dtype=tf.complex64)
    z_rotation = tf.cast(tf.linspace(0.0, m.pi/2., 15), dtype=tf.complex64)

    eps_tensor = anisotropy_rotation_all_axes(
        eps_tensor, x_rotation, y_rotation, z_rotation
    )[tf.newaxis, ...]

    non_magnetic_tensor = Air(
        run_on_device_decorator=run_on_device
    ).construct_tensor_singular()

    air_layer = (
        tf.linalg.inv(
            transfer_matrix_wrapper(
                k_x,
                non_magnetic_tensor,
                non_magnetic_tensor,
                k_0,
                thickness=air_gap_thickness,
                mode = "incidence"
            )
        )
    )[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    k_x = k_x[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

    material_layer = transfer_matrix_wrapper(
        k_x,
        eps_tensor,
        non_magnetic_tensor * tf.ones_like(eps_tensor),
        semi_infinite=True,
        mode = "all_anisotropy"
    )[tf.newaxis, ...]

    prism_layer = Ambient_Incident_Prism(
        eps_prism, incident_angle, run_on_device_decorator=run_on_device
    ).construct_tensor()[
        tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis, :, :
    ]

    transfer_matrix = prism_layer @ air_layer @ material_layer
    del material_layer, air_layer, prism_layer, k_x, eps_tensor, non_magnetic_tensor

    reflectivity_values = reflection_coefficients(transfer_matrix)

    all_axis_plot(
        reflectivity_values.numpy(),
        incident_angle.numpy().real,
        material,
        x_rotation.numpy().real,
        y_rotation.numpy().real,
        z_rotation.numpy().real,
        air_gap_thickness.numpy(),
    )


if __name__ == "__main__":

    materials = [CalciteLower, CalciteUpper, Sapphire, Quartz]
    main_all_anisotropy_axes(materials[3])
