import math as m

import tensorflow as tf

from anisotropy_utils import (anisotropy_rotation_all_axes,
                              anisotropy_rotation_one_axis)
from berreman import (berreman_all_anisotropy, berreman_incidence,
                      reflection_coefficients)
from device_config import run_on_device
from material_params import Air, Ambient_Incident_Prism, Sapphire
from plots import all_axis_plot, azimuthal_slider_plot


@run_on_device
def main_all_anisotropy_axes():
    eps_prism = 5.5
    incident_angle = tf.linspace(
        -tf.constant(0, dtype=tf.float32) / 2,
        tf.constant(m.pi, dtype=tf.float32) / 2,
        30,
    )
    kx = tf.cast(tf.sqrt(eps_prism) * tf.sin(incident_angle), dtype=tf.complex64)

    sapphire = Sapphire(frequency_length=300, run_on_device_decorator=run_on_device)

    k0 = sapphire.frequency * 2.0 * m.pi
    eps_tensor = sapphire.fetch_permittivity_tensor()

    air_gap_thickness = tf.cast(tf.linspace(0.0e-4, 2.5e-4, 6), dtype=tf.complex64)
    x_rotation = tf.cast(tf.linspace(0.0, 2 * m.pi, 2), dtype=tf.complex64)
    y_rotation = tf.cast(tf.linspace(0.0, m.pi / 2.0, 9), dtype=tf.complex64)
    z_rotation = tf.cast(tf.linspace(0.0, 2 * m.pi, 180), dtype=tf.complex64)

    eps_tensor = anisotropy_rotation_all_axes(
        eps_tensor, x_rotation, y_rotation, z_rotation
    )[tf.newaxis, ...]

    non_magnetic_tensor = Air(
        run_on_device_decorator=run_on_device
    ).construct_tensor_singular()

    air_layer = (
        tf.linalg.inv(
            berreman_incidence(
                kx,
                non_magnetic_tensor,
                non_magnetic_tensor,
                k0,
                thickness=air_gap_thickness,
            )
        )
    )[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    kx = kx[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]

    sapphire_layer = berreman_all_anisotropy(
        kx,
        eps_tensor,
        non_magnetic_tensor * tf.ones_like(eps_tensor),
        semi_infinite=True,
    )[tf.newaxis, ...]

    prism_layer = Ambient_Incident_Prism(
        eps_prism, incident_angle, run_on_device_decorator=run_on_device
    ).construct_tensor()[
        tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis, :, :
    ]

    T = prism_layer @ air_layer @ sapphire_layer
    del sapphire_layer, air_layer, prism_layer, kx, eps_tensor, non_magnetic_tensor

    r = reflection_coefficients(T)

    # all_axis_plot(r.numpy(), incident_angle.numpy().real, sapphire, x_rotation.numpy().real, y_rotation.numpy().real, z_rotation.numpy().real, air_gap_thickness.numpy())

    azimuthal_slider_plot(
        r.numpy(),
        incident_angle.numpy().real,
        sapphire,
        x_rotation.numpy().real,
        y_rotation.numpy().real,
        z_rotation.numpy().real,
        air_gap_thickness.numpy(),
    )


if __name__ == "__main__":
    # main_incident_one_axis_anisotropy()
    main_all_anisotropy_axes()
