import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams.update({'mathtext.default': 'regular' })


def plot_base_permittivity(wavenumber, eps_ext, eps_ord):
    fig, axs = plt.subplots(2, figsize=(5, 4))
    fig.suptitle('Permittivities')
    axs[0].plot(wavenumber, tf.math.real(eps_ext))
    axs[0].plot(wavenumber, tf.math.imag(eps_ext))
    axs[0].set(xlabel='Wavenumber', ylabel = 'Extraordinary')

    axs[1].plot(wavenumber, tf.math.real(eps_ord))
    axs[1].plot(wavenumber, tf.math.imag(eps_ord))
    axs[1].set(xlabel='Wavenumber', ylabel = 'Ordinary')
    
    plt.tight_layout()
    plt.show()


def contour_plot(plot_type, frequency, x_axis, distance, incident_angle, rotation_x, rotation_y, rotation_z, reflectivities):
    
    x_axis = np.round(np.degrees(x_axis),1)

    reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    if plot_type == 'theta':
        fig.suptitle("ATR for $\phi_y = " + str(int(round(np.degrees(rotation_y), 1))) + "^\circ, \phi_z = " + str(
            int(round(np.degrees(rotation_z), 1))) + "^\circ, \phi_x = " + str(
            int(round(np.degrees(rotation_x), 1))) + "^\circ, d = " + str(round(distance * 1e4, 3)) + "\mu m$")
    
    elif plot_type == 'azimuth':
        x_axis -= 90.
        fig.suptitle("ATR for $\phi_y$ = " + str(int(round(np.degrees(rotation_y), 1))) + "$^\circ$, $\\theta$ = " + str(
            int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $\phi_x$ = " + str(
            int(round(np.degrees(rotation_x), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")
    
    elif plot_type == 'y_anisotropy':
        fig.suptitle("ATR for $\phi_z$ = " + str(int(round(np.degrees(rotation_z), 1))) + "$^\circ$, $\\theta$ = " + str(
            int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $\phi_x$ = " + str(
            int(round(np.degrees(rotation_x), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")
    
    elif plot_type == 'x_anisotropy':
        fig.suptitle("ATR for $\phi_z$ = " + str(int(round(np.degrees(rotation_z), 1))) + "$^\circ$, $\phi_y$ = " + str(
            int(round(np.degrees(rotation_y), 1))) + "$^\circ$, $\\theta$ = " + str(
            int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap='magma')
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.mappable.set_clim(0.,)
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        ax[row, col].set_xlabel(plot_type.capitalize() + ' Rotation / $^\circ$')
        ax[row, col].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    plt.tight_layout()
    plt.show()
    plt.close()
