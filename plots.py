import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.widgets import Slider

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


def contour_plot(plot_type, reflectivities, frequency, x_axis, distance, incident_angle, rotation_x, rotation_y, rotation_z):
    
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


def all_axis_plot(reflectivities, incident_angle, frequency, rotation_x, rotation_y, rotation_z, distance = 0.):

    incident_angle = np.round(np.degrees(incident_angle),1)
    reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    # Create a function to update the plots
    def update(_):
        rotation_x_val = rotation_x_slider.val
        rotation_y_val = rotation_y_slider.val
        rotation_z_val = rotation_z_slider.val
        index_x = np.argmin(np.abs(rotation_x - rotation_x_val))
        index_y = np.argmin(np.abs(rotation_y - rotation_y_val))
        index_z = np.argmin(np.abs(rotation_z - rotation_z_val))

        data_list = [R_pp, R_ps, R_pp_total, R_sp, R_ss, R_ss_total]

        title = (
        "ATR for $\phi_y = "
        + str(int(round(np.degrees(rotation_y_slider.val), 1)))
        + "^\circ, \phi_z = "
        + str(int(round(np.degrees(rotation_z_slider.val), 1)))
        + "^\circ, \phi_x = "
        + str(int(round(np.degrees(rotation_x_slider.val), 1)))
        + "^\circ, d = "
        + str(round(distance * 1e4, 3))
        + "\mu m$"
        )
        fig.suptitle(title)

        for i, (_, title, row, col) in enumerate(ax_to_plot):
            new_data = data_list[i][:, :, index_x, index_y, index_z]
            im = ax[row, col].collections[0]
            im.set_array(new_data.ravel())
            im.set_clim(vmin=0, vmax=new_data.max())  # Update the colorbar limits
            colorbar_list[i].mappable.set_clim(0,)  # Update the colorbar limits correctly
            colorbar_list[i].draw_all()

    # Your existing plot code
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
  
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9, hspace=0.5, wspace=0.4)

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    colorbar_list = []
    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(incident_angle, frequency, data[:, :, 0, 0, 0], cmap='magma')
        cbar = plt.colorbar(im, ax=ax[row, col])
        colorbar_list.append(cbar)  # Store the colorbar in a list
        cbar.mappable.set_clim(0.,)
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(incident_angle.min(), incident_angle.max(), 5))
        ax[row, col].set_xlabel('Incident Angle / $^\circ$')
        ax[row, col].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    slider_x_ax = plt.axes([0.2, 0.15, 0.6, 0.02])
    slider_y_ax = plt.axes([0.2, 0.11, 0.6, 0.02])
    slider_z_ax = plt.axes([0.2, 0.07, 0.6, 0.02])
    
    rotation_x_slider = Slider(slider_x_ax, 'Rotation X', rotation_x.min(), rotation_x.max(), valinit=rotation_x[0], valstep=np.diff(rotation_x).mean())
    rotation_y_slider = Slider(slider_y_ax, 'Rotation Y', rotation_y.min(), rotation_y.max(), valinit=rotation_y[0], valstep=np.diff(rotation_y).mean())
    rotation_z_slider = Slider(slider_z_ax, 'Rotation Z', rotation_z.min(), rotation_z.max(), valinit=rotation_z[0], valstep=np.diff(rotation_z).mean())

    # Connect sliders to the update function
    rotation_x_slider.on_changed(update)
    rotation_y_slider.on_changed(update)
    rotation_z_slider.on_changed(update)

    update(None)

    # plt.tight_layout()
    plt.show()
    plt.close()