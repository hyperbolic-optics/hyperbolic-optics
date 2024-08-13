import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.colors as color


def plot_permittivity(material, eps_ext, eps_ord):
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        }
    )

    fig, axs = plt.subplots(
        2, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.1}
    )

    # Plot real part of permittivity
    axs[0].plot(
        material.frequency,
        tf.math.real(eps_ext),
        label=r"$\mathrm{Re}(\varepsilon_\mathrm{ext})$",
    )
    axs[0].plot(
        material.frequency,
        tf.math.real(eps_ord),
        label=r"$\mathrm{Re}(\varepsilon_\mathrm{ord})$",
    )
    axs[0].axhline(y=0, color="black", linewidth=1)
    axs[0].set(ylabel=r"$\mathrm{Re}(\epsilon)$")
    axs[0].legend()

    # Plot imaginary part of permittivity
    axs[1].plot(
        material.frequency,
        tf.math.imag(eps_ext),
        label=r"$\mathrm{Im}(\varepsilon_\mathrm{ext})$",
    )
    axs[1].plot(
        material.frequency,
        tf.math.imag(eps_ord),
        label=r"$\mathrm{Im}(\varepsilon_\mathrm{ord})$",
    )
    axs[1].set(xlabel=r"Wavenumber (cm$^{-1})$", ylabel=r"$\mathrm{Im}(\epsilon)$")
    axs[1].set_xlim(material.frequency[0].numpy(), material.frequency[-1].numpy())
    axs[1].set_ylim(
        0,
    )
    axs[1].legend()

    # Save and show figure
    plt.savefig(material.name + "Spectrum.png", dpi=300, bbox_inches="tight")
    plt.subplots_adjust(hspace=0.1)
    plt.show()
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def contour_plot_mueller_incidence(structure, param, title=None):
    x_axis = np.round(np.degrees(structure.incident_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(x_axis, frequency, param, cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    if title:
        cbar.set_label(title)
        ax.set_title(title)
    
    ax.set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
    ax.set_xlabel("Incident Angle / $^\circ$")
    ax.set_ylabel("$\omega/2\pi c$ (cm$^{-1}$)")

    plt.tight_layout()
    plt.show()

def contour_plot_mueller_azimuthal(structure, param, title=None):
    x_axis = np.round(np.degrees(structure.azimuthal_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(x_axis, frequency, param, cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    if title:
        cbar.set_label(title)
        ax.set_title(title)
    
    ax.set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
    ax.set_xlabel("Azimuthal Rotation / $^\circ$")
    ax.set_ylabel("$\omega/2\pi c$ (cm$^{-1}$)")

    plt.tight_layout()
    plt.show()

def contour_plot_mueller_dispersion(structure, param, title=None):
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    im = ax.pcolormesh(z_rotation, incident_angle, param, cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    if title:
        cbar.set_label(title)
        ax.set_title(title)
    
    ax.set_xticks(np.linspace(0, 2*np.pi, 5))
    ax.set_xticklabels(['0', '90', '180', '270', '360'])
    ax.set_yticklabels("")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    plt.tight_layout()
    plt.show()

def contour_plot_simple_incidence(structure, params):
    x_axis = np.round(np.degrees(structure.incident_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    ax_to_plot = [
        (params['S0'], "S0", 0, 0),
        (params['S1'], "S1", 0, 1),
        (params['S2'], "S2", 0, 2),
        (params['S3'], "S3", 1, 0),
        (params['DOP'], "DOP", 1, 1),
        (params['Ellipticity'], "Ellipticity", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap="magma")
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        ax[row, col].set_xlabel("Incident Angle / $^\circ$")
        ax[row, col].set_ylabel("$\omega/2\pi c$ (cm$^{-1}$)")

    plt.tight_layout()
    plt.show()

def contour_plot_simple_azimuthal(structure, params):
    x_axis = np.round(np.degrees(structure.azimuthal_angle.numpy().real), 1)
    frequency = structure.frequency.numpy().real

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    ax_to_plot = [
        (params['S0'], "S0", 0, 0),
        (params['S1'], "S1", 0, 1),
        (params['S2'], "S2", 0, 2),
        (params['S3'], "S3", 1, 0),
        (params['DOP'], "DOP", 1, 1),
        (params['Ellipticity'], "Ellipticity", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap="magma")
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        ax[row, col].set_xlabel("Azimuthal Rotation / $^\circ$")
        ax[row, col].set_ylabel("$\omega/2\pi c$ (cm$^{-1}$)")

    plt.tight_layout()
    plt.show()

def contour_plot_simple_dispersion(structure, params):
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real

    fig, ax = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    
    ax_to_plot = [
        (params['S0'], "S0", 0, 0),
        (params['S1'], "S1", 0, 1),
        (params['S2'], "S2", 0, 2),
        (params['S3'], "S3", 1, 0),
        (params['DOP'], "DOP", 1, 1),
        (params['Ellipticity'], "Ellipticity", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(z_rotation, incident_angle, data, cmap="magma")
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(0, 2*np.pi, 5))
        ax[row, col].set_xticklabels(['0', '90', '180', '270', '360'])
        ax[row, col].set_yticklabels("")
        ax[row, col].set_xlabel("")
        ax[row, col].set_ylabel("")
        ax[row, col].grid(False)

    plt.tight_layout()
    plt.show()