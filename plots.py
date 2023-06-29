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
        2, figsize=(12, 5), sharex=True, gridspec_kw={"hspace": 0.1}
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


def contour_plot_simple_incidence(structure):

    x_axis = np.round(np.degrees(structure.incident_angle), 1)
    frequency = structure.frequency.numpy().real
    reflectivities = [structure.r_pp,
                      structure.r_ps,
                      structure.r_sp,
                      structure.r_ss]

    reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
    # reflectivities = np.round(np.asarray(reflectivities).imag, 6)
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    
    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap="magma")
                                     #norm = color.SymLogNorm(vmin=data.min(), vmax=data.max(), linthresh = 0.01))
        cbar = plt.colorbar(im, ax=ax[row, col])
        # cbar.mappable.set_clim(
        #     0.0,
        # )
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        ax[row, col].set_xlabel("Incident Angle / $^\circ$")
        ax[row, col].set_ylabel("$\omega/2\pi c (cm^{-1})$")

    plt.tight_layout()
    plt.show()
    plt.close()


def contour_plot_simple_azimuthal(structure):
    
    x_axis = np.round(np.degrees(structure.azimuthal_angle), 1)
    frequency = structure.frequency.numpy().real
    reflectivities = np.asarray([structure.r_pp, structure.r_ps,
                      structure.r_sp, structure.r_ss])

    reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(x_axis, frequency, data, cmap="magma")
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.mappable.set_clim(
            0.0,
        )
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        ax[row, col].set_xlabel("Azimuthal Rotation / $^\circ$")
        ax[row, col].set_ylabel("$\omega/2\pi c (cm^{-1})$")

    plt.tight_layout()
    plt.show()
    plt.close()


def contour_plot_simple_dispersion(structure):
    
    reflectivities = np.asarray([structure.r_pp,
                      structure.r_ps,
                      structure.r_sp,
                      structure.r_ss])
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real

    reflectivities = np.round((reflectivities * np.conj(reflectivities)), 6).real

    # reflectivities = np.round(reflectivities.imag, 6)

    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp


    fig, ax = plt.subplots(2, 3, figsize=(18, 7), subplot_kw= dict(projection = 'polar'))
    plt.subplots_adjust(
        left=0.15, right=0.9, bottom=0.25, top=0.85, hspace=0.5, wspace=0.4
    )

    ax_to_plot = [
        (R_pp, "$|R_{pp}|^2$", 0, 0),
        (R_ps, "$|R_{ps}|^2$", 0, 1),
        (R_pp_total, "$|R_{pp}|^2 + |R_{ps}|^2$", 0, 2),
        (R_sp, "$|R_{sp}|^2$", 1, 0),
        (R_ss, "$|R_{ss}|^2$", 1, 1),
        (R_ss_total, "$|R_{ss}|^2 + |R_{sp}|^2$", 1, 2),
    ]

    colorbar_list = []
    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(
            z_rotation, incident_angle,  data, cmap="magma")
        cbar = plt.colorbar(im, ax=ax[row, col])
        colorbar_list.append(cbar)
        # cbar.mappable.set_clim(
        #     0,
        # )
        cbar.set_label(title)
        ax[row, col].set_title(title)
        # Adjust labels and ticks
        ax[row, col].set_xticks(np.linspace(0, 2*np.pi, 5))
        ax[row, col].set_xticklabels(['0', '90', '180', '270', '360'])  # azimuthal rotation in degrees
        ax[row, col].set_yticklabels("")  # incident angle in degrees

        # Remove the labels, polar coordinates speak for themselves
        ax[row, col].set_xlabel("")
        ax[row, col].set_ylabel("")
    
    plt.tight_layout()
    plt.show()
    plt.close()