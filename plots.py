import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.default': 'regular' })


def plot_base_permittivity(wavenumber, eps_ext, eps_ord):
    plt.rcParams["figure.figsize"] = (5,4)
    fig, axs = plt.subplots(2)
    fig.suptitle('Permittivities')
    axs[0].plot(wavenumber, eps_ext.real)
    axs[0].plot(wavenumber, eps_ext.imag)
    axs[0].set(xlabel='Wavenumber', ylabel = 'Extraordinary')

    axs[1].plot(wavenumber, eps_ord.real)
    axs[1].plot(wavenumber, eps_ord.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'Ordinary')
    
    plt.tight_layout()
    plt.show()


def contour_theta(frequency, x_axis, distance, anisotropy_rotation_y, rotation_z, rotation_x, reflectivities):
    
    reflectivities = (reflectivities * np.conj(reflectivities)).real
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2,3, figsize=(12, 7))
    fig.suptitle("ATR for $\phi_y$ = " + str(int(round(np.degrees(anisotropy_rotation_y),1))) + "$^\circ$, $\phi_z$ = " + str(int(round(np.degrees(rotation_z),1))) + "$^\circ$, $\phi_x$ = " + str(int(round(np.degrees(rotation_x),1))) + "$d = " + str(round(distance * 1e4, 3)) +"\mu m$")

    reflection_pp = ax[0,0].pcolormesh(x_axis, frequency, R_pp, cmap = 'magma')
    cbar_pp = plt.colorbar(reflection_pp, ax = ax[0,0])
    cbar_pp.mappable.set_clim(0., 1.)
    cbar_pp.set_label("$|r_{pp}|^2$")
    ax[0,0].set_title("$|r_{pp}|^2$")
    ax[0,0].set_xlabel('$k_x / k_0 $')
    ax[0,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ps = ax[0,1].pcolormesh(x_axis, frequency, R_ps, cmap = 'magma')
    cbar_ps = plt.colorbar(reflection_ps, ax = ax[0,1])
    cbar_ps.mappable.set_clim(0., 1.)
    cbar_ps.set_label("$|r_{ps}|^2$")
    ax[0,1].set_title("$|r_{ps}|^2$")
    ax[0,1].set_xlabel('$k_x / k_0 $')
    ax[0,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_pp_total = ax[0,2].pcolormesh(x_axis, frequency, R_pp_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_pp_total, ax = ax[0,2])
    cbar_pp_total.set_label("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_title("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_xlabel('$k_x / k_0 $')
    ax[0,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_sp = ax[1,0].pcolormesh(x_axis, frequency, R_sp, cmap = 'magma')
    cbar_sp = plt.colorbar(reflection_sp, ax = ax[1,0])
    cbar_sp.mappable.set_clim(0., 1.)
    cbar_sp.set_label("$|r_{sp}|^2$")
    ax[1,0].set_title("$|r_{sp}|^2$")
    ax[1,0].set_xlabel('$k_x / k_0 $')
    ax[1,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss = ax[1,1].pcolormesh(x_axis, frequency, R_ss, cmap = 'magma')
    cbar_ss = plt.colorbar(reflection_ss, ax = ax[1,1])
    cbar_ss.mappable.set_clim(0., 1.)
    cbar_ss.set_label("$|r_{ss}|^2$")
    ax[1,1].set_title("$|r_{ss}|^2$")
    ax[1,1].set_xlabel('$k_x / k_0 $')
    ax[1,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss_total = ax[1,2].pcolormesh(x_axis, frequency, R_ss_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_ss_total, ax = ax[1,2])
    cbar_pp_total.set_label("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_title("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_xlabel('$k_x / k_0 $')
    ax[1,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')
    
    plt.tight_layout()
    plt.show()
    plt.close()


def contour_azimuth(frequency, x_axis, incident_angle, distance, anisotropy_rotation_y, rotation_x, reflectivities):

    reflectivities = (reflectivities * np.conj(reflectivities)).real
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("ATR for $\phi_y$ = " + str(int(round(np.degrees(anisotropy_rotation_y), 1))) + "$^\circ$, $\\theta$ = " + str(int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $\phi_x$ = " + str(int(round(np.degrees(rotation_x), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(np.round(np.degrees(x_axis),1), frequency, data.T, cmap='magma')
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.mappable.set_clim(0., 1.)
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xlabel('Azimuth rotation / $^\circ$')
        ax[row, col].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    plt.tight_layout()
    plt.show()
    plt.close()


def contour_y_anisotropy(frequency, x_axis, distance, incident_angle, rotation_z, rotation_x, reflectivities):

    reflectivities = (reflectivities * np.conj(reflectivities)).real
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("ATR for $\phi_z$ = " + str(int(round(np.degrees(rotation_z), 1))) + "$^\circ$, $\\theta$ = " + str(int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $\phi_x$ = " + str(int(round(np.degrees(rotation_x), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(np.round(np.degrees(x_axis), 1), frequency, data.T, cmap='magma')
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.mappable.set_clim(0., 1.)
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xlabel('Y-Anisotropy Rotation / $^\circ$')
        ax[row, col].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    plt.tight_layout()
    plt.show()
    plt.close()


def contour_x_anisotropy(frequency, x_axis, distance, incident_angle, rotation_y, rotation_z, reflectivities):

    reflectivities = (reflectivities * np.conj(reflectivities)).real
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("ATR for $\phi_z$ = " + str(int(round(np.degrees(rotation_z), 1))) + "$^\circ$,$\\phi_y$ = " + str(int(round(np.degrees(rotation_y), 1))) + "$^\circ$, $\\theta$ = " + str(int(round(np.degrees(incident_angle), 1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) + "\mu m$")

    ax_to_plot = [
        (R_pp, "$|r_{pp}|^2$", 0, 0),
        (R_ps, "$|r_{ps}|^2$", 0, 1),
        (R_pp_total, "$|r_{pp}|^2 + |r_{ps}|^2$", 0, 2),
        (R_sp, "$|r_{sp}|^2$", 1, 0),
        (R_ss, "$|r_{ss}|^2$", 1, 1),
        (R_ss_total, "$|r_{ss}|^2 + |r_{sp}|^2$", 1, 2),
    ]

    for data, title, row, col in ax_to_plot:
        im = ax[row, col].pcolormesh(np.round(np.degrees(x_axis), 1), frequency, data.T, cmap='magma')
        cbar = plt.colorbar(im, ax=ax[row, col])
        cbar.mappable.set_clim(0., 1.)
        cbar.set_label(title)
        ax[row, col].set_title(title)
        ax[row, col].set_xlabel('X-Anisotropy Rotation / $^\circ$')
        ax[row, col].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    plt.tight_layout()
    plt.show()
    plt.close()
