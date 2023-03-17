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


def contour_theta(wavenumber, x_axis, distance, anisotropy_rotation_x, rotation_z, reflectivities):
    
    reflectivities = (reflectivities * np.conj(reflectivities)).real
    R_pp = reflectivities[0]
    R_ps = reflectivities[1]
    R_sp = reflectivities[2]
    R_ss = reflectivities[3]
    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    x_axis = np.degrees(x_axis)
    fig, ax = plt.subplots(2,3, figsize=(12, 7))
    fig.suptitle("ATR for $\phi_x$ = " + str(int(round(np.degrees(anisotropy_rotation_x),1))) + "$^\circ$, $\phi_z$ = " + str(int(round(np.degrees(rotation_z),1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")

    reflection_pp = ax[0,0].pcolormesh(x_axis, wavenumber, R_pp, cmap = 'magma')
    cbar_pp = plt.colorbar(reflection_pp, ax = ax[0,0])
    cbar_pp.mappable.set_clim(0., 1.)
    cbar_pp.set_label("$|r_{pp}|^2$")
    ax[0,0].set_title("$|r_{pp}|^2$")
    ax[0,0].set_xlabel('$k_x / k_0 $')
    ax[0,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ps = ax[0,1].pcolormesh(x_axis, wavenumber, R_ps, cmap = 'magma')
    cbar_ps = plt.colorbar(reflection_ps, ax = ax[0,1])
    cbar_ps.mappable.set_clim(0., 1.)
    cbar_ps.set_label("$|r_{ps}|^2$")
    ax[0,1].set_title("$|r_{ps}|^2$")
    ax[0,1].set_xlabel('$k_x / k_0 $')
    ax[0,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_pp_total = ax[0,2].pcolormesh(x_axis, wavenumber, R_pp_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_pp_total, ax = ax[0,2])
    cbar_pp_total.set_label("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_title("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_xlabel('$k_x / k_0 $')
    ax[0,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_sp = ax[1,0].pcolormesh(x_axis, wavenumber, R_sp, cmap = 'magma')
    cbar_sp = plt.colorbar(reflection_sp, ax = ax[1,0])
    cbar_sp.mappable.set_clim(0., 1.)
    cbar_sp.set_label("$|r_{sp}|^2$")
    ax[1,0].set_title("$|r_{sp}|^2$")
    ax[1,0].set_xlabel('$k_x / k_0 $')
    ax[1,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss = ax[1,1].pcolormesh(x_axis, wavenumber, R_ss, cmap = 'magma')
    cbar_ss = plt.colorbar(reflection_ss, ax = ax[1,1])
    cbar_ss.mappable.set_clim(0., 1.)
    cbar_ss.set_label("$|r_{ss}|^2$")
    ax[1,1].set_title("$|r_{ss}|^2$")
    ax[1,1].set_xlabel('$k_x / k_0 $')
    ax[1,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss_total = ax[1,2].pcolormesh(x_axis, wavenumber, R_ss_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_ss_total, ax = ax[1,2])
    cbar_pp_total.set_label("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_title("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_xlabel('$k_x / k_0 $')
    ax[1,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')
    
    plt.tight_layout()
    plt.show()
    plt.close()
