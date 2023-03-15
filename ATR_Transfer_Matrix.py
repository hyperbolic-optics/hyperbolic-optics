import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.linalg import expm
from scipy import constants


def permittivity_parameters():
    
    parameters = {
        "ordinary": {
            "high_freq": 2.356,
            "omega_Tn" : np.array([393.5, 450.0, 695.0, 797.0, 1065.0, 1158.0]),
            "gamma_Tn" : np.array([2.1, 4.5, 13.0, 6.9, 7.2, 9.3]),
            "omega_Ln" : np.array([403.0, 507.0, 697.6, 810.0, 1226.0, 1155.0]),
            "gamma_Ln" : np.array([2.8, 3.5, 13.0, 6.9, 12.5, 9.3])
        },
        "extraordinary": {
            "high_freq": 2.383,
            "omega_Tn" : np.array([363.5, 487.5, 777.0, 1071.0]),
            "gamma_Tn" : np.array([4.8, 4.0, 6.7, 6.8]),
            "omega_Ln" : np.array([386.7, 550.0, 790.0, 1229.0]),
            "gamma_Ln" : np.array([7.0, 3.2, 6.7, 12.0])
        }
    }

    return parameters


def permittivity_calc(wavenumber, high_freq, omega_Tn, gamma_Tn, omega_Ln, gamma_Ln):

    top_line = omega_Ln**2. - wavenumber**2. - 1j *  wavenumber * gamma_Ln
    bottom_line =  omega_Tn**2. - wavenumber**2. - 1j * wavenumber * gamma_Tn
    result = top_line / bottom_line
    
    return (high_freq * np.prod(result))


def permittivity_fetch(wavenumber, params):
    E_ext = []
    E_ord = []
    for element in wavenumber:
        E_ext.append(permittivity_calc(element, **params["extraordinary"]))
        E_ord.append(permittivity_calc(element, **params["ordinary"]))
    
    E_ext, E_ord = np.asarray(E_ext), np.asarray(E_ord)
    return E_ext, E_ord


def construct_quartz_tensors(eps_ord, eps_ext):

    permittivity = np.array(
        [
        [eps_ord, np.zeros_like(eps_ext), np.zeros_like(eps_ext)],
        [np.zeros_like(eps_ext), eps_ord, np.zeros_like(eps_ext)],
        [np.zeros_like(eps_ext), np.zeros_like(eps_ext), eps_ext]
        ]
    ).T
    permittivity = np.transpose(permittivity, (0,2,1))

    return permittivity


def magnetic_parameters():
    gamma = 1.05 # cm^-1 / T
    B0 = 0.3 # T
    Ba = 19.745 # T
    Be = 53.313 # T
    magnetisation = 445633.84 #A/cm

    resonant_frequency_squared = gamma**2. * (2. * Ba * Be + Ba ** 2.)
    damping_parameter = 1.27e-4 * np.sqrt(resonant_frequency_squared)

    return gamma, B0, Ba, magnetisation, resonant_frequency_squared, damping_parameter


def fetch_epsilon_mu(frequency, gamma, B0, Ba, magnetisation, resonant_frequency_squared, damping):
    
    X = 1./(resonant_frequency_squared - (frequency + B0 * gamma + 1.j * damping)**2.)
    Y = 1./(resonant_frequency_squared - (frequency - B0 * gamma + 1.j * damping)**2.)

    mu_3 = 1. + constants.mu_0 * gamma**2. * Ba * magnetisation * (X + Y)
    mu_t = constants.mu_0 * gamma**2. * Ba * magnetisation * (X-Y)
    
    magnet_permittivity = 5.5 + 0.j

    return mu_3, mu_t, magnet_permittivity


def construct_magnet_tensors(mu_3, mu_t, epsilon):

    permeability = np.array(
        [
        [mu_3, np.zeros_like(mu_3), - 1.j * mu_t],
        [np.zeros_like(mu_3), np.ones_like(mu_3), np.zeros_like(mu_3)],
        [1.j * mu_t, np.zeros_like(mu_3), mu_3]
        ]
    ).T


    permittivity = np.array(
        [
        [epsilon, 0., 0.],
        [0., epsilon, 0.],
        [0., 0., epsilon]
        ]
    )

    return permeability, permittivity


def anisotropy_rotation_matrix_y(matrix, theta):

    rotation_matrix = np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.],
        [-np.sin(theta), 0., np.cos(theta)]
    ])
    
    return rotation_matrix @ matrix @ rotation_matrix.T


def anisotropy_rotation_matrix_z(matrix, theta):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]
    ])

    return rotation_matrix @ matrix @ rotation_matrix.T


def air_tensor():

    both_tensor = np.array(
        [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ],
    )

    return both_tensor


def ambient_incident_prism(eps_prism, theta):
    n = np.sqrt(eps_prism)

    matrix = 0.5 * np.array([
        [0., 1., -1./(n * np.cos(theta)), 0.],
        [0., 1., 1./(n * np.cos(theta)), 0.],
        [1./np.cos(theta), 0., 0., 1./n],
        [-1./np.cos(theta), 0., 0., 1./n]
    ])

    return matrix


def layer_matrix(eps_tensor, mu_tensor, kx, k0, thickness, magnet = False, quartz = False):

    delta_11 = - kx * (eps_tensor[2,0]/eps_tensor[2,2])
    delta_12 = kx * ((mu_tensor[1,2]/mu_tensor[2,2]) - (eps_tensor[2,1]/eps_tensor[2,2]))
    delta_13 = mu_tensor[1,0] - (mu_tensor[1,2] * mu_tensor[2,0] / mu_tensor[2,2])
    delta_14 = mu_tensor[1,1] - (mu_tensor[1,2] * mu_tensor[2,1] / mu_tensor[2,2]) - (kx**2.)/eps_tensor[2,2]

    delta_21 = 0.
    delta_22 = -kx * mu_tensor[0,2]/mu_tensor[2,2]
    delta_23 = (mu_tensor[0,2] * mu_tensor[2,0] / mu_tensor[2,2]) - mu_tensor[0,0]
    delta_24 = (mu_tensor[0,2] * mu_tensor[2,1] / mu_tensor[2,2]) - mu_tensor[0,1]

    delta_31 = (eps_tensor[1,2] * eps_tensor[2,0] / eps_tensor[2,2]) - eps_tensor[1,0]
    delta_32 = (kx**2.)/mu_tensor[2,2] - eps_tensor[1,1] + (eps_tensor[1,2] * eps_tensor[2,1]/ eps_tensor[2,2])
    delta_33 = -kx * mu_tensor[2,0]/mu_tensor[2,2]
    delta_34 = kx * ((eps_tensor[1,2]/eps_tensor[2,2])-(mu_tensor[2,1]/mu_tensor[2,2]))

    delta_41 = eps_tensor[0,0] - (eps_tensor[0,2] * eps_tensor[2,0] / eps_tensor[2,2])
    delta_42 = eps_tensor[0,1] - (eps_tensor[0,2] * eps_tensor[2,1] / eps_tensor[2,2])
    delta_43 = 0.
    delta_44 = -kx * eps_tensor[0,2]/eps_tensor[2,2]

    delta = np.array(
        [
        [delta_11,delta_12,delta_13,delta_14],
        [delta_21,delta_22,delta_23,delta_24],
        [delta_31,delta_32,delta_33,delta_34],
        [delta_41,delta_42,delta_43,delta_44]
        ]
    )

    eigenvalues, vector = np.linalg.eig(delta)

    if quartz:
        idx = (1.j * eigenvalues).argsort()
        eigenvalues = eigenvalues[idx]
        vector = vector[:,idx]
        return vector.T
    
    if magnet:
        idx = (eigenvalues).argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        vector = vector[:,idx]
        return vector.T

    eigenvalues = np.diag(eigenvalues)
    partial = expm((1.j * eigenvalues * k0 * thickness))
    partial_complete = vector @ partial @ np.linalg.inv(vector)

    return partial_complete


def vector_sort(eigenvectors):

    eigenvector_columns = np.array([
        eigenvectors[0],
        [0,0,0,0],
        eigenvectors[1],
        [0,0,0,0]
    ]).T

    return eigenvector_columns
    

def reflection_coefficients(T):

    bottom_line = (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])
    
    r_pp = (T[:,0,0] * T[:,3,2] - T[:,3,0] * T[:,0,2]) / bottom_line
    
    r_ps = (T[:,0,0] * T[:,1,2] - (T[:,1,0] * T[:,0,2])) / bottom_line

    r_sp = (T[:,3,0] * T[:,2,2] - T[:,3,2] * T[:,2,0]) / bottom_line

    r_ss = (T[:,1,0] * T[:,2,2] - T[:,1,2] * T[:,2,0]) / bottom_line  
    
    return r_pp, r_ps, r_sp, r_ss


def plot_various_reflectivities(frequency, r_pp, r_ps, r_sp, r_ss, theta, rotation_z, anisotropy, distance):
    reflection1 = (r_pp * np.conj(r_pp)).real
    reflection2 = (r_ss * np.conj(r_ss)).real
    reflection3 = ((r_sp) * np.conj(r_sp)).real
    reflection4 = ((r_ps) * np.conj(r_ps)).real
    
    plt.rcParams["figure.figsize"] = (8,7)
    fig, (ax, ax2, ax3, ax4) = plt.subplots(ncols=4,figsize=(12,6), sharey=True)#gridspec_kw={"width_ratios":[1,1, 0.05]})
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle("ATR for $\\theta$ = " + str(int(round(np.degrees(theta),1))) + " $\phi_x$ = " + str(int(round((anisotropy),1))) + "$^\circ$, $\phi_z$ = " + str(int(round((rotation_z),1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")

    ax.set_title('$|r_{pp}|^2$')
    ax.plot(reflection1, frequency)
    ax.set(xlabel = '$|r_{pp}|^2$')
    ax.set(ylabel='$\omega/2\pi c (cm^{-1})$')

    ax2.set_title('$|r_{ss}|^2$')
    ax2.plot(reflection2, frequency)
    ax2.set(xlabel = '$|r_{ss}|^2$')

    ax3.set_title('$|r_{sp}|^2$')
    ax3.plot(reflection3, frequency)
    ax3.set(xlabel = '$|r_{sp}|^2$')

    ax4.set_title('$|r_{ps}|^2$')
    ax4.plot(reflection4, frequency)
    ax4.set(xlabel = '$|r_{ps}|^2$')

    # plt.tight_layout()
    plt.show()



def contour_theta(wavenumber, x_axis, distance, anisotropy_rotation_x, rotation_z, reflectivities):

    r_pp = reflectivities[:,0]
    r_ps = reflectivities[:,1]
    r_sp = reflectivities[:,2]
    r_ss = reflectivities[:,3]

    x_axis = np.degrees(x_axis)

    R_pp = (r_pp * np.conj(r_pp)).real.T
    R_ps = (r_ps * np.conj(r_ps)).real.T

    R_sp = (r_sp * np.conj(r_sp)).real.T
    R_ss = (r_ss * np.conj(r_ss)).real.T

    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    fig, ax = plt.subplots(2,3, figsize=(12, 7))

    fig.suptitle("ATR for $\phi_x$ = " + str(int(round(np.degrees(anisotropy_rotation_x),1))) + "$^\circ$, $\phi_z$ = " + str(int(round(np.degrees(rotation_z),1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")

    reflection_pp = ax[0,0].pcolormesh(x_axis, wavenumber, R_pp, cmap = 'magma')
    cbar_pp = plt.colorbar(reflection_pp, ax = ax[0,0])
    # cbar_pp.mappable.set_clim(0., 1.)
    cbar_pp.set_label("$|r_{pp}|^2$")
    ax[0,0].set_title("$|r_{pp}|^2$")
    ax[0,0].set_xlabel('$k_x / k_0 $')
    ax[0,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ps = ax[0,1].pcolormesh(x_axis, wavenumber, R_ps, cmap = 'magma')
    cbar_ps = plt.colorbar(reflection_ps, ax = ax[0,1])
    # cbar_ps.mappable.set_clim(0., 1.)
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
    # cbar_sp.mappable.set_clim(0., 1.)
    cbar_sp.set_label("$|r_{sp}|^2$")
    ax[1,0].set_title("$|r_{sp}|^2$")
    ax[1,0].set_xlabel('$k_x / k_0 $')
    ax[1,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss = ax[1,1].pcolormesh(x_axis, wavenumber, R_ss, cmap = 'magma')
    cbar_ss = plt.colorbar(reflection_ss, ax = ax[1,1])
    # cbar_pp.mappable.set_clim(0., 1.)
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


def contour_z_rotation(wavenumber, x_axis, distance, anisotropy_rotation_x, incident_angle, reflectivities):

    r_pp = reflectivities[:,0]
    r_ps = reflectivities[:,1]
    r_sp = reflectivities[:,2]
    r_ss = reflectivities[:,3]

    R_pp = (r_pp * np.conj(r_pp)).real.T
    R_ps = (r_ps * np.conj(r_ps)).real.T
    R_sp = (r_sp * np.conj(r_sp)).real.T
    R_ss = (r_ss * np.conj(r_ss)).real.T

    R_pp_total = R_pp + R_ps
    R_ss_total = R_ss + R_sp

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    fig, ax = plt.subplots(2,3, figsize=(12, 7))

    x_axis = np.degrees(x_axis)
    fig.suptitle("ATR for $\phi_x$ = " + str(int(round(np.degrees(anisotropy_rotation_x),1))) + "$^\circ$, $\\theta$ = " + str(int(round(np.degrees(incident_angle),1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")

    reflection_pp = ax[0,0].pcolormesh(x_axis, wavenumber, R_pp, cmap = 'magma')
    cbar_pp = plt.colorbar(reflection_pp, ax = ax[0,0])
    # cbar_pp.mappable.set_clim(0., 1.)
    cbar_pp.set_label("$|r_{pp}|^2$")
    ax[0,0].set_title("$|r_{pp}|^2$")
    ax[0,0].set_xlabel('$\phi_z$ / $^\circ$')
    ax[0,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ps = ax[0,1].pcolormesh(x_axis, wavenumber, R_ps, cmap = 'magma')
    cbar_ps = plt.colorbar(reflection_ps, ax = ax[0,1])
    # cbar_ps.mappable.set_clim(0., 1.)
    cbar_ps.set_label("$|r_{ps}|^2$")
    ax[0,1].set_title("$|r_{ps}|^2$")
    ax[0,1].set_xlabel('$\phi_z$ / $^\circ$')
    ax[0,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_pp_total = ax[0,2].pcolormesh(x_axis, wavenumber, R_pp_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_pp_total, ax = ax[0,2])
    cbar_pp_total.set_label("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_title("$|r_{pp}|^2 + |r_{ps}|^2$")
    ax[0,2].set_xlabel('$\phi_z$ / $^\circ$')
    ax[0,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_sp = ax[1,0].pcolormesh(x_axis, wavenumber, R_sp, cmap = 'magma')
    cbar_sp = plt.colorbar(reflection_sp, ax = ax[1,0])
    # cbar_sp.mappable.set_clim(0., 1.)
    cbar_sp.set_label("$|r_{sp}|^2$")
    ax[1,0].set_title("$|r_{sp}|^2$")
    ax[1,0].set_xlabel('$\phi_z$ / $^\circ$')
    ax[1,0].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss = ax[1,1].pcolormesh(x_axis, wavenumber, R_ss, cmap = 'magma')
    cbar_ss = plt.colorbar(reflection_ss, ax = ax[1,1])
    # cbar_pp.mappable.set_clim(0., 1.)
    cbar_ss.set_label("$|r_{ss}|^2$")
    ax[1,1].set_title("$|r_{ss}|^2$")
    ax[1,1].set_xlabel('$\phi_z$ / $^\circ$')
    ax[1,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_ss_total = ax[1,2].pcolormesh(x_axis, wavenumber, R_ss_total, cmap = 'magma')
    cbar_pp_total = plt.colorbar(reflection_ss_total, ax = ax[1,2])
    cbar_pp_total.set_label("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_title("$|r_{ss}|^2 + |r_{sp}|^2$")
    ax[1,2].set_xlabel('$\phi_z$ / $^\circ$')
    ax[1,2].set_ylabel('$\omega/2\pi c (cm^{-1})$')
    
    plt.tight_layout()
    plt.show()
    plt.close()



def main_magnet_contour():
    frequency = np.linspace(52.0, 53.5, 300)

    anisotropy_rotation = np.radians(0)
    rotation_z = np.radians(0)
    d = 23.e-4

    parameters = magnetic_parameters()
    mu_3, mu_t, eps_magnet = fetch_epsilon_mu(*(frequency, *parameters))
    mu_tensor, magnet_eps_tensor = construct_magnet_tensors(mu_3, mu_t, eps_magnet)

    mu_tensor = anisotropy_rotation_matrix_y(mu_tensor, anisotropy_rotation)
    mu_tensor = anisotropy_rotation_matrix_z(mu_tensor, rotation_z)


    eps_prism = 11.56
    incident_angle = np.linspace(-np.pi/2., np.pi/2., len(frequency))
    k0 = frequency * 2. * np.pi

    reflectivities = []

    for angle in incident_angle:
        prism_layer = ambient_incident_prism(eps_prism, angle)
        kx = np.sqrt(eps_prism) * np.sin(angle)
        transfer_matrices = []

        for i in range(0,len(frequency)):
            vector = layer_matrix(magnet_eps_tensor, mu_tensor[i], kx, 0, 0, magnet=True)
            magnet_eigenvectors = vector_sort(vector)

            air_layer = layer_matrix(air_tensor(), air_tensor(), kx, k0[i], d)

            transfer = prism_layer @ np.linalg.inv(air_layer) @ magnet_eigenvectors
            transfer_matrices.append(transfer)

        transfer_matrices = np.asarray(transfer_matrices)

        r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)


        reflectivities.append([r_pp,r_ps,r_sp,r_ss])
    
    reflectivities = np.asarray(reflectivities)
    
    contour_theta(frequency, incident_angle, d, anisotropy_rotation, rotation_z, reflectivities)


def main_magnet():
    frequency = np.linspace(52.0, 53.5, 300)

    anisotropy_rotation_x = 0.
    rotation_z = np.radians(45)
    d = 0.

    parameters = magnetic_parameters()
    mu_3, mu_t, eps_magnet = fetch_epsilon_mu(*(frequency, *parameters))
    mu_tensor, magnet_eps_tensor = construct_magnet_tensors(mu_3, mu_t, eps_magnet)

    mu_tensor = anisotropy_rotation_matrix_z(mu_tensor, rotation_z)


    eps_prism = 11.56
    angle = np.radians(45)
    k0 = frequency * 2. * np.pi


    prism_layer = ambient_incident_prism(eps_prism, angle)
    kx = np.sqrt(eps_prism) * np.sin(angle)
    transfer_matrices = []

    for i in range(0,len(frequency)):
        vector = layer_matrix(magnet_eps_tensor, mu_tensor[i], kx, 0, 0, magnet=True)
        magnet_eigenvectors = vector_sort(vector)

        # air_layer = layer_matrix(air_tensor(), air_tensor(), kx, k0[i], d)

        transfer = prism_layer @ magnet_eigenvectors
        transfer_matrices.append(transfer)

    transfer_matrices = np.asarray(transfer_matrices)

    r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)
    
    plot_various_reflectivities(frequency, r_pp, r_ps, r_sp, r_ss, angle, rotation_z, 0, d)


def main_quartz_contour():
    frequency = np.linspace(410,600,300)
    
    anisotropy_rotation_y = np.radians(45)
    rotation_z = np.radians(45)
    theta_cap = np.radians(90)
    d = 1.5e-4

    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_y(quartz_tensor, anisotropy_rotation_y)
    quartz_tensor = anisotropy_rotation_matrix_z(quartz_tensor, rotation_z)

    eps_prism = 5.5
    incident_angle = np.linspace(-theta_cap, theta_cap, len(frequency))
    k0 = frequency * 2. * np.pi

    reflectivities = []

    for angle in incident_angle:
        prism_layer = ambient_incident_prism(eps_prism, angle)
        kx = np.sqrt(eps_prism) * np.sin(angle)
        transfer_matrices = []

        for i in range(0,len(frequency)):
            vector = layer_matrix(quartz_tensor[i], air_tensor(), kx, 0, 0, quartz=True)
            quartz_eigenvectors = vector_sort(vector)

            air_layer = layer_matrix(air_tensor(), air_tensor(), kx, k0[i], d)

            transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_eigenvectors
            transfer_matrices.append(transfer)

        transfer_matrices = np.asarray(transfer_matrices)

        r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)
        reflectivities.append([r_pp,r_ps,r_sp,r_ss])


    reflectivities = np.asarray(reflectivities)

    contour_theta(frequency, incident_angle, d, anisotropy_rotation_y, rotation_z, reflectivities)


def main_quartz_rotation():
    frequency = np.linspace(410,600,300)
    
    anisotropy_rotation_y = np.radians(45)
    incident_angle = np.radians(45)
    rotation_z = np.linspace(0, 2 * np.pi, len(frequency))
    d = 1.5e-4

    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_y(quartz_tensor, anisotropy_rotation_y)

    eps_prism = 5.5
    k0 = frequency * 2. * np.pi

    reflectivities = []

    for rotation_angle in rotation_z:
        quartz_tensor_rotated = anisotropy_rotation_matrix_z(quartz_tensor, rotation_angle)
        prism_layer = ambient_incident_prism(eps_prism, incident_angle)
        kx = np.sqrt(eps_prism) * np.sin(incident_angle)
        transfer_matrices = []

        for i in range(0,len(frequency)):
            vector = layer_matrix(quartz_tensor_rotated[i], air_tensor(), kx, 0, 0, quartz=True)
            quartz_eigenvectors = vector_sort(vector)

            air_layer = layer_matrix(air_tensor(), air_tensor(), kx, k0[i], d)

            transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_eigenvectors
            transfer_matrices.append(transfer)

        transfer_matrices = np.asarray(transfer_matrices)

        r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)
        reflectivities.append([r_pp,r_ps,r_sp,r_ss])


    reflectivities = np.asarray(reflectivities)

    contour_z_rotation(frequency, rotation_z, d, anisotropy_rotation_y, incident_angle, reflectivities)


def main_quartz():
    frequency = np.linspace(410,600,300)
    
    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_y(quartz_tensor, np.radians(30))
    quartz_tensor = anisotropy_rotation_matrix_z(quartz_tensor, np.radians(30))


    eps_prism = 5.5
    incident_angle = np.radians(45)
    d=0.
    
    k0 = frequency * 2. * np.pi
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    transfer_matrices = []
    prism_layer = ambient_incident_prism(eps_prism, incident_angle)

    for i in range(0, len(frequency)):
        vector = layer_matrix(quartz_tensor[i], air_tensor(), kx, 0, 0, quartz=True)
        quartz_eigenvectors = vector_sort(vector)

        air_layer = layer_matrix(air_tensor(), air_tensor(), kx, k0[i], d)

        transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_eigenvectors
        transfer_matrices.append(transfer)

    transfer_matrices = np.asarray(transfer_matrices)

    r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)
    plot_various_reflectivities(frequency, r_pp, r_ps, r_sp, r_ss, incident_angle, 45, 45, d)


if __name__ == "__main__":
    # main_magnet_kz()
    main_quartz_rotation()