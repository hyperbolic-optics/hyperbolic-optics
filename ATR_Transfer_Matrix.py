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


def anisotropy_matrix(matrix, phi, beta):
    rotation_y = np.array([
        [np.cos(phi), 0., np.sin(phi)],
        [0., 1., 0.],
        [-np.sin(phi), 0., np.cos(phi)]
    ])

    rotation_z = np.array([
        [np.cos(beta), -np.sin(beta), 0.],
        [np.sin(beta), np.cos(beta), 0.],
        [0., 0., 1.]
    ])

    rotation_x = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    total_rotation = rotation_z @ rotation_y @ rotation_x

    return total_rotation @ matrix @ total_rotation.T


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

    cos_theta = np.cos(theta)
    n_cos_theta = n * cos_theta
    one_over_cos_theta = 1. / cos_theta
    one_over_n = 1. / n
    one_over_n_cos_theta = 1. / n_cos_theta

    matrix = np.zeros((theta.size, 4, 4))

    matrix[:, 0, 1] = 1.
    matrix[:, 1, 1] = 1.
    matrix[:, 0, 2] = -1./ (n * np.cos(theta))
    matrix[:, 1, 2] = 1./ (n * np.cos(theta))
    matrix[:, 2, 0] = 1./ np.cos(theta)
    matrix[:, 3, 0] = -1./ np.cos(theta)
    matrix[:, 2, 3] = 1./n
    matrix[:, 3, 3] = 1./n

    return 0.5 * matrix


def layer_matrix(eps_tensor, mu_tensor, kx, k0, thickness, quartz = False):

    delta = np.zeros((300, 300, 4, 4), dtype=np.complex128)

    delta[..., 0, 0] = -kx * (eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])
    delta[..., 0, 1] = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 0, 2] = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    delta[..., 0, 3] = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]

    delta[..., 1, 0] = np.zeros_like(kx)
    delta[..., 1, 1] = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    delta[..., 1, 2] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    delta[..., 1, 3] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    delta[..., 2, 0] = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    delta[..., 2, 1] = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    delta[..., 2, 2] = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    delta[..., 2, 3] = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    delta[..., 3, 0] = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    delta[..., 3, 1] = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 3, 2] = np.zeros_like(kx)
    delta[..., 3, 3] = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    delta = np.transpose(delta, (1,0,2,3))

    print(delta[0][0])

    exit()

    print(delta.shape)

    eigenvalues, vector = np.linalg.eig(delta)

    if quartz:
        
        order = (1.j * eigenvalues).argsort(axis=-1)[..., np.newaxis]
        vector = np.transpose(vector, (0, 1, 3, 2))
        vector = np.take_along_axis(vector, order, axis=-2)

        vector[..., 2:4, :] = 0
        vector[..., 2, :] = vector[..., 1, :]
        vector[..., 1, :] = 0

        vector = np.transpose(vector, (0, 1, 3, 2))

        return vector

    exit()

    return partial_complete
    

def reflection_coefficients(T):

    bottom_line = (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])
    
    r_pp = (T[:,0,0] * T[:,3,2] - T[:,3,0] * T[:,0,2]) / bottom_line
    
    r_ps = (T[:,0,0] * T[:,1,2] - (T[:,1,0] * T[:,0,2])) / bottom_line

    r_sp = (T[:,3,0] * T[:,2,2] - T[:,3,2] * T[:,2,0]) / bottom_line

    r_ss = (T[:,1,0] * T[:,2,2] - T[:,1,2] * T[:,2,0]) / bottom_line  
    
    return r_pp, r_ps, r_sp, r_ss


def contour_theta(wavenumber, x_axis, distance, anisotropy_rotation_x, rotation_z, reflectivities):

    r_pp = reflectivities[:, 0]
    r_ps = reflectivities[:, 1]
    r_sp = reflectivities[:, 2]
    r_ss = reflectivities[:, 3]

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


def main_quartz_contour():
    frequency = np.linspace(410,600,300)
    
    anisotropy_rotation_y = np.radians(0)
    rotation_z = np.radians(0)
    theta_cap = np.radians(90)
    d = 0.
    eps_prism = 5.5

    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_matrix(quartz_tensor, anisotropy_rotation_y, rotation_z)

    non_magnetic_tensor = np.tile(air_tensor(), (300, 1, 1))

    incident_angle = np.linspace(-theta_cap, theta_cap, len(frequency))
    k0 = frequency * 2. * np.pi
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    reflectivities = []

    prism_layer = ambient_incident_prism(eps_prism, incident_angle)
    
    quartz_eigenvectors = layer_matrix(quartz_tensor, non_magnetic_tensor, kx, 0, 0, quartz=True)

    reflectivities = []

    for i in range(len(quartz_eigenvectors)):
        transfers = []
        for j in range(len(prism_layer)):
            transfer = prism_layer[i] @ quartz_eigenvectors[i][j]
            transfers.append(transfer)
        
        transfers = np.asarray(transfers)
    
        reflections = reflection_coefficients(transfers)
        reflectivities.append(reflections)
    
    reflectivities = np.asarray(reflectivities)

    print(reflectivities.shape)

    contour_theta(frequency, incident_angle, d, anisotropy_rotation_y, rotation_z, reflectivities)


if __name__ == "__main__":
    # main_magnet_kz()
    main_quartz_contour()