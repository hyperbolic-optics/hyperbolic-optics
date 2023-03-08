import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import constants
from scipy.linalg import expm


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

    return permittivity


def anisotropy_rotation_matrix_x(matrix, theta):

    rotation_matrix = np.array([
        [1., 0., 0.],
        [0., np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
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
        [0., 1., -n * np.cos(theta), 0.],
        [0., 1., n * np.cos(theta), 0.],
        [1./np.cos(theta), 0., 0., 1./n],
        [-1./np.cos(theta), 0., 0., 1./n]
    ])

    return matrix


def layer_matrix(eps_tensor, kx, k0, thickness, incident_ambient = False, incident_exit = False):

    delta_11 = - kx * (eps_tensor[2,0]/eps_tensor[2,2])
    delta_12 = - kx * (eps_tensor[2,1]/eps_tensor[2,2])
    delta_13 = 0.
    delta_14 = 1. - (kx**2.)/eps_tensor[2,2]

    delta_21 = 0.
    delta_22 = 0.
    delta_23 = -1.
    delta_24 = 0.

    delta_31 = (eps_tensor[1,2] * eps_tensor[2,0] / eps_tensor[2,2]) - eps_tensor[1,0]
    delta_32 = (kx**2.) - eps_tensor[1,1] + (eps_tensor[1,2] * eps_tensor[2,1]/ eps_tensor[2,2])
    delta_33 = 0.
    delta_34 = kx * (eps_tensor[1,2]/eps_tensor[2,2])

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

    eigenvalue, vector = np.linalg.eig(delta)
    if incident_ambient:
        return np.linalg.inv(vector)
    
    if incident_exit:
        return vector

    eigenvalue = np.diag(eigenvalue)
    partial = expm(1.j * eigenvalue * k0 * thickness)
    partial_complete = vector @ partial @ np.linalg.inv(vector)


    return partial_complete


def reflection_coefficients(T):
    
    r_pp = (T[:,0,0] * T[:,3,2] - T[:,3,0] * T[:,0,2]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])
    
    r_ps = (T[:,0,0] * T[:,1,2] - T[:,1,0] * T[:,0,2]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    r_sp = (T[:,3,0] * T[:,2,2] - T[:,3,2] * T[:,2,0]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    r_ss = (T[:,1,0] * T[:,2,2] - T[:,1,2] * T[:,2,0]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    return r_pp, r_ps, r_sp, r_ss


def plot_reflectivity_components(frequency, r_pp, r_ps, r_sp, r_ss):
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(4)
    fig.suptitle('Reflection Coefficients')
    axs[0].plot(frequency, r_pp.real, label = 'Real')
    axs[0].plot(frequency, r_pp.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$r_{pp}$')
    axs[0].legend()

    axs[1].plot(frequency, r_ps.real, label = 'Real')
    axs[1].plot(frequency, r_ps.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$r_{ps}$')
    axs[1].legend()

    axs[2].plot(frequency, r_sp.real, label = 'Real')
    axs[2].plot(frequency, r_sp.imag, label = 'Imaginary')
    axs[2].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$r_{sp}$')
    axs[2].legend()

    axs[3].plot(frequency, r_ss.real, label = 'Real')
    axs[3].plot(frequency, r_ss.imag, label = 'Imaginary')
    axs[3].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$r_{ss}$')
    axs[3].legend()

    
    plt.show()
    plt.close()



def main():
    frequency = np.linspace(410,600,300)
    anisotropy_rotation = 0.
    
    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_x(quartz_tensor, anisotropy_rotation)

    eps_prism = 5.5
    incident_angle = -np.pi/4.
    
    k0 = frequency * 2. * np.pi
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)
    
    air_params = air_tensor()

    transfer_matrices = []
    prism_layer = ambient_incident_prism(eps_prism, incident_angle)
    
    for i in range(0,len(frequency)):
        quartz_layer = layer_matrix(quartz_tensor[i], kx, 0., 0., incident_exit=True)
        air_layer = layer_matrix(air_params, kx, k0[i], thickness = 0.)
        
        total_transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_layer
        
        transfer_matrices.append(total_transfer)

    transfer_matrices = np.asarray(transfer_matrices)

    r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)

    reflection = (r_pp * np.conj(r_pp)).real
    plt.plot(reflection, frequency)
    plt.show()

    


if __name__ == "__main__":
    main()