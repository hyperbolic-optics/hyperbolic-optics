import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import constants
import warnings
warnings.filterwarnings("error")

from scipy.linalg import expm


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
        [mu_3, np.zeros_like(mu_3), 1.j * mu_t],
        [np.zeros_like(mu_3), np.ones_like(mu_3), np.zeros_like(mu_3)],
        [-1.j * mu_t, np.zeros_like(mu_3), mu_3]
        ], dtype = object
    ).T

    permittivity = np.array(
        [
        [epsilon, 0., 0.],
        [0., epsilon, 0.],
        [0., 0., epsilon]
        ]
    )

    return permeability, permittivity


def air_tensor():

    both_tensor = np.array(
        [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ],
    )

    return both_tensor


def prism_tensor():

    eps = np.array(
        [
        [11.56, 0., 0.],
        [0., 11.56, 0.],
        [0., 0., 11.56]
        ],
    )

    return eps


def ambient_matrix_prism_inverted(incident_angle, n):
    matrix = np.array([
        [0., 0., np.cos(incident_angle), -np.cos(incident_angle)],
        [1., 1., 0., 0.],
        [-n * np.cos(incident_angle), n * np.cos(incident_angle), 0., 0.],
        [0., 0., n, n]
    ])

    return np.linalg.inv(matrix)

def layer_matrix(eps_tensor, mu_tensor, kx, k0, thickness, incident_ambient = False, incident_exit = False):

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



def main():
    frequency = np.linspace(52.0, 53.5, 300)
    
    eps_prism = 11.56
    angle = 0.
    k0 = frequency * 2. * np.pi
    kx = np.sqrt(eps_prism) * np.sin(angle)
    
    parameters = magnetic_parameters()
    mu_3, mu_t, eps_magnet = fetch_epsilon_mu(*(frequency, *parameters))
    mu_tensor, eps_tensor = construct_magnet_tensors(mu_3, mu_t, eps_magnet)
    air_params = air_tensor()
    tensor_p = prism_tensor()

    transfer_matrices = []
    #prism_layer = ambient_matrix_prism_inverted(angle, np.sqrt(eps_prism))
    
    for i in range(0,len(frequency)):
        prism_layer = layer_matrix(tensor_p, air_params, kx, k0[i], thickness = 0., incident_ambient = True)
        magnet_layer = layer_matrix(eps_tensor, mu_tensor[i], kx, k0[i], thickness = 0., incident_exit = True)
        air_layer = layer_matrix(air_params, air_params, kx, k0[i], thickness = 1.e-4)
        
        total_transfer = prism_layer @ air_layer @ magnet_layer
        
        transfer_matrices.append(total_transfer)

    transfer_matrices = np.asarray(transfer_matrices)

    r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)

    reflection = ((r_pp) * np.conj(r_pp) ).real

    plt.plot(frequency, mu_3.real)
    plt.show()
   





if __name__ == "__main__":
    main()