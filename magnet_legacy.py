import numpy as np
from scipy import constants

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

