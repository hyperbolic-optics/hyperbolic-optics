import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inverse
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def wavenumber_spectrum():
    return np.linspace(410,600,300)


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
    E_xx = []
    E_zz = []
    for element in wavenumber:
        E_xx.append(permittivity_calc(element, **params["extraordinary"]))
        E_zz.append(permittivity_calc(element, **params["ordinary"]))
    
    E_xx, E_zz = np.asarray(E_xx), np.asarray(E_zz)
    return E_xx, E_zz

def plot_permittivity(wavenumber, E_xx, E_zz):
    plt.rcParams["figure.figsize"] = (5,4)
    fig, axs = plt.subplots(2)
    fig.suptitle('Permittivities')
    axs[0].plot(wavenumber, E_xx.real)
    axs[0].plot(wavenumber, E_xx.imag)
    axs[0].set(xlabel='Wavenumber', ylabel = 'E_xx')

    axs[1].plot(wavenumber, E_zz.real)
    axs[1].plot(wavenumber, E_zz.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'E_zz')

    plt.show()

    
if __name__ == "__main__":
    params = permittivity_parameters()
    wavenumber = wavenumber_spectrum()
    E_xx, E_zz = permittivity_fetch(wavenumber, params)
    plot_permittivity(wavenumber, E_xx, E_zz)

