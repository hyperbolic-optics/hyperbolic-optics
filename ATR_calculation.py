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

def plot_base_permittivity(wavenumber, E_xx, E_zz):
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

def rotated_permittivities(E_ext, E_ord, rotation = git snp.pi/4.):
    E_xx = E_ext * (np.cos(rotation))**2 + E_ord * (np.sin(rotation))**2
    E_zz = E_ext * (np.sin(rotation))**2 + E_ord * (np.cos(rotation))**2
    
    E_xz = E_ord * np.cos(rotation) * np.sin(rotation) - E_ext * np.cos(rotation) * np.sin(rotation)

    return E_xx, E_zz, E_xz


def plot_rotated_permittivities(wavenumber, E_xx, E_zz, E_xz):
    plt.rcParams["figure.figsize"] = (7,5)
    fig, axs = plt.subplots(3)
    fig.suptitle('Permittivities')
    axs[0].plot(wavenumber, E_xx.real)
    axs[0].plot(wavenumber, E_xx.imag)
    axs[0].set(xlabel='Wavenumber', ylabel = 'E_xx')

    axs[1].plot(wavenumber, E_zz.real)
    axs[1].plot(wavenumber, E_zz.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'E_zz')

    axs[2].plot(wavenumber, E_xz.real)
    axs[2].plot(wavenumber, E_xz.imag)
    axs[2].set(xlabel='Wavenumber', ylabel = 'E_xz')

    plt.show()

def kvectors(wavenumber, E_xx, E_zz, E_xz, prism_permittivity = 5.5, angle=np.pi/4., rotation = np.pi/4.):
    k0 = wavenumber * 2 * np.pi
    kx = k0 * np.sqrt(prism_permittivity) * np.sin(angle)

    a = - E_zz
    b = - 2. * E_xz * kx
    c = - E_xx * kx**2 - k0**2 * (E_xx * E_zz - E_xz**2)

    kz1 = np.sqrt(k0**2 * prism_permittivity - kx**2 + 0j)
    kz2 = np.sqrt(k0**2 - kx**2 + 0j)

    kz3 = ((-1.*b) + np.sqrt((b**2) - (4. * a * c)))/(2.*a)
    kz3_other = ((-1.*b) - np.sqrt((b**2) - (4. * a * c)))/(2.*a)

    for i in range(len(kz3)):
        if kz3[i].imag<0:
            kz3[i] = kz3_other[i]


    return kx, kz1, kz2, kz3


def plot_kvectors(wavenumber, kz3):
    plt.rcParams["figure.figsize"] = (5,4)
    plt.suptitle('KVectors')
    plt.plot(wavenumber, kz3.real)
    plt.plot(wavenumber, kz3.imag)
    plt.xlabel('Wavenumber')
    plt.ylabel('kz3')

    plt.show()


def reflectivity_calculation(kx, kz1, kz2, kz3, E_xx, E_zz, E_xz, prism_permittivity = 5.5, d = 1.5e-4):
    A11 = -np.exp( 1.j * kz1 * d)
    A12 =  np.exp(-1.j * kz2 * d)
    A13 =  np.exp( 1.j * kz2 * d)
    A14 =  0.
    
    A21 =  kz1 * np.exp( 1.j * kz1 * d) 
    A22 =  kz2 * np.exp(-1.j * kz2 * d) * prism_permittivity
    A23 = -kz2 * np.exp( 1.j * kz2 * d) * prism_permittivity  
    A24 =  0.
    
    A31 =  0.
    A32 =  1.
    A33 =  1.
    A34 = -1.
  
    A41 =  0.
    A42 = -1.
    A43 =  1. 
    A44 =  (E_zz * kz3 + E_xz*kx) / (kz2*(E_xx * E_zz - E_xz**2))

    reflection_matrix = np.array([[A11, A12, A13, A14],
                           [A21, A22, A23, A24],
                           [A31, A32, A33, A34],
                           [A41, A42, A43, A44]])  

    column_matrix = np.array([np.exp(-1.j * kz1 * d), kz1 * np.exp(-1.j * kz1 * d), 0., 0.])  

    small_r = np.linalg.inv(reflection_matrix).dot(column_matrix)

    reflection_value = np.real(small_r[0] * np.conj(small_r[0]))

    return reflection_value


def fetch_reflectivity(kx,kz1,kz2,kz3, E_xx, E_zz, E_xz):
    reflectivity = []
    for i in range(len(kx)):
        reflection = reflectivity_calculation(kx[i], kz1[i], kz2[i], kz3[i], E_xx[i], E_zz[i], E_xz[i])
        reflectivity.append(reflection)
    return np.array(reflectivity)


def plot_reflectivity(reflectivity):
    plt.rcParams["figure.figsize"] = (2,4)
    plt.plot(reflectivity,wavenumber)
    plt.xlabel('Reflectivity')
    plt.ylabel('$\omega/2\pi c (cm^{-1})$')
    #plt.xlim(0,1)
    plt.title('ATR for $\epsilon_{ext}$ = $\epsilon_{xx}$ P-Polarised')

    plt.hlines(450,0,1,'black','dashed')
    plt.text(1.3, 450, '$\omega_{T2,ord}$', ha='right', va='center')

    plt.hlines(507,0,1,'black','dashed')
    plt.text(1.3, 507, '$\omega_{L2,ord}$', ha='right', va='center')

    plt.hlines(487.5,0,1,'black','dashed')
    plt.text(1.3, 487, '$\omega_{T2,ext}$', ha='right', va='center')

    plt.hlines(550,0,1,'black','dashed')
    plt.text(1.3, 550, '$\omega_{L2,ext}$', ha='right', va='center')

    plt.show()




if __name__ == "__main__":
    params = permittivity_parameters()
    wavenumber = wavenumber_spectrum()
    E_ext, E_ord = permittivity_fetch(wavenumber, params)
    #plot_base_permittivity(wavenumber, E_ext, E_ord)

    E_xx, E_zz, E_xz = rotated_permittivities(E_ext, E_ord)
    #plot_rotated_permittivities(wavenumber, E_xx, E_zz, E_xz)

    kx, kz1,kz2, kz3 = kvectors(wavenumber, E_xx, E_zz, E_xz)

    # plot_kvectors(wavenumber, kz3)

    reflectivity = fetch_reflectivity(kx,kz1,kz2,kz3, E_xx, E_zz, E_xz)
    plot_reflectivity(reflectivity)
