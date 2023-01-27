import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import constants


def magnetic_parameters():
    gamma = 1.05 # /cm /T
    H0 = 0.3 # T 
    Ha = 19.745 # T
    He = 53.313 # T
    magnetisation = 44563. # A/M

    resonant_frequency_squared = gamma**2 * (2. * Ha * He + Ha**2.)
    damping_parameter = 1.27e-4

    return gamma, H0, Ha, magnetisation, resonant_frequency_squared, damping_parameter


def calculate_permeability(frequency, gamma, H0, Ha, magnetisation, resonant_frequency_squared, damping):

    X = 1./ (resonant_frequency_squared - (frequency + gamma * H0 + 1.j * damping)**2)
    Y = 1./ (resonant_frequency_squared - (frequency - gamma * H0 + 1.j * damping)**2)
    
    mu_3 = 1. + (constants.mu_0 * gamma**2 * Ha * magnetisation * (X+Y))
    mu_t = constants.mu_0 * gamma**2 * Ha * magnetisation * (X-Y)

    mu_v = (mu_3**2. - mu_t**2.)/mu_3

    return mu_3, mu_t, mu_v


def retrieve_matrix(frequency, permeability_3, permeability_t):
    kx, ky, kz, k0, epsilon, mu_3, mu_t, omega = sp.symbols('k_x k_y k_z k_{0} \epsilon_{3} \mu_{3} \mu_{t} \omega')

    element_1 = k0**2 * epsilon * mu_3 - ky**2 - kz**2
    element_2 = k0**2 * epsilon * mu_t * sp.I + kx * ky
    element_3 = kx * kz
    element_4 = kx * ky - k0**2 * epsilon * mu_t * sp.I
    element_5 = k0**2 * epsilon * mu_3 - kx**2 - kz**2
    element_6 = ky * kz
    element_7 = kx * kz
    element_8 = ky * kz
    element_9 = (k0**2 * epsilon - kx**2 - ky**2)

    matrix = sp.Matrix([element_1, element_2, element_3, element_4, element_5, element_6, element_7, element_8, element_9]).reshape(3,3)
    
    new_matrices = [matrix.subs({
        k0 : 2 * np.pi * omega,
        kx : np.sin(np.pi/4.) * 2 * np.pi * omega * np.sqrt(11.56),
        kz : 0. + 0j,
        epsilon : 5.5 + 0j,
        mu_3 : item_1 + 0j,
        mu_t : item_2 + 0j
    }).subs({ omega : item_3 + 0j}) for (item_1, item_2, item_3) in zip(permeability_3, permeability_t, frequency)]

    return new_matrices


def solve_ky(matrices):
    determinants = [item.det() for item in matrices]
    ky_solutions = [sp.solve(item, sp.symbols('k_y')) for item in determinants]
    
    return np.asarray(ky_solutions).astype(complex)


def plot_ky(frequency, ky_solutions):

    zeroth_solutions = ky_solutions[:,0]
    first_solutions = ky_solutions[:,1]
    second_solutions = ky_solutions[:,2]
    third_solutions = ky_solutions[:,3]
    
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(4)
    fig.suptitle('$k_{y}$ Solutions')
    axs[0].plot(frequency, zeroth_solutions.real, label = 'Real')
    axs[0].plot(frequency, zeroth_solutions.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[0].legend()

    axs[1].plot(frequency, first_solutions.real, label = 'Real')
    axs[1].plot(frequency, first_solutions.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[1].legend()

    axs[2].plot(frequency, second_solutions.real, label = 'Real')
    axs[2].plot(frequency, second_solutions.imag, label = 'Imaginary')
    axs[2].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[2].legend()

    axs[3].plot(frequency, third_solutions.real, label = 'Real')
    axs[3].plot(frequency, third_solutions.imag, label = 'Imaginary')
    axs[3].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[3].legend()

    plt.show()
    plt.close()




def plot_analytical_and_matrix(frequency, analytical_ky, analytical_ky_other, matrix_ky, matrix_ky_other):
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(2)
    fig.suptitle('$k_y$ Solutions - Analytical and from Matrix')
    axs[0].plot(frequency, analytical_ky.real,'--', label = 'Analytical Real',)
    axs[0].plot(frequency, analytical_ky.imag,'--', label = 'Analytical Imaginary')
    axs[0].plot(frequency, matrix_ky.real,'-.', label = 'Matrix Real')
    axs[0].plot(frequency, matrix_ky.imag,'-.', label = 'Matrix Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_y$')
    axs[0].legend()

    axs[1].plot(frequency, analytical_ky_other.real,'--', label = 'Analytical Real')
    axs[1].plot(frequency, analytical_ky_other.imag,'--', label = 'Analytical Imaginary')
    axs[1].plot(frequency, matrix_ky_other.real,'-.', label = 'Matrix Real')
    axs[1].plot(frequency, matrix_ky_other.imag,'-.', label = 'Matrix Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[1].legend()

    plt.show()

def plot_permeability_ky(frequency, mu_3, mu_t, ky):
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(3)
    fig.suptitle('Permeabilities and ky')
    axs[0].plot(frequency, mu_3.real, label = 'Real')
    axs[0].plot(frequency, mu_3.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{3}$')
    axs[0].legend()

    axs[1].plot(frequency, mu_t.real, label = 'Real')
    axs[1].plot(frequency, mu_t.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{t}$')
    axs[1].legend()
    
    axs[2].plot(frequency, ky.real, label = 'Real')
    axs[2].plot(frequency, ky.imag, label = 'Imaginary')
    axs[2].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[2].legend()

    plt.show()
    plt.close()


def plot_permeability_components(frequency, permeability_3, permeability_t, mu_v):

    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(3)
    fig.suptitle('Permeabilities and $\mu_{v}$')
    axs[0].plot(frequency, permeability_3.real, label = 'Real')
    axs[0].plot(frequency, permeability_3.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{3}$')
    axs[0].legend()

    axs[1].plot(frequency, permeability_t.real, label = 'Real')
    axs[1].plot(frequency, permeability_t.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{t}$')
    axs[1].legend()
    
    axs[2].plot(frequency, mu_v.real, label = 'Real')
    axs[2].plot(frequency, mu_v.imag, label = 'Imaginary')
    axs[2].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{v}$')
    axs[2].legend()

    plt.show()
    plt.close()


def fetch_reflectivity(frequency, mu_3, mu_t, mu_v,  incidence = 0., permittivity = 5.5):

    k0 = frequency * 2. * np.pi + 0.j
    kx = k0 * np.sin(incidence) * np.sqrt(11.56) + 0.j
    
    k1y = np.sqrt(k0**2 * 11.56 - kx**2 + 0.j)

    k2y = np.sqrt(permittivity * mu_v * k0**2 - kx**2)

    top_line = k1y * mu_v - k2y - 1.j * kx * (mu_t/mu_3)
    bottom_line = k1y * mu_v + k2y + 1.j * kx * (mu_t/mu_3)

    reflectivity = top_line/bottom_line

    return reflectivity * np.conjugate(reflectivity)


def plot_reflectivity(wavenumber, reflectivity, incident_angle):
    plt.rcParams["figure.figsize"] = (3,5)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.plot(reflectivity, wavenumber)
    plt.xlabel('Reflectivity')
    plt.ylabel('$\omega/2\pi c (cm^{-1})$')
    #plt.xlim(0,1)
    plt.title(r'ATR for $\theta$ = ${}^\circ$ S-Polarised'.format(np.degrees(incident_angle)))

    plt.show()
    plt.close()
    


def main():
    frequency = np.linspace(52.01, 54, 1000)
    parameters = magnetic_parameters()
    permeability_3, permeability_t, mu_v = calculate_permeability(*(frequency, *parameters))
    
    #plot_permeability_components(frequency, permeability_3, permeability_t, mu_v)

    #matrices = retrieve_matrix(frequency, permeability_3, permeability_t)
    #ky_solutions = solve_ky(matrices)
    #ky_matrix = ky_solutions[:,3]
    #ky_matrix_other = ky_solutions[:,0]

    #plot_permeability_ky(frequency, permeability_3, permeability_t, ky_matrix)
    #plot_permeability_ky(frequency, permeability_3, permeability_t, ky_other_analytical)

    reflectivity = fetch_reflectivity(frequency, permeability_3, permeability_t, mu_v)
    plot_reflectivity(frequency, reflectivity, 0.)


    # plot_analytical_and_matrix(frequency, ky_analytical, ky_other_analytical, ky_matrix, ky_matrix_other)


if __name__ == "__main__":
    main()