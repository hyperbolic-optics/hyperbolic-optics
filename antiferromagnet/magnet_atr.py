import numpy as np
import matplotlib.pyplot as plt
import sympy as sp



def magnetic_parameters():
    gamma = 0.105 # cm^-1 / kOe
    H0 = 3 # kOe
    Ha = 197 # kOe
    He = 533 # kOe
    magnetisation = 560 # G

    return gamma, H0, Ha, He, magnetisation


def calculate_permeability(frequency, gamma, H0, Ha, He, magnetisation):

    resonant_frequency = np.sqrt(Ha * gamma**2 * (Ha + 2 * He))
    damping_parameter = np.linspace(1.27e-4, 1.27e-6, len(frequency))
    tau = 1/resonant_frequency * 1/damping_parameter

    X = 1/ ((frequency + H0 * gamma + 1j/tau)**2 - resonant_frequency**2)
    Y = 1/ ((frequency - H0 * gamma + 1j/tau)**2 - resonant_frequency**2)

    mu_3 = 1 - 4 * np.pi * gamma**2 * magnetisation * Ha * (X+Y)
    mu_t = 4 * np.pi * gamma**2 * magnetisation * Ha * (Y-X)

    return mu_3, mu_t


def plot_permeability_components(frequency, mu_3, mu_t):
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(2)
    fig.suptitle('Permeabilities')
    axs[0].plot(frequency, mu_3.real, label = 'Real')
    axs[0].plot(frequency, mu_3.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{3}$')
    axs[0].legend()

    axs[1].plot(frequency, mu_t.real, label = 'Real')
    axs[1].plot(frequency, mu_t.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$\mu_{t}$')
    axs[1].legend()

    plt.show()
    plt.close()


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
    element_9 = 1. # (k0**2 * epsilon - kx**2 - ky**2)

    matrix = sp.Matrix([element_1, element_2, element_3, element_4, element_5, element_6, element_7, element_8, element_9]).reshape(3,3)
    
    new_matrices = [matrix.subs({
        k0 : 2 * np.pi * omega,
        kx : np.sin(np.pi/4.) * 2 * np.pi * omega * np.sqrt(11.56),
        kz : 0 + 0j,
        epsilon : 5.5 + 0j,
        mu_3 : item_1 + 0j,
        mu_t : item_2 + 0j
    }).subs({ omega : item_3 + 0j}) for (item_1, item_2, item_3) in zip(permeability_3, permeability_t, frequency)]

    return new_matrices


def solve_ky(matrices):
    determinants = [item.det() for item in matrices]
    ky_solutions = [sp.solve(item, sp.symbols('k_y')) for item in determinants]

    ky_solutions = np.asarray(ky_solutions).astype(complex)
    
    return ky_solutions


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


def no_phi(wavenumber, permeability_3, permeability_t, magnet_permittivity=5.5, prism_permittivity=11.56, incident_angle=np.pi/4.):

    k0 = wavenumber * 2 * np.pi
    kx = k0 * np.sqrt(prism_permittivity) * np.sin(incident_angle)
    
    ky = 0. - np.sqrt((k0**2 * magnet_permittivity * (permeability_3**2 - permeability_t**2)/permeability_3) - kx**2)

    ky_other = 0. + np.sqrt((k0**2 * magnet_permittivity * (permeability_3**2 - permeability_t**2)/permeability_3) - kx**2)

    return ky, ky_other



def main():
    frequency = np.linspace(52.01, 54, 250)
    parameters = magnetic_parameters()
    permeability_3, permeability_t = calculate_permeability(*(frequency, *parameters))

    matrices = retrieve_matrix(frequency, permeability_3, permeability_t)
    ky_solutions = solve_ky(matrices)
    ky_matrix = ky_solutions[:,0]
    ky_matrix_other = ky_solutions[:,1]

    ky_analytical, ky_other_analytical = no_phi(frequency, permeability_3, permeability_t)

    plot_analytical_and_matrix(frequency, ky_analytical, ky_other_analytical, ky_matrix, ky_matrix_other)


if __name__ == "__main__":
    main()