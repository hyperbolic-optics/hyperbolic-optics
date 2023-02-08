import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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


def calculate_permeability(frequency, gamma, B0, Ba, magnetisation, resonant_frequency_squared, damping):
    
    X = 1./(resonant_frequency_squared - (frequency + B0 * gamma + 1.j * damping)**2.)
    Y = 1./(resonant_frequency_squared - (frequency - B0 * gamma + 1.j * damping)**2.)

    mu_3 = 1. + constants.mu_0 * gamma**2. * Ba * magnetisation * (X + Y)
    mu_t = constants.mu_0 * gamma**2. * Ba * magnetisation * (X-Y)

    mu_v = (mu_3**2. - mu_t**2.)/mu_3

    return mu_3, mu_t, mu_v


def fetch_wavevectors_permittivity(frequency, theta, phi):
    prism_permittivity = 11.56 + 0.j
    
    k0 = frequency * np.pi * 2. + 0.j
    kx = k0 * np.sqrt(prism_permittivity) * np.sin(theta)
    kz = k0 * np.sqrt(prism_permittivity) * np.cos(phi)

    ky1 = np.sqrt(k0**2. * prism_permittivity - kx**2 + 0.j)
    ky2 = np.sqrt(k0**2. - kx**2. + 0.j)
    
    magnet_permittivity = 5.5 + 0.j
    permittivity_2 = 1. + 0.j


    return k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity


def retrieve_roots(mu_3, mu_t, k0, kx, kz, epsilon):

    fourth_power = epsilon*k0**2.*mu_3

    third_power = 0.

    second_power = epsilon*k0**2.*(-epsilon*k0**2.*mu_3**2. - epsilon*k0**2.*mu_3 + epsilon*k0**2.*mu_t**2. + 2.*kx**2.*mu_3 + kz**2.*mu_3 + kz**2.)

    first_power = 0.

    zero_power = epsilon*k0**2.*(epsilon**2.*k0**4.*mu_3**2. - epsilon**2.*k0**4.*mu_t**2. - epsilon*k0**2.*kx**2.*mu_3**2. - epsilon*k0**2.*kx**2.*mu_3 + epsilon*k0**2.*kx**2.*mu_t**2. - 2.*epsilon*k0**2.*kz**2.*mu_3 + kx**4.*mu_3 + kx**2.*kz**2.*mu_3 + kx**2.*kz**2. + kz**4.)

    ordering = [[
        fourth_power[i],
        third_power,
        second_power[i],
        first_power,
        zero_power[i]
    ] for i in range(0, len(mu_3))]

    #ordering = np.asarray(ordering)

    roots = [np.roots(item) for item in ordering]

    return np.asarray(roots)


def restructure_roots(roots): 
    positive_roots = []
    original = np.round(roots[0],9) + 0. + 0.j
    original = original[np.where(original.imag>=0.)]
    original = original[np.where(original.real>=0.)]

    if original[1].real > original[0].real:
        original = np.flip(original)
    positive_roots.append(original)

    previous = original
    
    for item in roots[1:]:
        item = np.round(item,8) + 0. + 0.j # to deal with issue of very very small float errors
        item = item[np.where(item.imag>=0.)]
        item = item[np.where(item.real>=0.)]

        condition_1 = abs(item[0].imag - previous[1].imag) < abs(item[0].imag - previous[0].imag)
        condition_2 = abs(item[0].real - previous[1].real) < abs(item[0].real - previous[0].real)

        condition_3 = abs(item[1].imag - previous[0].imag) < abs(item[1].imag - previous[1].imag)
        condition_4 = abs(item[1].real - previous[0].real) < abs(item[1].real - previous[1].real)


        if (condition_1 and condition_2) or (condition_3 and condition_4):
            item = np.flip(item)

        positive_roots.append(item)

        previous = item
    

    
    positive_roots = np.asarray(positive_roots)
    kya = positive_roots[:, 0]
    kyb = positive_roots[:, 1]

    return kya, kyb


def solve_A_B(mu_3, mu_t, k0, kx, ky, kz, permittivity):
    
    element_1 = permittivity * mu_3 * k0**2. - (kz**2.) - (ky**2.)
    element_2 = 1.j * mu_t * permittivity * k0**2. + kx * ky
    element_3 = -1.j * mu_t * permittivity * k0**2. + kx * ky
    element_4 = permittivity * mu_3 * k0**2. - (kz**2.) - (kx**2.)

    matrices = np.array([[element_1, element_2], [element_3, element_4]])

    new_matrices = matrices.T

    for i in range(len(new_matrices)):
        new_matrices[i] = new_matrices[i].T

    
    column_matrix = np.array([-1. * kx * kz, -1. * ky * kz]).T.reshape(300,2,1)

    A = []
    B = []
    for i in range(len(new_matrices)):
        element = np.linalg.inv(new_matrices[i]).dot(column_matrix[i])
        A.append(element[0])
        B.append(element[1])

    A, B = np.asarray(A).reshape((len(kx)),), np.asarray(B).reshape((len(kx)),)

    return A,B


def ATR_matrix(kx, ky1, ky2, kz, ky3a, ky3b, Aa, Ab, Ba, Bb, prism_permittivity, permittivity_2, permittivity_magnet, d):
    M11 = -1. * np.exp(1.j * ky1 * d)
    M13 = np.exp(-1.j * ky2 * d)
    M15 = np.exp(1.j * ky2 * d)
    M12, M14, M16, M17, M18 = 0., 0., 0., 0., 0.

    M22 = -1. * np.exp(1.j * ky1 * d)
    M24 = np.exp(-1.j * ky2 * d)
    M26 = np.exp(1.j * ky2 * d)
    M21, M23, M25, M27, M28 = 0., 0., 0., 0., 0.

    M31 = ((kx * kz) / (prism_permittivity * ky1)) * np.exp(1.j * ky1 * d)
    M32 = prism_permittivity**-1. * np.exp(1.j * ky1 * d) * (((kz**2.)/ky1) + ky1 )
    M33 = ((kx * kz) / (permittivity_2 * ky2)) * np.exp(-1.j * ky2 * d)
    M34 = (1./permittivity_2) * (np.exp(-1.j * ky2 * d)) * (kz**2./ky2 + ky2)
    M35 = ((kx * kz)/(permittivity_2 * ky2)) * -1. * np.exp(1.j * ky2 * d)
    M36 = -1. * permittivity_2**-1. * np.exp(1.j * ky2 * d) * (((kz**2.)/ky2) + ky2 )
    M37, M38 = 0., 0.

    M41 = (1./prism_permittivity) * np.exp(1.j * ky1 * d) * (ky1 + (kx**2.)/(ky1))
    M42 = ((kx * kz)/(prism_permittivity * ky1)) * np.exp(1.j * ky1 * d)
    M43 = (1./ permittivity_2) * np.exp(-1.j * ky2 * d) * (ky2 + (kx**2.)/(ky2))
    M44 = ((kx * kz)/(permittivity_2 * ky2)) * np.exp(-1.j * ky2 * d)
    M45 = (-1./ permittivity_2) * np.exp(1.j * ky2 * d) * ((kx**2.)/(ky2) + ky2 )
    M46 = (-1. * (kx * kz)/(permittivity_2 * ky2)) * np.exp(1.j * ky2 * d)
    M47, M48 = 0., 0.

    M53 = 1.
    M55 = 1.
    M57 = -1. * Aa
    M58 = -1. * Ab
    M51, M52, M54, M56 = 0., 0., 0., 0.

    M64 = 1.
    M66 = 1.
    M67 = -1.
    M68 = -1.
    M61, M62, M63, M65 = 0., 0., 0., 0.

    M73 = -1. * kx * kz / (ky2 * permittivity_2)
    M74 = (-1./permittivity_2) * (kz**2./ky2 - ky2)
    M75 = kx * kz / (ky2 * permittivity_2)
    M76 = (1./permittivity_2) * (kz**2./ky2 + ky2)
    M77 = (1./ permittivity_magnet) * (ky3a - kz * Ba)
    M78 = (1./ permittivity_magnet) * (ky3b - kz * Bb)
    M71, M72 = 0., 0.

    M83 = (-1./permittivity_2) * (ky2 + kx**2./ky2)
    M84 = -1. * kz * kx / (permittivity_2 * ky2)
    M85 = (1./permittivity_2) * (ky2 + kx**2./ky2)
    M86 = kz * kx / (permittivity_2 * ky2)
    M87 = (1./ permittivity_magnet) * (Aa * ky3a - kx * Ba)
    M88 = (1./ permittivity_magnet) * (Ab * ky3b - kx * Bb)
    M81, M82 = 0., 0.

    reflection_matrix = np.array([
        [M11, M12, M13, M14, M15, M16, M17, M18],
        [M21, M22, M23, M24, M25, M26, M27, M28],
        [M31, M32, M33, M34, M35, M36, M37, M38],
        [M41, M42, M43, M44, M45, M46, M47, M48],
        [M51, M52, M53, M54, M55, M56, M57, M58],
        [M61, M62, M63, M64, M65, M66, M67, M68],
        [M71, M72, M73, M74, M75, M76, M77, M78],
        [M81, M82, M83, M84, M85, M86, M87, M88],
    ])

    v1 = np.exp(-1.j * ky1 * d)
    v2 = np.exp(-1.j * ky1 * d)
    v3 = np.exp(-1.j * ky1 * d) * (kx * kz)/(prism_permittivity * ky1) + (np.exp(-1.j * ky1 * d)/prism_permittivity) * (ky1 + kz**2./ky1)
    v4 = (np.exp(-1.j * ky1 * d)/prism_permittivity) * (ky1 + kx**2./ky1) + (np.exp(-1.j * ky1 * d)/prism_permittivity) * (kx * kz)/(prism_permittivity * ky1)

    column_matrix = np.array([v1, v2, v3, v4, 0., 0., 0., 0.]).reshape(8,1)

    small_r = np.linalg.inv(reflection_matrix).dot(column_matrix)

    reflection_values = small_r[0] * np.conj(small_r[0]) , small_r[1] * np.conj(small_r[1])
    
    return reflection_values


def fetch_reflectivity(kx,ky1,ky2,kz, ky3a, ky3b, Aa, Ab, Ba, Bb, prism_permittivity, permittivity_2, permittivity_magnet, d):
    reflectivity_x = []
    reflectivity_z = []

    for i in range(len(kx)):
        reflection_x, reflection_z = ATR_matrix(
            kx[i], 
            ky1[i], 
            ky2[i], 
            kz[i], 
            ky3a[i], 
            ky3b[i], 
            Aa[i], 
            Ab[i], 
            Ba[i],
            Bb[i],
            prism_permittivity,
            permittivity_2,
            permittivity_magnet,
            d
            )
        reflectivity_x.append(np.real(reflection_x[0]))
        reflectivity_z.append(np.real(reflection_z[0]))
    return np.asarray(reflectivity_x), np.asarray(reflectivity_z)


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


def plot_ky(frequency, zeroth_solutions, first_solutions):
    axis = np.linspace(0 ,len(frequency), len(frequency))
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(2)
    fig.suptitle('$k_{y}$ Solutions')
    axs[0].plot(axis, zeroth_solutions.real, label = 'Real')
    axs[0].plot(axis, zeroth_solutions.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[0].legend()

    axs[1].plot(axis, first_solutions.real, label = 'Real')
    axs[1].plot(axis, first_solutions.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[1].legend()

    plt.show()
    plt.close()


def plot_AB(frequency, Aa, Ab, Ba, Bb):
    
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(4)
    fig.suptitle('AB Solutions')
    axs[0].plot(frequency, Aa.real, label = 'Real')
    axs[0].plot(frequency, Aa.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$A_a$')
    axs[0].legend()

    axs[1].plot(frequency, Ba.real, label = 'Real')
    axs[1].plot(frequency, Ba.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$B_a$')
    axs[1].legend()

    axs[2].plot(frequency, Ab.real, label = 'Real')
    axs[2].plot(frequency, Ab.imag, label = 'Imaginary')
    axs[2].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$A_b$')
    axs[2].legend()

    axs[3].plot(frequency, Bb.real, label = 'Real')
    axs[3].plot(frequency, Bb.imag, label = 'Imaginary')
    axs[3].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$B_b$')
    axs[3].legend()

    plt.show()
    plt.close()
    

def plot_reflectivity(wavenumber, reflectivity, incident_angle):
    plt.rcParams["figure.figsize"] = (6,6)
    params = {'mathtext.default': 'regular' }  
    plt.rcParams.update(params)  

    fig, ax = plt.subplots()      
    image = ax.plot(reflectivity, wavenumber)
    ax.set_xlabel('Reflectivity')
    ax.set_ylabel('$\omega/2\pi c (cm^{-1})$')
    #plt.xlim(0,1)
    ax.set_ylim(wavenumber.min(), wavenumber.max())
    ax.set_title(r'ATR for $\theta$ = ${}^\circ$ S-Polarised'.format(np.degrees(incident_angle)))
    ax.set_yticks(np.arange(min(wavenumber), max(wavenumber)+0.01, 0.25))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    
    #plt.tight_layout()
    plt.show()
    plt.close()
    

def contour_wavevector(wavenumber, x_axis, wavevector, distance, rotation):
    
    rotation = np.degrees(rotation)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.figure()
    plt.pcolormesh(x_axis, wavenumber, wavevector, cmap = 'magma')
    cbar = plt.colorbar()
    cbar.set_label("Wavevector")
    plt.title("$K_y$ Solution $\phi$ = " + str(int(round(rotation,2))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")
    plt.xlabel('$\\theta$')
    plt.ylabel('$\omega/2\pi c (cm^{-1})$')
    plt.show()
    #plt.savefig(f"plots/ATR_{int(rotation)}_degrees.png")
    plt.close()


def contour_reflection(wavenumber, x_axis, reflectivity, distance, rotation):
    
    fig, ax = plt.subplots()
    rotation = np.degrees(rotation)
    x_axis = np.sqrt(11.56) * np.sin(x_axis)
    
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    image = ax.pcolormesh(x_axis, wavenumber, reflectivity, cmap = 'magma')
    cbar = plt.colorbar(image, ax = ax)
    cbar.mappable.set_clim(0., 1.)
    cbar.set_label("Reflectivity")
    ax.set_title("ATR for $\phi$ = " + str(int(round(rotation,1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")
    ax.set_xlabel('$k_x / k_0 $')
    ax.set_ylabel('$\omega/2\pi c (cm^{-1})$')
    ax.set_yticks(np.arange(min(wavenumber), max(wavenumber)+0.001, 0.25))
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    
    plt.show()
    #plt.savefig(f"plots/ATR_{int(rotation)}_degrees.png")
    plt.close()


def main_ky():
    kyas = []
    kybs = []
    frequency = np.linspace(52., 53.5, 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.56
    incident_angles = np.linspace(-1.57,1.57, 300)

    for theta in incident_angles:
        k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
        ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

        kya, kyb, = restructure_roots(ky_roots)
        kyas.append(kya)
        kybs.append(kyb)


    wavevectors = np.asarray(kyas).reshape(len(frequency),len(kyas)).real
    contour_wavevector(frequency, incident_angles, wavevectors.T.real, 0., phi)
 

def main_matrix_contour():
    frequency = np.linspace(52.0, 53.5, 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.57
    incident_angles = np.linspace(-1.57,1.57, 300)
    d = 1.e-4
    reflectivities = []
    
    for theta in incident_angles:

        k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
        ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

        kya, kyb, = restructure_roots(ky_roots)

        Aa, Ba = solve_A_B(mu_3, mu_t, k0, kx, kya, kz, magnet_permittivity)
        Ab, Bb = solve_A_B(mu_3, mu_t, k0, kx, kyb, kz, magnet_permittivity)

        reflectivity_x, reflectivity_z = fetch_reflectivity(
            kx,
            ky1,
            ky2,
            kz, 
            kya, 
            kyb, 
            Aa, 
            Ab, 
            Ba, 
            Bb, 
            prism_permittivity,
            permittivity_2, 
            magnet_permittivity,
            d
            )

        
        reflectivities.append(reflectivity_x)
    
    reflectivities = np.asarray(reflectivities)
    
    contour_reflection(frequency, incident_angles, reflectivities.T, d, phi)


def main_single_theta():
    frequency = np.linspace(52., 53.5, 300)

    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.57
    theta = 0.01
    d = 1.e-4

    k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
    ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

    kya, kyb = restructure_roots(ky_roots)
    

    Aa, Ba = solve_A_B(mu_3, mu_t, k0, kx, kya, kz, magnet_permittivity)
    Ab, Bb = solve_A_B(mu_3, mu_t, k0, kx, kyb, kz, magnet_permittivity)

    reflectivity_x, reflectivity_z = fetch_reflectivity(
            kx,
            ky1,
            ky2,
            kz, 
            kya, 
            kya, 
            Ba, 
            Ab, 
            Aa, 
            Bb, 
            prism_permittivity,
            permittivity_2, 
            magnet_permittivity,
            d
            )

    plot_reflectivity(frequency, reflectivity_z, theta)



if __name__ == "__main__":
    main_matrix_contour()