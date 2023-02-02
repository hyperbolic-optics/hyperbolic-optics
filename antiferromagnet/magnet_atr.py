import numpy as np
import matplotlib.pyplot as plt


def magnetic_parameters():
    gamma = 0.105 # cm^-1 / kOe
    H0 = 3 # kOe
    Ha = 197 # kOe
    He = 533 # kOe
    magnetisation = 0.56 # G

    resonant_frequency = np.sqrt(Ha * gamma**2. * (Ha + 2. * He))
    damping_parameter = 1.27e-4
    tau = 1./resonant_frequency * 1./damping_parameter

    return gamma, H0, Ha, magnetisation, resonant_frequency, tau


def calculate_permeability(frequency, gamma, H0, Ha, magnetisation, resonant_frequency, tau):

    X = 1/ ((frequency + H0 * gamma + 1j/tau)**2. - resonant_frequency**2.)
    Y = 1/ ((frequency - H0 * gamma + 1j/tau)**2. - resonant_frequency**2.)
    
    mu_3 = 1. - 4. * np.pi * gamma**2. * magnetisation * Ha * (X+Y)
    mu_t = 4. * np.pi * gamma**2. * magnetisation * Ha * (Y-X)

    mu_v = (mu_3**2. - mu_t**2.)/mu_3

    return mu_3, mu_t, mu_v


def fetch_wavevectors_permittivity(frequency, theta, phi):
    prism_permittivity = 11.56
    
    k0 = frequency * np.pi * 2.
    kx = k0 * np.sqrt(prism_permittivity) * np.sin(theta)
    kz = k0 * np.sqrt(prism_permittivity) * np.cos(phi)

    ky1 = np.sqrt(k0**2 * prism_permittivity - kx**2 + 0.j)
    ky2 = np.sqrt(k0**2 - kx**2 + 0.j)
    
    magnet_permittivity = 5.5
    permittivity_2 = 1.


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
    np.set_printoptions(suppress=True)
    
    positive_roots = []
    original = np.round(roots[0],9) + 0. + 0.j
    original = original[np.where(original.imag>=0.)]
    original = original[np.where(original.real>=0.)]
    if original[1].real > original[0].real:
        original = np.flip(original)
    positive_roots.append(original)
    
    previous = original
    
    for i, item in enumerate(roots[1:]):
        item = np.round(item,8) + 0. + 0.j # to deal with issue of very very small float errors
        item = item[np.where(item.imag>=0.)]
        item = item[np.where(item.real>=0.)]

        condition_1 = abs(item[0].imag - previous[1].imag) < abs(item[0].imag - previous[0].imag)
        condition_2 = abs(item[0].real - previous[1].real) < abs(item[0].real - previous[0].real)

        if condition_1 and condition_2:
            item = np.flip(item)

        positive_roots.append(item)
        previous = item

    positive_roots = np.asarray(positive_roots)
    kya = positive_roots[:, 0]
    kyb = positive_roots[:, 1]

    return kya, kyb


def solve_A_B(mu_3, mu_t, k0, kx, ky, kz, permittivity):
    
    element_1 = permittivity * mu_3 * k0**2 - kz**2 - ky**2
    element_2 = 1.j * mu_t * permittivity * k0**2 + kx * ky
    element_3 = -1.j * mu_t * permittivity * k0**2 + kx * ky
    element_4 = permittivity * mu_3 * k0**2 - kz**2 -kx**2

    inverses = []
    for a,b,c,d in zip(element_1, element_2, element_3, element_4):
        matrix = np.array([[a,b],[c,d]])
        inverses.append(np.linalg.inv(matrix))

    inverses = np.asarray(inverses)
    column_matrix = np.array([[-1. * kx * kz],[-1. * ky * kz]]).T


    A = []
    B = []
    for i, j in zip(inverses, column_matrix):
        A.append(i.dot(j.reshape(2,1))[0][0])
        B.append(i.dot(j.reshape(2,1))[1][0])

    return np.asarray(A), np.asarray(B)


def ATR_matrix(kx, ky1, ky2, kz, ky3a, ky3b, Aa, Ab, Ba, Bb, prism_permittivity, permittivity_2, permittivity_magnet, d = 1.5e-4):

    M11 = - 1. * np.exp(1.j * ky1 * d)
    M13 = np.exp(-1.j * ky2 * d)
    M15 = np.exp(1.j * ky2 * d)
    M12, M14, M16, M17, M18 = 0., 0., 0., 0., 0.

    M22 = -1. * np.exp(1.j * ky1 * d)
    M24 = np.exp(-1.j * ky2 * d)
    M26 = np.exp(1.j * ky2 * d)
    M21, M23, M25, M27, M28 = 0., 0., 0., 0., 0.

    M31 = ((kx * kz) / (prism_permittivity * ky1)) * np.exp(1.j * ky1 * d)
    M32 = prism_permittivity**-1. * np.exp(1.j * ky1 * d) * (((kz**2)/ky1) - ky1 )
    M33 = ((-1. * kx * kz) / (permittivity_2 * ky2)) * np.exp(-1.j * ky2 * d)
    M34 = (-1./permittivity_2) * (np.exp(-1.j * ky2 * d)) * (kz**2/ky2 - ky2)
    M35 = ((-1. * kx * kz)/(permittivity_2 * ky2)) * np.exp(1.j * ky2 * d)
    M36 = permittivity_2**-1. * np.exp(1.j * ky2 * d) * (((-1. * kz**2)/ky2) + ky2 )
    M37, M38 = 0., 0.

    M41 = (1./prism_permittivity) * np.exp(1.j * ky1 * d) * (ky1 - (kx**2)/(ky1))
    M42 = ((-1. * kx * kz)/(prism_permittivity * ky1)) * np.exp(1.j * ky1 * d)
    M43 = (1./ permittivity_2) * np.exp(-1.j * ky2 * d) * (ky2 + (kx**2)/(ky2))
    M44 = ((kx * kz)/(permittivity_2 * ky2)) * np.exp(-1.j * ky2 * d)
    M45 = (1./ permittivity_2) * np.exp(1.j * ky2 * d) * ((kx**2)/(ky2) - ky2 )
    M46 = ((kx * kz)/(permittivity_2 * ky2)) * np.exp(1.j * ky2 * d)
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
    M74 = (-1./permittivity_2) * (kz**2/ky2 - ky2)
    M75 = kx * kz / (ky2 * permittivity_2)
    M76 = (-1./permittivity_2) * (kz**2/ky2 + ky2)
    M77 = (1./ permittivity_magnet) * (ky3a - kz * Ba)
    M78 = (1./ permittivity_magnet) * (ky3b - kz * Bb)
    M71, M72 = 0., 0.

    M83 = (-1./permittivity_2) * (ky2 + kx**2/ky2)
    M84 = -1. * kz * kx / (permittivity_2 * ky2)
    M85 = (1./permittivity_2) * (ky2 + kx**2/ky2)
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

    column_matrix = np.array([np.exp(-1.j * ky1 * d), np.exp(-1.j * ky1 * d), 0., 0., 0., 0., 0., 0.])

    small_r = np.linalg.inv(reflection_matrix).dot(column_matrix)

    reflection_values = small_r[0] * np.conj(small_r[0]), small_r[1] * np.conj(small_r[1])
    
    return reflection_values


def fetch_reflectivity(kx,ky1,ky2,kz, ky3a, ky3b, Aa, Ab, Ba, Bb, prism_permittivity, permittivity_2, permittivity_magnet, d = 1.5e-4):
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
        reflectivity_x.append(np.real(reflection_x))
        reflectivity_z.append(np.real(reflection_z))
    return np.asarray(reflectivity_x), np.asarray(reflectivity_z)


def plot_ky(frequency, zeroth_solutions, first_solutions):
    
    plt.rcParams["figure.figsize"] = (8,7)
    fig, axs = plt.subplots(2)
    fig.suptitle('$k_{y}$ Solutions')
    axs[0].plot(frequency, zeroth_solutions.real, label = 'Real')
    axs[0].plot(frequency, zeroth_solutions.imag, label = 'Imaginary')
    axs[0].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[0].legend()

    axs[1].plot(frequency, first_solutions.real, label = 'Real')
    axs[1].plot(frequency, first_solutions.imag, label = 'Imaginary')
    axs[1].set(xlabel='$\omega/2\pi c (cm^{-1})$', ylabel = '$k_{y}$')
    axs[1].legend()

    plt.show()
    plt.close()


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


def plot_reflectivity(wavenumber, reflectivity, incident_angle):
    plt.rcParams["figure.figsize"] = (6,6)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.plot(reflectivity, wavenumber)
    plt.xlabel('Reflectivity')
    plt.ylabel('$\omega/2\pi c (cm^{-1})$')
    #plt.xlim(0,1)
    plt.title(r'ATR for $\theta$ = ${}^\circ$ S-Polarised'.format(np.degrees(incident_angle)))
    plt.tight_layout()
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
    
    rotation = np.degrees(rotation)
    x_axis = np.sqrt(11.56) * np.sin(x_axis)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.figure()
    plt.pcolormesh(x_axis, wavenumber, reflectivity, cmap = 'magma')
    cbar = plt.colorbar()
    cbar.set_label("Reflectivity")
    plt.title("ATR for $\phi$ = " + str(int(round(rotation,2))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")
    plt.xlabel('$k_x / k_0 $')
    plt.ylabel('$\omega/2\pi c (cm^{-1})$')
    plt.show()
    #plt.savefig(f"plots/ATR_{int(rotation)}_degrees.png")
    plt.close()


def analytical_reflection(kx, ky1, ky3, mu_3, mu_t, mu_v):

    top_line = ky1 * mu_v  - ky3 - 1.j * kx * (mu_t/mu_3)

    bottom_line = ky1 * mu_v  + ky3 + 1.j * kx * (mu_t/mu_3)

    small_r = top_line/bottom_line

    reflection = small_r * np.conj(small_r)

    return reflection


def main_heatmap_analytical():
    frequency = np.linspace(52.01, 54., 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))
    
    phi = 0.
    incident_angles = np.linspace(-np.pi/2., np.pi/2., 300)

    reflectivities = []

    for theta in incident_angles:
        k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
        ky_3 = np.sqrt(magnet_permittivity * mu_v * k0**2 - (kx**2))
        reflectivity = analytical_reflection(kx, ky1, ky_3, mu_3, mu_t, mu_v)
        reflectivities.append(reflectivity)
    
    reflectivities = np.asarray(reflectivities).reshape(len(frequency),len(reflectivities)).astype(float)

    contour_reflection(frequency,incident_angles, reflectivities.T, 0., phi)


def main_ky():
    kyas = []
    kybs = []
    frequency = np.linspace(52.01, 54., 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.57
    incident_angles = np.linspace(-1.57,1.57, 300)

    for theta in incident_angles:
        k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
        ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

        kya, kyb, = restructure_roots(ky_roots)
        kyas.append(kya)
        kybs.append(kyb)

    wavevectors = np.asarray(kyas).reshape(len(frequency),len(kyas))
    contour_wavevector(frequency, incident_angles, wavevectors.T.real, 0., phi)


def main_analytical_reflection():
    frequency = np.linspace(52.01, 54., 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.57
    incident_angles = np.linspace(-1.57, 1.57, 300)
    reflectivities = []

    for theta in incident_angles:
        k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
        ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

        kya, kyb, = restructure_roots(ky_roots)

        Aa, Ba = solve_A_B(mu_3, mu_t, k0, kx, kya, kz, magnet_permittivity)
        Ab, Bb = solve_A_B(mu_3, mu_t, k0, kx, kyb, kz, magnet_permittivity)
        
        reflectivity = analytical_reflection(kx, ky1, kya, mu_3, mu_t, mu_v)
        reflectivities.append(reflectivity)
    
    reflectivities = np.asarray(reflectivities)
    contour_reflection(frequency,incident_angles, reflectivities.T.real, 0., phi)
    

def main_matrix():
    frequency = np.linspace(52.01, 54., 300)
    parameters = magnetic_parameters()
    mu_3, mu_t, mu_v = calculate_permeability(*(frequency, *parameters))

    phi = 1.57
    theta = np.pi/4.

    k0, kx, kz, ky1, ky2, prism_permittivity, permittivity_2, magnet_permittivity = fetch_wavevectors_permittivity(frequency, theta, phi)
    ky_roots = retrieve_roots(mu_3,mu_t, k0, kx, kz, magnet_permittivity)

    kya, kyb, = restructure_roots(ky_roots)

    Aa, Ba = solve_A_B(mu_3, mu_t, k0, kx, kya, kz, magnet_permittivity)
    Ab, Bb = solve_A_B(mu_3, mu_t, k0, kx, kyb, kz, magnet_permittivity)
    
    # reflectivity_x, reflectivity_z = fetch_reflectivity(
    #     kx,
    #     ky1,
    #     ky2,
    #     kz, 
    #     kya, 
    #     kyb, 
    #     Aa, 
    #     Ab, 
    #     Ba, 
    #     Bb, 
    #     prism_permittivity,
    #     permittivity_2, 
    #     magnet_permittivity, 
    #     d = 1.e-4
    #     )
    # print(reflectivity)
    
    #plot_reflectivity(frequency, reflectivity, theta)


if __name__ == "__main__":
    main_matrix()