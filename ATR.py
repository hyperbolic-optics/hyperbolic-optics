import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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


def anisotropy_rotation_matrix_y(matrix, theta):

    rotation_matrix = np.array([
        [np.cos(theta), 0., np.sin(theta)],
        [0., 1., 0.],
        [-np.sin(theta), 0., np.cos(theta)]
    ])

    return rotation_matrix @ matrix @ (rotation_matrix).T


def anisotropy_rotation_matrix_z(matrix, theta):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]
    ])

    return rotation_matrix @ matrix @ np.linalg.inv(rotation_matrix)


def air_tensor():

    both_tensor = np.array(
        [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ],
    )

    return both_tensor


def prism_tensor(eps_prism):

    tensor = np.array(
        [
        [eps_prism, 0., 0.],
        [0., eps_prism, 0.],
        [0., 0., eps_prism]
        ],
    )

    return tensor


def ambient_incident_prism(eps_prism, theta):
    n = np.sqrt(eps_prism)

    matrix = 0.5 * np.array([
        [0., 1., -n * np.cos(theta), 0.],
        [0., 1., n * np.cos(theta), 0.],
        [1./np.cos(theta), 0., 0., 1./n],
        [-1./np.cos(theta), 0., 0., 1./n]
    ])

    return matrix


def layer_matrix(eps_tensor, kx, k0, thickness, prism = False, quartz = False):

    delta_11 = -1. * kx * (eps_tensor[2,0]/eps_tensor[2,2])
    delta_12 = -1. * kx * (eps_tensor[2,1]/eps_tensor[2,2])
    delta_13 = 0.
    delta_14 = 1. - (kx**2.)/eps_tensor[2,2]

    delta_21 = 0.
    delta_22 = 0.
    delta_23 = -1.
    delta_24 = 0.

    delta_31 = ((eps_tensor[1,2] * eps_tensor[2,0]) / eps_tensor[2,2]) - eps_tensor[1,0]
    delta_32 = (kx**2.) - eps_tensor[1,1] + ((eps_tensor[1,2] * eps_tensor[2,1])/ eps_tensor[2,2])
    delta_33 = 0.
    delta_34 = kx * (eps_tensor[1,2]/eps_tensor[2,2])

    delta_41 = eps_tensor[0,0] - (eps_tensor[0,2] * eps_tensor[2,0] / eps_tensor[2,2])
    delta_42 = eps_tensor[0,1] - (eps_tensor[0,2] * eps_tensor[2,1] / eps_tensor[2,2])
    delta_43 = 0.
    delta_44 = -1. * kx * eps_tensor[0,2]/eps_tensor[2,2]

    delta = np.array(
        [
        [delta_11,delta_12,delta_13,delta_14],
        [delta_21,delta_22,delta_23,delta_24],
        [delta_31,delta_32,delta_33,delta_34],
        [delta_41,delta_42,delta_43,delta_44]
        ]
    )

    eigenvalue, vector = np.linalg.eig(delta)
    if prism:
        return np.linalg.inv(vector)
    
    if quartz:
        return eigenvalue, vector

    partial = expm(np.diag((1.j * eigenvalue * k0 * thickness)))
    partial_complete = vector @ partial @ np.linalg.inv(vector)
    return partial_complete


def value_vector_sort(eigenvalues, eigenvectors):

    positive_eigenvectors = eigenvectors.T[np.where(eigenvalues.imag>0)]

    eigenvector_columns = np.array([
        [positive_eigenvectors[0]],
        [np.zeros_like(eigenvalues)],
        [positive_eigenvectors[1]],
        [np.zeros_like(eigenvalues)]
    ]).T[:,0,:]

    return eigenvector_columns
    

def reflection_coefficients(T): 
    
    r_pp = (T[:,0,0] * T[:,3,2] - T[:,3,0] * T[:,0,2]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])
    
    r_ps = (T[:,0,0] * T[:,1,2] - T[:,1,0] * T[:,0,2]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    r_sp = (T[:,3,0] * T[:,2,2] - T[:,3,2] * T[:,2,0]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    r_ss = (T[:,1,0] * T[:,2,2] - T[:,1,2] * T[:,2,0]) / (T[:,0,0] * T[:,2,2] - T[:,0,2] * T[:,2,0])

    return r_pp, r_ps, r_sp, r_ss


def plot_various_reflectivities(frequency, r_pp, r_ps, r_sp, r_ss):
    reflection = (r_pp * np.conj(r_pp)).real
    reflection6 = (r_ss * np.conj(r_ss)).real
    reflection1 = ((r_ps/r_ss) * np.conj(r_ps/r_ss)).real
    reflection2 = ((r_ps/r_pp) * np.conj(r_ps/r_pp)).real
    reflection3 = ((r_sp/r_ss) * np.conj(r_sp/r_ss)).real
    reflection4 = ((r_sp/r_pp) * np.conj(r_sp/r_pp)).real
    
    plt.rcParams["figure.figsize"] = (8,7)
    fig, (ax, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6,figsize=(12,6), sharey=True)#gridspec_kw={"width_ratios":[1,1, 0.05]})
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Reflectivity for various configurations')

    ax.set_title('$|r_{pp}|^2$')
    ax.plot(reflection, frequency)
    ax.set(xlabel = '$|r_{pp}|^2$')
    ax.set(ylabel='$\omega/2\pi c (cm^{-1})$')

    ax2.set_title('$|r_{ss}|^2$')
    ax2.plot(reflection6, frequency)
    ax2.set(xlabel = '$|r_{ss}|^2$')

    ax3.set_title('$|\dfrac{r_{ps}}{r_{ss}}|^2$')
    ax3.plot(reflection1, frequency)
    ax3.set(xlabel = '$|\dfrac{r_{ps}}{r_{ss}}|^2$')

    ax4.set_title('$|\dfrac{r_{ps}}{r_{pp}}|^2$')
    ax4.plot(reflection2, frequency)
    ax4.set(xlabel = '$|\dfrac{r_{ps}}{r_{pp}}|^2$')

    ax5.set_title('$|\dfrac{r_{sp}}{r_{ss}}|^2$')
    ax5.plot(reflection3, frequency)
    ax5.set(xlabel = '$|\dfrac{r_{sp}}{r_{ss}}|^2$')

    ax6.set_title('$|\dfrac{r_{sp}}{r_{pp}}|^2$')
    ax6.plot(reflection4, frequency)
    ax6.set(xlabel = '$|\dfrac{r_{sp}}{r_{pp}}|^2$')

    # plt.tight_layout()
    plt.show()


def contour_reflection(wavenumber, x_axis, reflectivity, distance, rotation):
    
    fig, ax = plt.subplots()
    rotation = np.degrees(rotation)
    x_axis = np.sqrt(5.5) * np.sin(x_axis)
    
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    image = ax.pcolormesh(x_axis, wavenumber, reflectivity, cmap = 'magma')
    cbar = plt.colorbar(image, ax = ax)
    cbar.mappable.set_clim(0., 1.)
    cbar.set_label("Reflectivity")
    ax.set_title("ATR for $\phi$ = " + str(int(round(rotation,1))) + "$^\circ$, $d = " + str(round(distance * 1e4, 3)) +"\mu m$")
    ax.set_xlabel('$k_x / k_0 $')
    ax.set_ylabel('$\omega/2\pi c (cm^{-1})$')
    
    plt.show()
    #plt.savefig(f"plots/ATR_{int(rotation)}_degrees.png")
    plt.close()


def contour_all_polarisations(wavenumber, x_axis, distance, anisotropy_rotation_x, rotation_z, reflectivities):

    r_pp = reflectivities[:,0]
    r_ps = reflectivities[:,1]
    r_sp = reflectivities[:,2]
    r_ss = reflectivities[:,3]

    R_pp = (r_pp * np.conj(r_pp)).real.T
    R_ps = (r_ps * np.conj(r_ps)).real.T
    R_sp = (r_sp * np.conj(r_sp)).real.T
    R_ss = (r_ss * np.conj(r_ss)).real.T

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    fig, ax = plt.subplots(2,2, figsize=(9, 7))

    x_axis = np.sqrt(5.5) * np.sin(x_axis)
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
    # cbar_p.mappable.set_clim(0., 1.)
    cbar_ps.set_label("$|r_{ps}|^2$")
    ax[0,1].set_title("$|r_{ps}|^2$")
    ax[0,1].set_xlabel('$k_x / k_0 $')
    ax[0,1].set_ylabel('$\omega/2\pi c (cm^{-1})$')

    reflection_sp = ax[1,0].pcolormesh(x_axis, wavenumber, R_sp, cmap = 'magma')
    cbar_sp = plt.colorbar(reflection_sp, ax = ax[1,0])
    # cbar_pp.mappable.set_clim(0., 1.)
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
    
    plt.tight_layout()
    plt.show()
    plt.close()



def main_contour():
    frequency = np.linspace(410,600,300)
    
    anisotropy_rotation_x = np.radians(90)
    rotation_z = 0.
    d = 1.5e-4

    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_y(quartz_tensor, anisotropy_rotation_x)
    quartz_tensor = anisotropy_rotation_matrix_z(quartz_tensor, rotation_z)

    eps_prism = 5.5
    incident_angle = np.linspace(-np.pi/2., np.pi/2., len(frequency))
    k0 = frequency * 2. * np.pi

    reflectivities = []

    for angle in incident_angle:
        prism_layer = ambient_incident_prism(eps_prism, angle)
        kx = np.sqrt(eps_prism) * np.sin(angle)
        transfer_matrices = []

        for i in range(0,len(frequency)):
            value, vector = layer_matrix(quartz_tensor[i], kx, 0, 0, quartz=True)
            quartz_eigenvectors = value_vector_sort(value, vector)

            air_layer = layer_matrix(air_tensor(), kx, k0[i], d)

            transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_eigenvectors
            transfer_matrices.append(transfer)

        transfer_matrices = np.asarray(transfer_matrices)

        r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)

        reflectivities.append([r_pp,r_ps,r_sp,r_ss])


    reflectivities = np.asarray(reflectivities)

    # contour_reflection(frequency, incident_angle, reflectivities.T, d, anisotropy_rotation)

    contour_all_polarisations(frequency, incident_angle, d, anisotropy_rotation_x, rotation_z, reflectivities)



def main():
    frequency = np.linspace(410,600,300)
    
    params = permittivity_parameters()
    eps_ext, eps_ord = permittivity_fetch(frequency, params)
    quartz_tensor = construct_quartz_tensors(eps_ord, eps_ext)
    quartz_tensor = anisotropy_rotation_matrix_y(quartz_tensor, np.radians(45))
    quartz_tensor = anisotropy_rotation_matrix_z(quartz_tensor, np.radians(45))


    eps_prism = 5.5
    incident_angle = np.radians(30)
    
    k0 = frequency * 2. * np.pi
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)

    transfer_matrices = []
    prism_layer = ambient_incident_prism(eps_prism, incident_angle)

    for i in range(0, len(frequency)):
        value, quartz_eigenvectors = layer_matrix(quartz_tensor[i], kx, 0, 0, quartz=True)
        quartz_eigenvectors = value_vector_sort(value, quartz_eigenvectors)

        air_layer = layer_matrix(air_tensor(), kx, k0[i], 1.5e-4)

        transfer = prism_layer @ np.linalg.inv(air_layer) @ quartz_eigenvectors
        transfer_matrices.append(transfer)

    transfer_matrices = np.asarray(transfer_matrices)

    r_pp, r_ps, r_sp, r_ss = reflection_coefficients(transfer_matrices)
    plot_various_reflectivities(frequency, r_pp, r_ps, r_sp, r_ss)

if __name__ == "__main__":
    main_contour()