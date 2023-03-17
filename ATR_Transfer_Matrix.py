import numpy as np
from scipy.linalg import expm
import material_params
import plots

def anisotropy_rotation(matrix, phi, beta):
    rotation_y = np.array([
        [np.cos(phi), 0., np.sin(phi)],
        [0., 1., 0.],
        [-np.sin(phi), 0., np.cos(phi)]
    ])

    rotation_z = np.array([
        [np.cos(beta), -np.sin(beta), 0.],
        [np.sin(beta), np.cos(beta), 0.],
        [0., 0., 1.]
    ])

    rotation_x = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    total_rotation = rotation_z @ rotation_y @ rotation_x

    return total_rotation @ matrix @ total_rotation.T


def air_tensor():

    both_tensor = np.array(
        [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ],
    )

    return both_tensor


def layer_matrix(eps_tensor, mu_tensor, kx, k0, thickness, quartz = False):

    delta = np.zeros((len(eps_tensor), len(kx), 4, 4), dtype=np.complex128)

    kx = kx[:, np.newaxis]
    eps_tensor = eps_tensor[np.newaxis, ...]
    mu_tensor = mu_tensor[np.newaxis, ...]

    delta[..., 0, 0] = -kx * (eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2])
    delta[..., 0, 1] = kx * ((mu_tensor[..., 1, 2] / mu_tensor[..., 2, 2]) - (eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 0, 2] = (mu_tensor[..., 1, 0] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]))
    delta[..., 0, 3] = mu_tensor[..., 1, 1] - (mu_tensor[..., 1, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - (kx ** 2) / eps_tensor[..., 2, 2]

    delta[..., 1, 0] = np.zeros_like(kx)
    delta[..., 1, 1] = -kx * mu_tensor[..., 0, 2] / mu_tensor[..., 2, 2]
    delta[..., 1, 2] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 0])
    delta[..., 1, 3] = ((mu_tensor[..., 0, 2] * mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]) - mu_tensor[..., 0, 1])

    delta[..., 2, 0] = ((eps_tensor[..., 1, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]) - eps_tensor[..., 1, 0])
    delta[..., 2, 1] = (kx ** 2) / mu_tensor[..., 2, 2] - eps_tensor[..., 1, 1] + (eps_tensor[..., 1, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2])
    delta[..., 2, 2] = -kx * mu_tensor[..., 2, 0] / mu_tensor[..., 2, 2]
    delta[..., 2, 3] = kx * ((eps_tensor[..., 1, 2] / eps_tensor[..., 2, 2]) - (mu_tensor[..., 2, 1] / mu_tensor[..., 2, 2]))

    delta[..., 3, 0] = (eps_tensor[..., 0, 0] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 0] / eps_tensor[..., 2, 2]))
    delta[..., 3, 1] = (eps_tensor[..., 0, 1] - (eps_tensor[..., 0, 2] * eps_tensor[..., 2, 1] / eps_tensor[..., 2, 2]))
    delta[..., 3, 2] = np.zeros_like(kx)
    delta[..., 3, 3] = -kx * eps_tensor[..., 0, 2] / eps_tensor[..., 2, 2]

    delta = np.transpose(delta, (1,0,2,3))

    eigenvalues, vector = np.linalg.eig(delta)

    if quartz:
        
        order = (1.j * eigenvalues).argsort(axis=-1)[..., np.newaxis]
        vector = np.transpose(vector, (0, 1, 3, 2))
        vector = np.take_along_axis(vector, order, axis=-2)

        vector[..., 2:4, :] = 0
        vector[..., 2, :] = vector[..., 1, :]
        vector[..., 1, :] = 0

        vector = np.transpose(vector, (0, 1, 3, 2))

        return vector
    
    eye_matrix = np.eye(4, dtype=np.complex128)
    eigenvalues_diag = eye_matrix * eigenvalues[..., np.newaxis]

    k0_expanded = k0[:, np.newaxis, np.newaxis]
    partial = expm(1.j * eigenvalues_diag * k0_expanded * thickness)
    partial_complete = vector @ partial @ np.linalg.inv(vector)

    return partial_complete
    

def reflection_coefficients(T):

    bottom_line = (T[...,0,0] * T[...,2,2] - T[...,0,2] * T[...,2,0])
    
    r_pp = (T[...,0,0] * T[...,3,2] - T[...,3,0] * T[...,0,2]) / bottom_line
    
    r_ps = (T[...,0,0] * T[...,1,2] - (T[...,1,0] * T[...,0,2])) / bottom_line

    r_sp = (T[...,3,0] * T[...,2,2] - T[...,3,2] * T[...,2,0]) / bottom_line

    r_ss = (T[...,1,0] * T[...,2,2] - T[...,1,2] * T[...,2,0]) / bottom_line  
    
    return np.array([r_pp, r_ps, r_sp, r_ss])


def main_quartz_contour():
    eps_prism = 5.5
    air_gap_thickness = 0.
    
    quartz = material_params.Quartz(300)
    frequency = quartz.frequency
    k0 = frequency * 2. * np.pi

    incident_angle = np.linspace(-np.pi/2., np.pi/2., quartz.frequency_length)
    kx = np.sqrt(eps_prism) * np.sin(incident_angle)
    
    anisotropy_rotation_y = np.radians(90)
    rotation_z = np.radians(0)

    quartz_tensor = quartz.fetch_permittivity_tensor()
    quartz_tensor = anisotropy_rotation(quartz_tensor, anisotropy_rotation_y, rotation_z)
    non_magnetic_tensor = np.tile(air_tensor(), (quartz.frequency_length, 1, 1))

    prism_layer = material_params.Ambient_Incident_Prism(eps_prism, incident_angle).construct_tensor()
    air_layer = np.linalg.inv(layer_matrix(non_magnetic_tensor, non_magnetic_tensor, kx, k0, air_gap_thickness))
    quartz_layer = layer_matrix(quartz_tensor, non_magnetic_tensor, kx, 0, 0, quartz=True)

    transfer = prism_layer @ air_layer @ quartz_layer

    reflectivities = reflection_coefficients(transfer)

    plots.contour_theta(frequency, incident_angle, air_gap_thickness, anisotropy_rotation_y, rotation_z, reflectivities)


if __name__ == "__main__":
    main_quartz_contour()