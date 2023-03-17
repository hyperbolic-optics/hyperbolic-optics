import numpy as np
from scipy.linalg import expm



def layer_matrix_incidence(eps_tensor, mu_tensor, kx, k0, thickness, quartz = False):

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
   

def layer_matrix_incidence_azimuth(eps_tensor, mu_tensor, kx, k0, thickness, quartz=False):

    delta = np.zeros((eps_tensor.shape[0], eps_tensor.shape[1], 4, 4), dtype=np.complex128)

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

    k0_expanded = k0[:, np.newaxis, np.newaxis, np.newaxis]
    partial = expm(1.j * eigenvalues_diag * k0_expanded * thickness)
    partial_complete = vector @ partial @ np.linalg.inv(vector)

    return partial_complete

