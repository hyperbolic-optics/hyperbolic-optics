import numpy as np


class Quartz(object):

    def __init__(self, frequency_length):
        self.frequency_length = frequency_length
        self.name = "Quartz"
        self.frequency = np.linspace(410,600, frequency_length)
    
    def permittivity_parameters(self):
    
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


    def permittivity_calc(self, high_freq, omega_Tn, gamma_Tn, omega_Ln, gamma_Ln):

        frequency = self.frequency[np.newaxis, :]
        omega_Ln_expanded = omega_Ln[:, np.newaxis]
        gamma_Ln_expanded = gamma_Ln[:, np.newaxis]
        omega_Tn_expanded = omega_Tn[:, np.newaxis]
        gamma_Tn_expanded = gamma_Tn[:, np.newaxis]

        top_line = omega_Ln_expanded**2.- frequency**2. - 1.j * frequency * gamma_Ln_expanded
        bottom_line = omega_Tn_expanded**2. - frequency**2. - 1.j * frequency * gamma_Tn_expanded
        result = top_line / bottom_line
        
        return (high_freq * np.prod(result, axis=0))


    def permittivity_fetch(self):

        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])

        return eps_ext, eps_ord


    def fetch_permittivity_tensor(self):
        
        eps_ext, eps_ord = self.permittivity_fetch()

        tensor = np.zeros((self.frequency_length, 3, 3), dtype = np.complex128)

        tensor[:, 0, 0] = eps_ord
        tensor[:, 1, 1] = eps_ord
        tensor[:, 2, 2] = eps_ext
        
        return tensor


class Ambient_Incident_Prism(object):

    def __init__(self, permittivity, theta) :
        self.permittivity = permittivity
        self.theta = theta

    def construct_tensor(self):
        n = np.sqrt(self.permittivity)

        matrix = np.zeros((self.theta.size, 4, 4))

        matrix[:, 0, 1] = 1.
        matrix[:, 1, 1] = 1.
        matrix[:, 0, 2] = -1./ (n * np.cos(self.theta))
        matrix[:, 1, 2] = 1./ (n * np.cos(self.theta))
        matrix[:, 2, 0] = 1./ np.cos(self.theta)
        matrix[:, 3, 0] = -1./ np.cos(self.theta)
        matrix[:, 2, 3] = 1./n
        matrix[:, 3, 3] = 1./n

        return 0.5 * matrix


    def construct_tensor_singular(self, permittivity, theta):
        n = np.sqrt(self.permittivity)

        matrix = np.zeros((4, 4))

        matrix[0, 1] = 1.
        matrix[1, 1] = 1.
        matrix[0, 2] = -1./ (n * np.cos(self.theta))
        matrix[1, 2] = 1./ (n * np.cos(self.theta))
        matrix[2, 0] = 1./ np.cos(self.theta)
        matrix[3, 0] = -1./ np.cos(self.theta)
        matrix[2, 3] = 1./n
        matrix[3, 3] = 1./n

        return 0.5 * matrix


class Air(object):
    def __init__(self):
        pass

    def construct_tensor_singular(self):
        tensor = np.array(
            [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
            ],
        )
        return tensor