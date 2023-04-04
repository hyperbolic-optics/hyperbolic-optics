import numpy as np
import tensorflow as tf
from device_config import run_on_device


class Quartz(object):

    def __init__(self, frequency_length, run_on_device_decorator):
        self.frequency_length = frequency_length
        self.name = "Quartz"
        self.frequency = tf.cast(tf.linspace(410.0, 600.0, frequency_length), dtype=tf.complex128)
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def permittivity_parameters(self):
    
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(2.356, dtype = tf.complex128),
                "omega_Tn" : tf.constant([393.5, 450.0, 695.0, 797.0, 1065.0, 1158.0], dtype = tf.complex128),
                "gamma_Tn" : tf.constant([2.1, 4.5, 13.0, 6.9, 7.2, 9.3], dtype = tf.complex128),
                "omega_Ln" : tf.constant([403.0, 507.0, 697.6, 810.0, 1226.0, 1155.0], dtype = tf.complex128),
                "gamma_Ln" : tf.constant([2.8, 3.5, 13.0, 6.9, 12.5, 9.3], dtype = tf.complex128)
            },
            "extraordinary": {
                "high_freq": tf.constant(2.383,dtype = tf.complex128),
                "omega_Tn" : tf.constant([363.5, 487.5, 777.0, 1071.0], dtype = tf.complex128),
                "gamma_Tn" : tf.constant([4.8, 4.0, 6.7, 6.8], dtype = tf.complex128),
                "omega_Ln" : tf.constant([386.7, 550.0, 790.0, 1229.0], dtype = tf.complex128),
                "gamma_Ln" : tf.constant([7.0, 3.2, 6.7, 12.0], dtype = tf.complex128)
            }
        }

        return parameters

    @run_on_device
    def permittivity_calc(self, high_freq, omega_Tn, gamma_Tn, omega_Ln, gamma_Ln):

        frequency = tf.expand_dims(self.frequency, 0)
        omega_Ln_expanded = tf.expand_dims(omega_Ln, 1)
        gamma_Ln_expanded = tf.expand_dims(gamma_Ln, 1)
        omega_Tn_expanded = tf.expand_dims(omega_Tn, 1)
        gamma_Tn_expanded = tf.expand_dims(gamma_Tn, 1)
        complex_one_j = tf.constant(1j, dtype=tf.complex128)

        top_line = omega_Ln_expanded**2. - frequency**2. - complex_one_j * frequency * gamma_Ln_expanded
        bottom_line = omega_Tn_expanded**2. - frequency**2. - complex_one_j * frequency * gamma_Tn_expanded
        result = top_line / bottom_line

        return high_freq * tf.reduce_prod(result, axis=0)

    @run_on_device
    def permittivity_fetch(self):
        params = self.permittivity_parameters()

        eps_ext = self.permittivity_calc(**params["extraordinary"])
        eps_ord = self.permittivity_calc(**params["ordinary"])

        return eps_ext, eps_ord

    @run_on_device
    def fetch_permittivity_tensor(self):
        eps_ext, eps_ord = self.permittivity_fetch()

        diag_tensors = tf.stack([eps_ord, eps_ord, eps_ext], axis=1)
        eps_tensor = tf.linalg.diag(diag_tensors)
        
        return eps_tensor


# class Ambient_Incident_Prism(object):

#     def __init__(self, permittivity, theta) :
#         self.permittivity = permittivity
#         self.theta = theta

#     def construct_tensor(self):
#         n = np.sqrt(self.permittivity)

#         matrix = np.zeros((self.theta.size, 4, 4))

#         matrix[:, 0, 1] = 1.
#         matrix[:, 1, 1] = 1.
#         matrix[:, 0, 2] = -1./ (n * np.cos(self.theta))
#         matrix[:, 1, 2] = 1./ (n * np.cos(self.theta))
#         matrix[:, 2, 0] = 1./ np.cos(self.theta)
#         matrix[:, 3, 0] = -1./ np.cos(self.theta)
#         matrix[:, 2, 3] = 1./n
#         matrix[:, 3, 3] = 1./n

#         return 0.5 * matrix


#     def construct_tensor_singular(self, permittivity, theta):
#         n = np.sqrt(self.permittivity)

#         matrix = np.zeros((4, 4))

#         matrix[0, 1] = 1.
#         matrix[1, 1] = 1.
#         matrix[0, 2] = -1./ (n * np.cos(self.theta))
#         matrix[1, 2] = 1./ (n * np.cos(self.theta))
#         matrix[2, 0] = 1./ np.cos(self.theta)
#         matrix[3, 0] = -1./ np.cos(self.theta)
#         matrix[2, 3] = 1./n
#         matrix[3, 3] = 1./n

#         return 0.5 * matrix


# class Air(object):
#     def __init__(self):
#         pass

#     def construct_tensor_singular(self):
#         tensor = np.array(
#             [
#             [1., 0., 0.],
#             [0., 1., 0.],
#             [0., 0., 1.]
#             ],
#         )
#         return tensor