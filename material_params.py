import tensorflow as tf
from device_config import run_on_device

class AnisotropicMaterial(object):
    
    def __init__(self, frequency_length, run_on_device_decorator):
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def permittivity_calc(self, high_freq, omega_Tn, gamma_Tn, omega_Ln, gamma_Ln):

        frequency = tf.expand_dims(self.frequency, 0)
        omega_Ln_expanded = tf.expand_dims(omega_Ln, 1)
        gamma_Ln_expanded = tf.expand_dims(gamma_Ln, 1)
        omega_Tn_expanded = tf.expand_dims(omega_Tn, 1)
        gamma_Tn_expanded = tf.expand_dims(gamma_Tn, 1)
        complex_one_j = tf.constant(1j, dtype=tf.complex64)

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



class Quartz(AnisotropicMaterial):

    def __init__(self, frequency_length, run_on_device_decorator):
        super().__init__(frequency_length, run_on_device_decorator)
        self.name = "Quartz"
        self.frequency = tf.cast(tf.linspace(410.0, 600.0, self.frequency_length), dtype=tf.complex64)

    @run_on_device
    def permittivity_parameters(self):
    
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(2.356, dtype = tf.complex64),
                "omega_Tn" : tf.constant([393.5, 450.0, 695.0, 797.0, 1065.0, 1158.0], dtype = tf.complex64),
                "gamma_Tn" : tf.constant([2.1, 4.5, 13.0, 6.9, 7.2, 9.3], dtype = tf.complex64),
                "omega_Ln" : tf.constant([403.0, 507.0, 697.6, 810.0, 1226.0, 1155.0], dtype = tf.complex64),
                "gamma_Ln" : tf.constant([2.8, 3.5, 13.0, 6.9, 12.5, 9.3], dtype = tf.complex64)
            },
            "extraordinary": {
                "high_freq": tf.constant(2.383,dtype = tf.complex64),
                "omega_Tn" : tf.constant([363.5, 487.5, 777.0, 1071.0], dtype = tf.complex64),
                "gamma_Tn" : tf.constant([4.8, 4.0, 6.7, 6.8], dtype = tf.complex64),
                "omega_Ln" : tf.constant([386.7, 550.0, 790.0, 1229.0], dtype = tf.complex64),
                "gamma_Ln" : tf.constant([7.0, 3.2, 6.7, 12.0], dtype = tf.complex64)
            }
        }

        return parameters


class Ambient_Incident_Prism(object):

    def __init__(self, permittivity, theta, run_on_device_decorator) :
        self.permittivity = permittivity
        self.theta = theta
        self.run_on_device = run_on_device_decorator

    @run_on_device
    def construct_tensor(self):
        n = tf.sqrt(self.permittivity) 
        cos_theta = tf.cos(self.theta)
        n_cos_theta = n * cos_theta

        # Combine updates into a single tensor with shape [180, 4, 4]
        element1 = tf.stack([tf.zeros_like(self.theta), tf.ones_like(self.theta), -1./n_cos_theta, tf.zeros_like(self.theta)], axis=-1)
        element2 = tf.stack([tf.zeros_like(self.theta), tf.ones_like(self.theta), 1./n_cos_theta, tf.zeros_like(self.theta)],axis=-1)
        element3 = tf.stack([1./cos_theta, tf.zeros_like(self.theta), tf.zeros_like(self.theta), 1./n * tf.ones_like(self.theta)],axis=-1)
        element4 = tf.stack([-1./cos_theta, tf.zeros_like(self.theta), tf.zeros_like(self.theta), 1./n * tf.ones_like(self.theta)],axis=-1)
        
        matrix = tf.stack([element1, element2, element3, element4], axis=1)

        return 0.5 * tf.cast(matrix, dtype=tf.complex64)


    @run_on_device
    def construct_tensor_singular(self):
        n = tf.sqrt(self.permittivity) 
        cos_theta = tf.cos(self.theta)
        n_cos_theta = n * cos_theta

        # Combine updates into a single tensor with shape [180, 4, 4]
        element1 = tf.stack([0., 1., -1./n_cos_theta, 0.])
        element2 = tf.stack([0., 1., 1./n_cos_theta, 0.])
        element3 = tf.stack([1./cos_theta, 0., 0., 1./n])
        element4 = tf.stack([-1./cos_theta, 0., 0., 1./n])
        
        matrix = tf.stack([element1, element2, element3, element4], axis=0)

        return 0.5 * tf.cast(matrix, dtype=tf.complex64)


class Air(object):
    def __init__(self, run_on_device_decorator):
        self.run_on_device = run_on_device_decorator
        pass
    
    @run_on_device
    def construct_tensor_singular(self):
        tensor = tf.constant(
            [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
            ],
        dtype= tf.complex64)
        return tensor
