import tensorflow as tf
from device_config import run_on_device
from scipy import constants

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
    

class Calcite(AnisotropicMaterial):

    def __init__(self, frequency_length, run_on_device_decorator):
        super().__init__(frequency_length, run_on_device_decorator)

    @run_on_device
    def permittivity_parameters(self):
    
        parameters = {
            "ordinary": {
                "high_freq": tf.constant(2.7, dtype = tf.complex64),
                "omega_Tn" : tf.constant([712, 1407., 297., 223., 102.], dtype = tf.complex64),
                "gamma_Tn" : tf.constant([4., 10., 14.4, 11.4, 5.7], dtype = tf.complex64),
                "omega_Ln" : tf.constant([715, 1549., 381., 239., 123.], dtype = tf.complex64),
                "gamma_Ln" : tf.constant([4., 10., 14.4, 11.4, 5.7], dtype = tf.complex64)
            },
            "extraordinary": {
                "high_freq": tf.constant(2.4,dtype = tf.complex64),
                "omega_Tn" : tf.constant([872., 303., 92.], dtype = tf.complex64),
                "gamma_Tn" : tf.constant([1.3, 9.1, 5.6], dtype = tf.complex64),
                "omega_Ln" : tf.constant([890., 387., 136.], dtype = tf.complex64),
                "gamma_Ln" : tf.constant([1.3, 9.1, 5.6], dtype = tf.complex64)
            }
        }

        return parameters



class CalciteLower(Calcite):
    def __init__(self, frequency_length, run_on_device_decorator):
        super().__init__(frequency_length, run_on_device_decorator)
        self.name = "Calcite-Lower"
        self.frequency = tf.cast(tf.linspace(820., 970., self.frequency_length), dtype=tf.complex64)


class CalciteUpper(Calcite):
    def __init__(self, frequency_length, run_on_device_decorator):
        super().__init__(frequency_length, run_on_device_decorator)
        self.name = "Calcite-Upper"
        self.frequency = tf.cast(tf.linspace(1200., 1700., self.frequency_length), dtype=tf.complex64)


class Antiferromagnet(object):

    @run_on_device
    def __init__(self, frequency_length, run_on_device_decorator):
        self.frequency_length = frequency_length
        self.run_on_device = run_on_device_decorator
        self.frequency = tf.cast(tf.linspace(52.0, 54.0, self.frequency_length), dtype=tf.complex64)
        
        self.gamma = tf.cast(1.05, dtype=tf.complex64) # cm^-1 / T
        self.B0 = tf.cast(0.3, dtype=tf.complex64) # T
        self.Ba = tf.cast(19.745, dtype=tf.complex64) # T
        self.Be = tf.cast(53.313, dtype=tf.complex64) # T
        self.magnetisation = tf.cast(445633.84, dtype=tf.complex64) #A/cm

        self.resonant_frequency_squared = self.gamma**2. * (2. * self.Ba * self.Be + self.Ba ** 2.)
        self.damping_parameter = 1.27e-4 * tf.sqrt(self.resonant_frequency_squared)
    
    @run_on_device
    def fetch_epsilon_mu(self):

        X = 1./(self.resonant_frequency_squared - (self.frequency + self.B0 * self.gamma + tf.constant(1j, dtype=tf.complex64) * self.damping_parameter)**2.)
        Y = 1./(self.resonant_frequency_squared - (self.frequency - self.B0 * self.gamma + tf.constant(1j, dtype=tf.complex64) * self.damping_parameter)**2.)

        mu_3 = 1. + constants.mu_0 * self.gamma**2. * self.Ba * self.magnetisation * (X + Y)
        mu_t = constants.mu_0 * self.gamma**2. * self.Ba * self.magnetisation * (X-Y)
        
        magnet_permittivity = tf.constant(5.5 + 0.j, dtype=tf.complex64)

        return mu_3, mu_t, magnet_permittivity
    
    @run_on_device
    def magnet_tensors(self):

        mu_3, mu_t, magnet_permittivity = self.fetch_epsilon_mu()

        permeability_tensor = tf.stack(
            [
            [mu_3, tf.zeros_like(mu_3), -tf.constant(1.j, dtype=tf.complex64) * mu_t],
            [tf.zeros_like(mu_3), tf.ones_like(mu_3), tf.zeros_like(mu_3)],
            [tf.constant(1.j, dtype=tf.complex64) * mu_t, tf.zeros_like(mu_3), mu_3]
            ], axis=-1)
        permeability_tensor = tf.transpose(permeability_tensor, perm=[1, 0, 2])

        permittivity_tensor = tf.stack([
            [magnet_permittivity, 0., 0.],
            [0., magnet_permittivity, 0.],
            [0., 0., magnet_permittivity]
            ], axis=-1) * tf.ones_like(permeability_tensor)
        
        return permeability_tensor, permittivity_tensor


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
