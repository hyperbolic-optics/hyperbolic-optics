# class Antiferromagnet:
#     """
#     Class representing an antiferromagnetic material.

#     Attributes:
#         frequency_length (int): The length of the frequency range.
#         frequency (tf.Tensor): The frequency range for the antiferromagnetic material.
#         name (str): The name of the material.
#         gamma (tf.Tensor): The gyromagnetic ratio.
#         B0 (tf.Tensor): The external magnetic field.
#         Ba (tf.Tensor): The anisotropy field.
#         Be (tf.Tensor): The exchange field.
#         magnetisation (tf.Tensor): The magnetization of the material.
#         resonant_frequency_squared (tf.Tensor): The square of the resonant frequency.
#         damping_parameter (tf.Tensor): The damping parameter.
#     """

#     def __init__(self, frequency_length):
#         """
#         Initialize the Antiferromagnet class.

#         Args:
#             frequency_length (int): The length of the frequency range.
#         """
#         self.frequency_length = frequency_length
#         self.frequency = tf.cast(
#             tf.linspace(52.0, 54.0, self.frequency_length), dtype=tf.complex128
#         )
#         self.name = "FeF2"

#         self.gamma = tf.cast(1.05, dtype=tf.complex128)  # cm^-1 / T
#         self.B0 = tf.cast(0.3, dtype=tf.complex128)  # T
#         self.Ba = tf.cast(19.745, dtype=tf.complex128)  # T
#         self.Be = tf.cast(53.313, dtype=tf.complex128)  # T
#         self.magnetisation = tf.cast(445633.84, dtype=tf.complex128)  # A/cm

#         self.resonant_frequency_squared = self.gamma**2.0 * (
#             2.0 * self.Ba * self.Be + self.Ba**2.0
#         )
#         self.damping_parameter = 1.27e-4 * tf.sqrt(self.resonant_frequency_squared)

#     def fetch_epsilon_mu(self):
#         """
#         Fetch the epsilon and mu values for the antiferromagnetic material.

#         Returns:
#             tuple: A tuple containing mu_3, mu_t, and magnet_permittivity.
#         """
#         X = 1.0 / (
#             self.resonant_frequency_squared
#             - (
#                 self.frequency
#                 + self.B0 * self.gamma
#                 + tf.constant(1j, dtype=tf.complex128) * self.damping_parameter
#             )
#             ** 2.0
#         )
#         Y = 1.0 / (
#             self.resonant_frequency_squared
#             - (
#                 self.frequency
#                 - self.B0 * self.gamma
#                 + tf.constant(1j, dtype=tf.complex128) * self.damping_parameter
#             )
#             ** 2.0
#         )

#         mu_3 = 1.0 + constants.mu_0 * self.gamma**2.0 * self.Ba * self.magnetisation * (
#             X + Y
#         )
#         mu_t = (
#             constants.mu_0 * self.gamma**2.0 * self.Ba * self.magnetisation * (X - Y)
#         )

#         magnet_permittivity = tf.constant(5.5 + 0.0j, dtype=tf.complex128)

#         return mu_3, mu_t, magnet_permittivity

#     def magnet_tensors(self):
#         """
#         Get the permeability and permittivity tensors for the antiferromagnetic material.

#         Returns:
#             tuple: A tuple containing the permeability_tensor and permittivity_tensor.
#         """
#         mu_3, mu_t, magnet_permittivity = self.fetch_epsilon_mu()

#         permeability_tensor = tf.stack(
#             [
#                 [
#                     mu_3,
#                     tf.zeros_like(mu_3),
#                     -tf.constant(1.0j, dtype=tf.complex128) * mu_t,
#                 ],
#                 [tf.zeros_like(mu_3), tf.ones_like(mu_3), tf.zeros_like(mu_3)],
#                 [
#                     tf.constant(1.0j, dtype=tf.complex128) * mu_t,
#                     tf.zeros_like(mu_3),
#                     mu_3,
#                 ],
#             ],
#             axis=-1,
#         )
#         permeability_tensor = tf.transpose(permeability_tensor, perm=[1, 0, 2])

#         permittivity_tensor = tf.stack(
#             [
#                 [magnet_permittivity, 0.0, 0.0],
#                 [0.0, magnet_permittivity, 0.0],
#                 [0.0, 0.0, magnet_permittivity],
#             ],
#             axis=-1,
#         ) * tf.ones_like(permeability_tensor)

#         return permeability_tensor, permittivity_tensor