import unittest
import tensorflow as tf
from waves import Wave, WaveProfile

class TestWave(unittest.TestCase):

    def setUp(self):

        self.kx = tf.constant([1.0], dtype=tf.complex128)
        self.eps_tensor = tf.eye(3, dtype=tf.complex128)
        self.mu_tensor = tf.eye(3, dtype=tf.complex128)
        self.mode = "Incident"

        self.wave = Wave(self.kx, self.eps_tensor, self.mu_tensor, self.mode)

    def test_mode_reshaping(self):

        k_x, eps_tensor, mu_tensor = self.wave.mode_reshaping()
        self.assertEqual(k_x.shape, (1, 1))
        self.assertEqual(eps_tensor.shape, (1, 3, 3))
        self.assertEqual(mu_tensor.shape, (3, 3))

    def test_delta_matrix_calc(self):

        self.wave.delta_matrix_calc()

        # Ensure berreman_matrix exists before reshaping
        assert hasattr(self.wave, "berreman_matrix")

        self.wave.berreman_matrix = tf.reshape(self.wave.berreman_matrix, [4, 4])

    def test_wave_sorting(self):

        self.wave.delta_matrix_calc()

        # Ensure berreman_matrix exists
        assert hasattr(self.wave, "berreman_matrix")

        # Print the original shape for debugging
        print("Original Berreman matrix shape:", self.wave.berreman_matrix.shape)

        # Reshape to ensure it's square
        self.wave.berreman_matrix = tf.reshape(self.wave.berreman_matrix, [4, 4])

        print("Reshaped Berreman matrix shape:", self.wave.berreman_matrix.shape)

        transmitted_waves, reflected_waves, transmitted_fields, reflected_fields = self.wave.wave_sorting()

        self.assertEqual(transmitted_waves.shape[-1], 2)
        self.assertEqual(reflected_waves.shape[-1], 2)

if __name__ == "__main__":
    unittest.main()
