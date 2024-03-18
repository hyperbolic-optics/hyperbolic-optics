import tensorflow as tf

from material_params import GalliumOxide
from matplotlib import pyplot as plt


if __name__ == "__main__":
    print("Material: Gallium Oxide")
    gallium = GalliumOxide()
    eps_xx = gallium.permittivity_calc()

    plt.plot(tf.math.real(gallium.frequency), tf.math.real(eps_xx))
    plt.show()