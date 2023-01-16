"""
Not intended for functional code, just as reference
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_base_permittivity(wavenumber, E_xx, E_zz):
    plt.rcParams["figure.figsize"] = (5,4)
    fig, axs = plt.subplots(2)
    fig.suptitle('Permittivities')
    axs[0].plot(wavenumber, E_xx.real)
    axs[0].plot(wavenumber, E_xx.imag)
    axs[0].set(xlabel='Wavenumber', ylabel = 'E_xx')

    axs[1].plot(wavenumber, E_zz.real)
    axs[1].plot(wavenumber, E_zz.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'E_zz')

    plt.show()

def plot_rotated_permittivities(wavenumber, E_xx, E_zz, E_xz):
    plt.rcParams["figure.figsize"] = (7,5)
    fig, axs = plt.subplots(3)
    fig.suptitle('Permittivities at 90 Degrees')
    axs[0].plot(wavenumber, E_xx.real)
    axs[0].plot(wavenumber, E_xx.imag)
    axs[0].set(xlabel='Wavenumber', ylabel = 'E_xx')

    axs[1].plot(wavenumber, E_zz.real)
    axs[1].plot(wavenumber, E_zz.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'E_zz')

    axs[2].plot(wavenumber, E_xz.real)
    axs[2].plot(wavenumber, E_xz.imag)
    axs[2].set(xlabel='Wavenumber', ylabel = 'E_xz')

    plt.show()

def plot_kvectors(wavenumber, kz3, kz3_other):
    plt.rcParams["figure.figsize"] = (4,4)
    fig, axs = plt.subplots(2)
    fig.suptitle('k_vectors')
    axs[0].plot(wavenumber, kz3.real)
    axs[0].plot(wavenumber, kz3_other.real)
    axs[0].set(xlabel='Wavenumber', ylabel = 'Real')

    axs[1].plot(wavenumber, kz3.imag)
    axs[1].plot(wavenumber, kz3_other.imag)
    axs[1].set(xlabel='Wavenumber', ylabel = 'Imag')

    plt.show()


