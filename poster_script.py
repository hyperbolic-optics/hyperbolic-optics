import json
import numpy as np
import os

from layers import Structure
import matplotlib.pyplot as plt
from material_params import Quartz

def our_payload():
    payload = json.dumps({
    "ScenarioData": {
        "type": "Dispersion",
        "azimuthalAngle": {"min": 0, "max": 360},
        "incidentAngle": {"min": 0, "max": 90},
        "frequency": 496
    },
    "Layers": [
        {
            "type": "Ambient Incident Layer",
            "permittivity": 5.5
        },
        {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 2.
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Quartz",
            "rotationX": 90,
            "rotationY": 90,
            "rotationZ": 0,
        }]
    })
    
    return payload


def plot(reflectivity):

    plt.rcParams.update({
    'font.family': 'Arial'
    })

    X = np.linspace(-90, 90, reflectivity.shape[1])
    Y = np.linspace(410,600, reflectivity.shape[0])
    fig, ax = plt.subplots(figsize=(9, 6))
    c = ax.pcolormesh(X, Y, reflectivity, cmap='magma', vmin=0, vmax=1)
    ax.set_xlabel('Incident Angle / $^\circ$', fontsize=24)
    ax.set_ylabel('$\omega/2\pi c (cm^{-1})$', fontsize=26)
    
    ax.tick_params(axis='both', which='major', labelsize=22)
    # ax.axes.get_yaxis().set_ticks([])

    # set x-axis limit and ticks
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.linspace(-90, 90, 5))

    # Colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Reflectivity', fontsize=20)
    cbar.ax.tick_params(labelsize=15)

    # Save plot
    fig.savefig(f"Rp_45_theory_incident.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)



def plot_polar(data):

    radius = np.linspace(0, 90, data.shape[0])
    rotation = np.linspace(0, 2*np.pi, data.shape[1])

    plt.rcParams.update({
        'font.family': 'Arial'
    })

    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(projection='polar'))
    c = ax.pcolormesh(rotation, radius, data, cmap='magma')
    
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Remove the labels, polar coordinates speak for themselves
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Adjust labels and ticks
    ax.set_xticks(np.linspace(0, 2*np.pi, 5))
    ax.set_xticklabels(['0°', '90°', '180°   ', '270°',''])  # azimuthal rotation in degrees
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Colorbar
    cbar = fig.colorbar(c, ax=ax, pad=0.1) # Increase pad value to increase distance
    cbar.set_label("Reflectivity", fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    
    # Save plot
    fig.savefig(f"dispersion.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)





def main():
    payload = json.loads(our_payload())
    structure = Structure()
    structure.execute(payload)
    reflectivity = (structure.r_pp * np.conj(structure.r_pp) + structure.r_ps * np.conj(structure.r_ps)).numpy().real
    plot_polar(reflectivity)

if __name__ == "__main__":
    main()
