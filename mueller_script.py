import tensorflow as tf
import numpy as np
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure
from payloads import mock_dispersion_payload
import json
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Arial'

def setup_structure():
    payload = json.loads(mock_dispersion_payload())
    structure = Structure()
    structure.execute(payload)
    return structure

def plot_dispersion(fig, ax, structure, param, param_name):
    incident_angle = structure.incident_angle.numpy().real
    z_rotation = structure.azimuthal_angle.numpy().real
    
    if np.min(param) >= 0:
        cmap = 'magma'
        norm = plt.Normalize(vmin=0, vmax=np.max(param))
    else:
        cmap = 'seismic'
        abs_max = max(abs(np.min(param)), abs(np.max(param)))
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    
    im = ax.pcolormesh(z_rotation, incident_angle, param, cmap=cmap, norm=norm, shading='auto')
    ax.set_xticks(np.linspace(0, 2*np.pi, 5))
    ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=12)
    ax.set_yticks([])
    ax.grid(False)
    
    cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(param_name, fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    return im

def create_grid_plot(structure, params):
    fig = plt.figure(figsize=(22, 11))
    
    param_names = ['S0', 'S1', 'S2', 'S3', 'DOP', 'Ellipticity', 'Azimuth']
    
    ncols, nrows = 4, 2
    width, height = 0.2, 0.4
    hspace, vspace = 0.1, 0.1
    left, bottom = 0.05, 0.05

    for i, param in enumerate(param_names):
        row = i // ncols
        col = i % ncols
        
        pos_left = left + col * (width + hspace)
        pos_bottom = bottom + (nrows - 1 - row) * (height + vspace)
        
        ax = fig.add_axes([pos_left, pos_bottom, width, height], projection='polar')
        plot_dispersion(fig, ax, structure, params[param], param)

    return fig

def test_mueller(structure, polarization_type, **polarization_args):
    mueller = Mueller(structure, debug=False)
    
    if polarization_type != 'unpolarized':
        mueller.set_incident_polarization(polarization_type, **polarization_args)
    
    mueller.add_optical_component('anisotropic_sample')
    return mueller.get_all_parameters()

def main():
    structure = setup_structure()
    
    # Create test_figures folder if it doesn't exist
    if not os.path.exists('test_figures'):
        os.makedirs('test_figures')
    
    # Test with unpolarized light
    unpolarized_params = test_mueller(structure, 'unpolarized')
    unpolarized_fig = create_grid_plot(structure, unpolarized_params)
    unpolarized_fig.savefig("test_figures/unpolarized_dispersion_grid.png", dpi=300, bbox_inches='tight')
    plt.close(unpolarized_fig)

    # Test with polarized light states
    polarized_states = [
        ('linear', {'angle': 45}),
        ('linear', {'angle': 45}),
        ('circular', {'handedness': 'right'}),
        ('elliptical', {'alpha': 30, 'ellipticity': 15})
    ]

    for pol_type, pol_args in polarized_states:
        params = test_mueller(structure, pol_type, **pol_args)
        fig = create_grid_plot(structure, params)
        fig.savefig(f"test_figures/{pol_type}_dispersion_grid.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("All plots have been generated and saved in the 'test_figures' folder.")

if __name__ == '__main__':
    main()