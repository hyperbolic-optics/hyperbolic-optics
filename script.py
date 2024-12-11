"""
script.py

Takes JSON payload containing data on multilayer structure,
creates the structure object then produces the reflectivity spectra.
"""

import json
import tensorflow as tf
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload
tf.get_logger().setLevel("ERROR")
from hyperbolic_optics.plots import contour_plot_mueller_incidence, contour_plot_mueller_azimuthal, contour_plot_mueller_dispersion

import sys
print(sys.executable)

def main():
    """
    Main function
    """
    mode = 'dispersion'
    if mode == 'incident':
        payload = json.loads(mock_incident_payload())
    elif mode == 'azimuthal':
        payload = json.loads(mock_azimuthal_payload())
    elif mode == 'dispersion':
        payload = json.loads(mock_dispersion_payload())
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    
    
    structure = Structure()
    
    structure.execute(payload)
    # structure.plot()
    

    mueller = Mueller(structure)
    mueller.set_incident_polarization('linear', **{"angle": 0})  # First polarizer
    mueller.add_optical_component('anisotropic_sample', structure.r_pp, structure.r_ps, structure.r_sp, structure.r_ss)
    # mueller.add_optical_component('linear_polarizer', 45)  # Second polarizer
    mueller.get_all_parameters()
    contour_plot_mueller_incidence(structure,mueller.get_stokes_parameters()['S0'])
    # # contour_plot_mueller_azimuthal(structure,reflectivity)
    # contour_plot_mueller_dispersion(structure,mueller.get_stokes_parameters()['S0'])

if __name__ == '__main__':
    main()
