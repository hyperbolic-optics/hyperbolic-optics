"""
script.py

Takes JSON payload containing data on multilayer structure,
creates the structure object then produces the reflectivity spectra.
"""

import json
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from hyperbolic_optics.plots import contour_plot_mueller_incidence, contour_plot_mueller_azimuthal, contour_plot_mueller_dispersion

def main():
    """
    Main function
    """
    mode = 'azimuthal'
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
    

    mueller = Mueller()
    mueller.add_optical_component('linear_polarizer', 0)  # First polarizer
    mueller.add_optical_component('anisotropic_sample', structure.r_pp, structure.r_ps, structure.r_sp, structure.r_ss)
    # # mueller.add_optical_component('linear_polarizer', 45)  # Second polarizer
    reflectivity = mueller.calculate_reflectivity()
    # contour_plot_mueller_incidence(structure,reflectivity)
    contour_plot_mueller_azimuthal(structure,reflectivity)
    # # contour_plot_mueller_dispersion(structure,reflectivity)

if __name__ == '__main__':
    main()
