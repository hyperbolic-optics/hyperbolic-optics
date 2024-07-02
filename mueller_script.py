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
    
    # Create the Mueller object
    mueller = Mueller(structure)
    # Set the incident polarization (optional)
    mueller.set_incident_polarization('linear', angle = 40)
    # Add optical components before the anisotropic sample (if any)
    mueller.add_optical_component('anisotropic_sample')
    # mueller.add_optical_component('linear_polarizer', 0)  # First polarizer
    # mueller.add_optical_component('half_wave_plate', 90)  # First quarter wave plate

    reflectivity = mueller.get_azimuth()

    contour_plot_mueller_dispersion(structure, reflectivity)


if __name__ == '__main__':
    main()
