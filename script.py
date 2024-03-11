"""
script.py

Takes JSON payload containing data on multilayer structure,
creates the structure object then produces the reflectivity spectra.
"""

import json
from structure import Structure
from payloads import mock_incident_payload, mock_azimuthal_payload, mock_dispersion_payload
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from plots import contour_plot_simple_incidence

def main():
    """
    Main function
    """
    mode = 'incident'
    if mode == 'incident':
        payload = json.loads(mock_incident_payload())
    elif mode == 'azimuthal':
        payload = json.loads(mock_azimuthal_payload())
    elif mode == 'dispersion':
        payload = json.loads(mock_dispersion_payload())
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    
    print(f"Payload: {payload}")  # Add this line to check the payload
    
    print("Creating Structure object")  # Add this line
    structure = Structure()
    
    print("Executing Structure object")  # Add this line
    structure.execute(payload)
    
    print("Finished executing Structure object")  # Add this line

if __name__ == '__main__':
    print("Starting main function")  # Add this line
    main()
    print("Finished main function")  # Add this line