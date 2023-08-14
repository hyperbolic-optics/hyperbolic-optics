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
    structure = Structure()
    structure.execute(payload)
    print(structure.layers)
    
    crystal_layer = structure.layers[-1]


    thickness = 1.e-2

    forward_waves = crystal_layer.profile['transmitted']['propagation']
    backward_waves = crystal_layer.profile['reflected']['propagation']

    eigenvalues = tf.linalg.diag(tf.stack([
        forward_waves[..., 0],
        backward_waves[..., 0],
        forward_waves[..., 1],
        backward_waves[..., 1]],
        axis=-1))
    k_0 = structure.k_0[:, tf.newaxis, tf.newaxis, tf.newaxis]

    propagation_matrix = tf.linalg.expm(-1.j * eigenvalues * k_0 * thickness)

    print(propagation_matrix.shape)
    
    
    
    
    


if __name__ == '__main__':
    main()
