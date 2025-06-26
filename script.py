"""
script.py

Takes JSON payload containing data on multilayer structure,
creates the structure object then produces the reflectivity spectra.
"""

import json
import tensorflow as tf
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from payloads import (mock_incident_payload, mock_azimuthal_payload, 
                     mock_dispersion_payload, mock_simple_payload, 
                     mock_simple_dielectric_payload)
tf.get_logger().setLevel("ERROR")
from hyperbolic_optics.plots import plot_mueller_dispersion, plot_mueller_azimuthal, plot_kx_frequency

import sys
print(sys.executable)

def main():
    """
    Main function
    """
    mode = 'simple'  # Changed to test simple mode
    
    if mode == 'incident':
        payload = json.loads(mock_incident_payload())
    elif mode == 'azimuthal':
        payload = json.loads(mock_azimuthal_payload())
    elif mode == 'dispersion':
        payload = json.loads(mock_dispersion_payload())
    elif mode == 'simple':
        payload = json.loads(mock_simple_payload())
    elif mode == 'simple_dielectric':
        payload = json.loads(mock_simple_dielectric_payload())
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    
    structure = Structure()
    structure.execute(payload)
    
    # For simple scenario, the results will be scalar values
    print(f"Scenario type: {structure.scenario.type}")
    print(f"Incident angle: {structure.incident_angle} radians")
    print(f"Azimuthal angle: {structure.azimuthal_angle} radians") 
    print(f"Frequency: {structure.frequency} cm^-1")
    print(f"k_x: {structure.k_x}")
    print(f"k_0: {structure.k_0}")
    
    # # Print reflection coefficients (these should be scalars for Simple mode)
    # print(f"r_pp: {structure.r_pp}")
    # print(f"r_ss: {structure.r_ss}")
    # print(f"r_ps: {structure.r_ps}")
    # print(f"r_sp: {structure.r_sp}")
    
    # Calculate reflectivity
    R_pp = tf.abs(structure.r_pp)**2
    R_ss = tf.abs(structure.r_ss)**2
    R_ps = tf.abs(structure.r_ps)**2
    R_sp = tf.abs(structure.r_sp)**2
    
    print(f"R_pp: {R_pp}")
    print(f"R_ss: {R_ss}")
    print(f"R_ps: {R_ps}")
    print(f"R_sp: {R_sp}")
    
    # Mueller matrix analysis for simple scenario
    mueller = Mueller(structure)
    mueller.set_incident_polarization('linear', **{"angle": 0})
    mueller.add_optical_component('anisotropic_sample')
    
    # For Simple scenario, Stokes parameters will be scalars
    # stokes_params = mueller.get_all_parameters()
    # print(f"Stokes parameters:")
    # for param, value in stokes_params.items():
    #     print(f"  {param}: {value}")

if __name__ == '__main__':
    main()