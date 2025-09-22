
"""
FeF2 Magnetic Material Example

This script demonstrates magnetic material analysis using FeF2 (iron difluoride),
an antiferromagnetic material. Shows incident angle vs frequency analysis with
Mueller matrix polarimetry using s-polarized incident light.

Based on:
N. R. Anderson and R. E. Camley, "Attenuated total reflection study of bulk and 
surface polaritons in antiferromagnets and hexagonal ferrites: Propagation at 
arbitrary angles," J. Appl. Phys., vol. 113, 013904, Jan. 2013.
DOI: 10.1063/1.4770467
"""

from hyperbolic_optics.structure import Structure
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.plots import plot_kx_frequency

def main():
    """
    Analyze FeF2 magnetic material with incident angle sweep and Mueller matrix method.
    """
    
    # Define incident scenario with FeF2 magnetic material
    payload = {
        "ScenarioData": {
            "type": "Incident"  # Frequency vs incident angle analysis
        },
        "Layers": [
            {
                "type": "Ambient Incident Layer",
                "permittivity": 12.5  # High-index prism for coupling
            },
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 30.0,    # 30 μm air gap
                "permittivity": 1.0   # Air
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "FeF2",   # Iron difluoride antiferromagnet
                "rotationY": 0,
                "rotationZ": 0
            }
        ]
    }
    
    # Create and execute the simulation
    structure = Structure()
    structure.execute(payload)

    mueller = Mueller(structure, debug=False)
    mueller.set_incident_polarization('linear', angle=90)  # s-polarized (90°)
    mueller.add_optical_component('anisotropic_sample')
    
    # Get all Mueller matrix parameters
    all_params = mueller.get_all_parameters()
    
    # Extract S0 parameter (intensity/reflectivity)
    S0_intensity = all_params['S0']  # Stokes parameter S0 (total intensity)
    
    # Plot S0 intensity from Mueller matrix analysis
    plot_kx_frequency(
        structure,
        S0_intensity,
    )
    

if __name__ == "__main__":
    main()