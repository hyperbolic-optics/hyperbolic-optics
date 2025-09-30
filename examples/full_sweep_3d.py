#!/usr/bin/env python3
"""
Full 3D Parameter Sweep Example

This script demonstrates the full 3D parameter sweep capability:
frequency × incident_angle × azimuthal_angle

The output will have shape [N_freq, N_incident, N_azimuthal] for each
reflection coefficient (r_pp, r_ss, r_ps, r_sp).
"""

from hyperbolic_optics.structure import Structure


def main():
    """
    Generate a full 3D parameter sweep for Calcite.
    """
    # Define full sweep scenario
    payload = {
        "ScenarioData": {
            "type": "FullSweep",
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 90,
            },
        ],
    }

    # Create and execute the simulation
    print("Starting full 3D parameter sweep...")
    print("This will sweep: frequency × incident_angle × azimuthal_angle")
    print()

    structure = Structure()
    structure.execute(payload)

    # Calculate total reflectivity
    R_total = abs(structure.r_pp) ** 2 + abs(structure.r_ps) ** 2

    # Display results
    print("Full sweep complete!")
    print(f"  R_total shape: {R_total.shape}")
    print("  Expected: [N_freq, N_incident, N_azimuthal]")
    print()
    print(f"  r_pp shape: {structure.r_pp.shape}")
    print(f"  r_ss shape: {structure.r_ss.shape}")
    print(f"  r_ps shape: {structure.r_ps.shape}")
    print(f"  r_sp shape: {structure.r_sp.shape}")
    print()

    # Some statistics
    print(f"  R_total min: {R_total.min():.4f}")
    print(f"  R_total max: {R_total.max():.4f}")
    print(f"  R_total mean: {R_total.mean():.4f}")


if __name__ == "__main__":
    main()
