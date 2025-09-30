#!/usr/bin/env python3
"""
k-Space Dispersion Plot Example

This script demonstrates how to generate k-space dispersion plots (kx vs ky)
showing the reflectivity in momentum space for a Calcite crystal.
"""


from hyperbolic_optics.plots import plot_mueller_dispersion
from hyperbolic_optics.structure import Structure


def main():
    """
    Generate a k-space dispersion plot for Calcite.
    """
    # Define dispersion scenario
    payload = {
        "ScenarioData": {
            "type": "Dispersion",
            "frequency": 1460.0,  # Fixed frequency in cm^-1
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 25.0},
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.5,
                "permittivity": 1.0,
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 70,  # Optical axis tilt
                "rotationZ": 0,
            },
        ],
    }

    # Create and execute the simulation
    structure = Structure()
    structure.execute(payload)

    # Calculate total reflectivity
    R_total = abs(structure.r_pp) ** 2 + abs(structure.r_ps) ** 2

    # Plot removed for benchmarking
    print(f"Dispersion calculation complete. R_total shape: {R_total.shape}")

    plot_mueller_dispersion(structure, R_total)


if __name__ == "__main__":
    main()
