#!/usr/bin/env python3
"""
Basic Calcite Reflection Example

This script demonstrates the most basic usage of the hyperbolic-optics package
by calculating reflection coefficients for a simple Calcite crystal structure.
"""


import numpy as np

from hyperbolic_optics.structure import Structure


def main():
    """
    Calculate basic reflection coefficients for Calcite at a single frequency and angle.
    """
    print("=== Basic Calcite Reflection Example ===\n")

    # Define a simple structure with Calcite
    payload = {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,  # 45 degree incident angle
            "azimuthal_angle": 0.0,  # No azimuthal rotation
            "frequency": 1460.0,  # Frequency in cm^-1
        },
        "Layers": [
            {
                "type": "Ambient Incident Layer",
                "permittivity": 50.0,  # High-index prism
            },
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.1,  # Thin air gap (in mm)
                "permittivity": 1.0,  # Air
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,  # Rotate optical axis
                "rotationZ": 0,
            },
        ],
    }

    # Create and execute the simulation
    print("Creating structure and calculating...")
    structure = Structure()
    structure.execute(payload)

    # Extract reflection coefficients
    r_pp = structure.r_pp  # p-to-p polarization
    r_ss = structure.r_ss  # s-to-s polarization
    r_ps = structure.r_ps  # p-to-s polarization
    r_sp = structure.r_sp  # s-to-p polarization

    # Calculate reflectivities (|r|²)
    R_pp = abs(r_pp) ** 2
    R_ss = abs(r_ss) ** 2
    R_ps = abs(r_ps) ** 2
    R_sp = abs(r_sp) ** 2

    # Display results
    print("Results:")
    print(f"  Incident angle: {structure.scenario.incident_angle * 180/np.pi:.1f}°")
    print(f"  Frequency: {structure.frequency:.1f} cm⁻¹")
    print("  Material: Calcite")
    print()
    print("Reflection Coefficients:")
    print(f"  r_pp = {r_pp:.6f}")
    print(f"  r_ss = {r_ss:.6f}")
    print(f"  r_ps = {r_ps:.6f}")
    print(f"  r_sp = {r_sp:.6f}")
    print()
    print("Reflectivities (|r|²):")
    print(f"  R_pp = {R_pp:.4f}")
    print(f"  R_ss = {R_ss:.4f}")
    print(f"  R_ps = {R_ps:.4f}")
    print(f"  R_sp = {R_sp:.4f}")
    print()
    print(f"Total reflectivity for p-polarized light: {R_pp + R_ps:.4f}")
    print(f"Total reflectivity for s-polarized light: {R_ss + R_sp:.4f}")


if __name__ == "__main__":
    main()
