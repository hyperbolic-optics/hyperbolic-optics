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

    # For a Simple scenario the coefficients are 0-d complex arrays; convert to
    # Python complex so they format cleanly.
    r_pp, r_ss = complex(structure.r_pp), complex(structure.r_ss)
    r_ps, r_sp = complex(structure.r_ps), complex(structure.r_sp)

    def fmt(z: complex) -> str:
        return f"{z.real:+.6f} {z.imag:+.6f}j"

    incident_deg = float(np.degrees(structure.incident_angle))
    frequency = float(np.squeeze(structure.frequency))

    print("Results:")
    print(f"  Incident angle: {incident_deg:.1f} deg")
    print(f"  Frequency: {frequency:.1f} cm^-1")
    print("  Material: Calcite")
    print()
    print("Reflection coefficients:")
    print(f"  r_pp = {fmt(r_pp)}")
    print(f"  r_ss = {fmt(r_ss)}")
    print(f"  r_ps = {fmt(r_ps)}")
    print(f"  r_sp = {fmt(r_sp)}")
    print()
    print("Reflectivities (|r|^2):")
    print(f"  R_pp = {abs(r_pp) ** 2:.4f}")
    print(f"  R_ss = {abs(r_ss) ** 2:.4f}")
    print(f"  R_ps = {abs(r_ps) ** 2:.4f}")
    print(f"  R_sp = {abs(r_sp) ** 2:.4f}")
    print()
    print(f"Total p-polarized reflectivity: {abs(r_pp) ** 2 + abs(r_ps) ** 2:.4f}")
    print(f"Total s-polarized reflectivity: {abs(r_ss) ** 2 + abs(r_sp) ** 2:.4f}")


if __name__ == "__main__":
    main()
