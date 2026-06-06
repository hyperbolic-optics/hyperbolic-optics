#!/usr/bin/env python3
"""Sweep a calcite film's thickness as an extra output dimension.

Uses :class:`~hyperbolic_optics.sweep.ThicknessSweep` to re-run a structure across
a range of film thicknesses and stack the results along a leading thickness axis.
Here, at a single frequency and incident angle (Otto coupling), we watch how the
reflectance, transmittance into the vacuum substrate, and the absorption in the
calcite film evolve as the film thickens:

    prism (eps=25) / 0.5 um air gap / calcite film (swept) / vacuum

Run:  python examples/thickness_sweep.py
"""

import numpy as np

from hyperbolic_optics.sweep import ThicknessSweep

PAYLOAD = {
    "ScenarioData": {
        "type": "Simple",
        "incidentAngle": 30.0,
        "azimuthal_angle": 0.0,
        "frequency": 1460.0,
    },
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 25.0},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
        {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.0, "rotationY": 90},
        {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
    ],
}
FILM_INDEX = 2


def main():
    thicknesses = np.linspace(0.1, 4.0, 80)  # microns
    sweep = ThicknessSweep(PAYLOAD, layer_index=FILM_INDEX, thicknesses=thicknesses)

    reflectance = sweep.reflectance("p")
    transmittance = sweep.transmittance("p")
    film = next(e for e in sweep.layer_absorption("p") if e["index"] == FILM_INDEX)
    absorption = film["absorptance"]

    print(f"swept {len(sweep)} thicknesses; reflectance shape {reflectance.shape}")
    print(
        f"max |R + T + A - 1| = {np.max(np.abs(reflectance + transmittance + absorption - 1)):.2e}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thicknesses, reflectance, label="R (reflectance)", color="crimson", lw=2)
    ax.plot(thicknesses, transmittance, label="T (into vacuum)", color="navy", lw=2)
    ax.plot(thicknesses, absorption, label="A (calcite film)", color="darkorange", lw=2)
    ax.set_xlabel("calcite film thickness (µm)")
    ax.set_ylabel("fraction of incident power")
    ax.set_title(r"Power balance vs film thickness ($\theta=30^\circ$, 1460 cm$^{-1}$)")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out = "thickness_sweep_power.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
