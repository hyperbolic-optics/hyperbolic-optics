#!/usr/bin/env python3
"""Azimuthal layer-resolved absorption in a MoO3/AlN/SiC heterostructure.

Reproduces Fig. 2 of Passler, Jeannin & Paarmann, "Layer-Resolved Absorption of
Light in Arbitrarily Anisotropic Heterostructures," J. Opt. Soc. Am. B 37, 1060
(2020) [arXiv:2002.03832]:

    KRS5 prism / 8 um air gap / 1.2 um alpha-MoO3 / 1.0 um AlN / SiC substrate,
    Otto geometry, theta = 28 deg, p-polarized, swept over azimuth Phi.

Only MoO3 is biaxial (in-plane anisotropic), so the azimuth dependence of the
absorption comes from the air/MoO3 hyperbolic phonon polariton; AlN and SiC are
c-cut uniaxial and contribute Phi-independent features.

Energy conservation gives the layer decomposition directly:

    R + A_MoO3 + A_AlN + A_SiC = 1,

where A_SiC is the power transmitted into (and absorbed by) the semi-infinite SiC
substrate, i.e. ``FieldProfile.transmittance``.

Note: the MoO3/AlN/SiC phonon parameters in material_params.json are literature
values flagged for verification -- sanity-check them against the cited sources
before using the figure quantitatively.

Run:  python examples/layer_resolved_absorption.py
"""

import numpy as np

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure

# Otto-geometry heterostructure (thicknesses in microns; KRS5 prism n ~ 2.4).
PAYLOAD = {
    "ScenarioData": {"type": "Azimuthal", "incidentAngle": 28.0},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 5.5},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 8.0, "permittivity": 1.0},
        {
            "type": "Crystal Layer",
            "material": "MoO3",
            "thickness": 1.2,
            "rotationX": 90,
            "rotationZ": 0,
        },
        {
            "type": "Crystal Layer",
            "material": "AlN",
            "thickness": 1.0,
            "rotationY": 0,
            "rotationZ": 0,
        },
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "SiC",
            "rotationY": 0,
            "rotationZ": 0,
        },
    ],
}

MOO3_INDEX, ALN_INDEX = 2, 3  # layer positions in PAYLOAD["Layers"]


def compute():
    """Run the sweep and return (frequency, azimuth_deg, R, A_MoO3, A_AlN, A_SiC)."""
    structure = Structure()
    structure.execute(PAYLOAD)
    fp = FieldProfile(structure)

    reflectance = fp.reflectance("p")  # (F, Phi)
    a_sic = fp.transmittance("p")  # flux into the lossy SiC substrate = its absorption
    absorption = {entry["index"]: entry["absorptance"] for entry in fp.layer_absorption("p")}
    a_moo3 = absorption[MOO3_INDEX]
    a_aln = absorption[ALN_INDEX]

    frequency = np.squeeze(np.asarray(structure.frequency))
    azimuth_deg = np.degrees(np.squeeze(np.asarray(structure.azimuthal_angle)))

    residual = np.max(np.abs(reflectance + a_moo3 + a_aln + a_sic - 1.0))
    print(f"max |R + A_MoO3 + A_AlN + A_SiC - 1| = {residual:.2e}")
    print(f"output shape (freq x azimuth) = {reflectance.shape}")
    return frequency, azimuth_deg, reflectance, a_moo3, a_aln, a_sic


def plot(frequency, azimuth_deg, reflectance, a_moo3, a_aln, a_sic):
    """Plot reflectance and the three layer absorptions vs (Phi, omega)."""
    import matplotlib.pyplot as plt

    half = azimuth_deg <= 180.0  # the figure shows Phi in [0, 180]
    phi = azimuth_deg[half]
    extent = [phi.min(), phi.max(), frequency.min(), frequency.max()]

    panels = [
        ("Reflectance $R_p$", reflectance[:, half], "Reds"),
        (r"Absorption $A_p$ MoO$_3$", a_moo3[:, half], "Oranges"),
        (r"Absorption $A_p$ AlN", a_aln[:, half], "Oranges"),
        (r"Absorption $A_p$ SiC", a_sic[:, half], "Oranges"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(
            data, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.set_xlabel(r"azimuthal angle $\Phi$ (deg)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"frequency $\omega$ (cm$^{-1}$)")
    fig.suptitle("Azimuthal layer-resolved absorption: MoO$_3$/AlN/SiC (Otto, $\\theta=28^\\circ$)")
    fig.tight_layout()
    plt.show()


def main():
    results = compute()
    try:
        plot(*results)
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")


if __name__ == "__main__":
    main()
