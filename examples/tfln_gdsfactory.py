#!/usr/bin/env python3
"""Thin-film LiNbO₃ in Luxtelligence's LNOI400 PDK: anisotropy at 1550 nm.

Stacks are read from the PDK's LayerStack (https://github.com/Luxtelligence/lxt_pdk_gf)
in a slab region, under its rib waveguide and under a partially etched line
grating. The PDK does not record the crystal cut; this assumes x-cut
(``Oriented("LiNbO3", x_cut(angle))``). The panels show effects only a crystal
produces: p→s conversion of the slab, the Mueller element M₂₃, and a grating's
−1 order for the optic axis across or along the ridges.

Run:  uv run --with lnoi400 python examples/tfln_gdsfactory.py
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import gdsfactory as gf  # noqa: E402
import lnoi400  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

OUT = Path(__file__).parent / "output"
WAVELENGTH = 1.55
CUT = hog.x_cut  # assumed: an x-cut film; the PDK does not record the cut


@gf.cell
def grating_on_slab(period_um: float = 1.2, n_periods: int = 12) -> gf.Component:
    """LN ridges of half-period width on LN_RIDGE, over an LN_SLAB region."""
    component = gf.Component()
    component.add_ref(
        hog.line_grating(period_um=period_um, fill=0.5, n_periods=n_periods, layer="LN_RIDGE")
    )
    size = period_um * n_periods
    component.add_polygon(
        [(-5.0, -5.0), (size + 5.0, -5.0), (size + 5.0, 25.0), (-5.0, 25.0)], layer="LN_SLAB"
    )
    return component


def _style(ax):
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b6a64")
    ax.tick_params(colors="#6b6a64", labelcolor="#1f1f1e", labelsize=8)


def main():
    OUT.mkdir(exist_ok=True)
    lnoi400.PDK.activate()
    layer_stack = lnoi400.PDK.layer_stack
    ink, blue, orange = "#1f1f1e", "#2a78d6", "#eb6834"

    # The PDK's own rib waveguide, read at its centre and beside the ridge.
    rib = lnoi400.PDK.get_component("straight_rwg1000")
    x_mid = rib.dbbox().center().x
    for label, y in (("centre", 0.0), ("beside the ridge", 3.0)):
        column = hog.vertical_stack(rib, layer_stack, (x_mid, y), {"ln": hog.Oriented("LiNbO3")})
        layers = ", ".join(f"{layer.name} {layer.thickness_um:g} µm" for layer in column.layers)
        print(f"straight_rwg1000, {label}: {layers} | {column.exit}")

    # --- (a, b): the slab region, beside the grating --------------------------
    cell = grating_on_slab()
    in_slab = (-2.0, 10.0)
    azimuths = np.linspace(0.0, 180.0, 61)
    angles = np.linspace(0.0, 75.0, 61)
    materials = {"ln": hog.Oriented("LiNbO3", CUT(0.0))}
    slab, planar = hog.simulate_stack(
        cell, layer_stack, in_slab, [WAVELENGTH], angles, azimuths, materials=materials
    )
    conversion = planar.R[0, :, :, 1, 0]  # [angle, azimuth], p in -> s out
    m23 = planar.mueller()[0, np.argmin(np.abs(angles - 45.0)), :, 1, 2]

    # --- (c): the grating, etched into the PDK stack, for two orientations ----
    # at 30°, the −1 order propagates in air once |sin 30° − λ/Λ| < 1, i.e. Λ > 1.03 µm
    periods = np.linspace(1.05, 2.0, 120)  # the guided-mode resonances are sharp
    first_order = {}
    for label, axis in (("axis across the ridges", 0.0), ("axis along the ridges", 90.0)):
        oriented = {"ln": hog.Oriented("LiNbO3", CUT(axis))}
        curves = {"p": [], "s": []}
        for period in periods:
            _, stack, response = hog.simulate_grating(
                grating_on_slab(period_um=float(period)), "LN_RIDGE", layer_stack,
                [WAVELENGTH], [30.0], max_order=12, materials=oriented,
            )  # fmt: skip
            minus_one = response.order(-1)
            for index, pol in enumerate("ps"):
                curves[pol].append(float(response.R[0, 0, minus_one, :, index].sum()))
        first_order[label] = curves
    print("grating stack:", [(layer.name, layer.thickness_um) for layer in stack.layers])

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    ax = axes[0]
    _style(ax)
    image = ax.pcolormesh(azimuths, angles, conversion, cmap="Blues", shading="auto")
    fig.colorbar(image, ax=ax, label="p→s reflectance")
    ax.set_title("(a) slab region: p→s conversion", fontsize=9, color=ink, loc="left")
    ax.set_xlabel("in-plane rotation of the optic axis (°)", fontsize=9, color=ink)
    ax.set_ylabel("angle of incidence (°)", fontsize=9, color=ink)

    ax = axes[1]
    _style(ax)
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.plot(azimuths, m23, color=blue, linewidth=2)
    ax.axhline(0.0, color="#6b6a64", linewidth=0.8)
    ax.set_title("(b) Mueller M₂₃ at 45° (zero if isotropic)", fontsize=9, color=ink, loc="left")
    ax.set_xlabel("in-plane rotation of the optic axis (°)", fontsize=9, color=ink)
    ax.set_ylabel("M₂₃ (reflection)", fontsize=9, color=ink)

    ax = axes[2]
    _style(ax)
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    for (label, curves), style in zip(first_order.items(), ("-", "--"), strict=True):
        for pol, color in (("p", blue), ("s", orange)):
            ax.plot(periods, curves[pol], color=color, linestyle=style, linewidth=2,
                    label=f"{pol} in, {label}")  # fmt: skip
    ax.set_title("(c) LN_RIDGE grating: −1 order reflected into air, 30°", fontsize=9,
                 color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("grating period (µm)", fontsize=9, color=ink)
    ax.set_ylabel("diffraction efficiency", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Luxtelligence LNOI400 PDK at 1550 nm: stacks read from the PDK LayerStack, "
        "x-cut LiNbO₃ assumed",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "tfln_gdsfactory.png", dpi=160)
    print("slab column:", [(layer.name, layer.thickness_um) for layer in slab.layers], slab.exit)
    peak = np.unravel_index(np.argmax(conversion), conversion.shape)
    print(f"max p→s conversion {conversion.max():.4f} at {angles[peak[0]]:.0f}°, "
          f"axis rotated {azimuths[peak[1]]:.0f}°")  # fmt: skip
    print(f"|M23| up to {np.abs(m23).max():.4f} at 45°")
    print(f"Figure: {OUT / 'tfln_gdsfactory.png'}")


if __name__ == "__main__":
    main()
