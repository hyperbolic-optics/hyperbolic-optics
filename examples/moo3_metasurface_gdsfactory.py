#!/usr/bin/env python3
"""A hyperbolic α-MoO₃ metasurface polarization converter: design → GDS → re-simulation.

α-MoO₃ is in-plane hyperbolic between about 820 and 970 cm⁻¹: ε_x < 0 < ε_y
along the two crystal axes in the flake plane. Elliptical MoO₃ pillars on a
gold mirror make a reflective polarization converter:

    air | MoO₃ ellipses (h) in air | KBr spacer (s) | gold | Si

At normal incidence, x-polarized (p) light is converted to y (s) only if the
structure's principal axes are not along x and y. Here two things set them,
the crystal axes and the ellipse's orientation, and both are optimised:

- ellipse radii and rotation;
- crystal rotation in the plane (a continuous orientation, differentiated
  through the tensor rotation);
- MoO₃ and spacer thicknesses.

The objective is the mean cross-polarized reflectance R_ps (p in, s out)
over 880–920 cm⁻¹, from 2D RCWA with 81 harmonics. Gradients run through
MoO₃'s measured dispersion.

The design is then drawn as a gdsfactory PCell of 15 × 15 cells and written
to GDS. The GDS is read back, and the unit cell is extracted from its
polygons. The stack is read from a LayerStack, and the design is
re-simulated at 81 and 225 harmonics. So what is verified is what would be
fabricated, not the optimiser's own model.

Needs the gdsfactory extra:  uv sync --extra gdsfactory
Run:  python examples/moo3_metasurface_gdsfactory.py   (writes to examples/output/)
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import gdsfactory as gf  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer  # noqa: E402

from hyperbolic_optics import dispersion, rcwa2d  # noqa: E402
from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

OUT = Path(__file__).parent / "output"
PERIOD = 4.0  # µm; subwavelength (λ ≈ 11 µm), so only the zeroth order leaves
BAND = np.linspace(880.0, 920.0, 5)  # cm⁻¹, where MoO₃ is in-plane hyperbolic
LAYER = (1, 0)
DESIGN_ORDER = (4, 4)  # 81 harmonics while optimising; verified at 225 below
EPS_KBR, EPS_SI = 2.31, 11.7  # transparent here


def gold(wavelength_um):
    """Drude gold, ω_p ≈ 7.28e4 cm⁻¹, γ ≈ 215 cm⁻¹ (Ordal et al.); approximate."""
    w = 1e4 / wavelength_um
    return 1.0 - 7.28e4**2 / (w**2 + 1j * 215.0 * w)


BOUNDS = {  # parameter: (low, high)
    "a_um": (0.3, 1.9),  # ellipse semi-axes: the cell is 4 µm, so they stay apart
    "b_um": (0.3, 1.9),
    "ellipse_rad": (-np.pi / 2, np.pi / 2),
    "crystal_deg": (0.0, 90.0),
    "moo3_um": (0.1, 1.5),
    "spacer_um": (0.2, 3.0),
}


def physical(raw):
    return {k: lo + (hi - lo) * jax.nn.sigmoid(raw[k]) for k, (lo, hi) in BOUNDS.items()}


def unconstrained(values):
    return {k: jnp.log((values[k] - lo) / (hi - values[k])) for k, (lo, hi) in BOUNDS.items()}


def ellipse(p, n_vertices=48):
    return rcwa2d.ellipse(
        (PERIOD / 2, PERIOD / 2), (p["a_um"], p["b_um"]), n_vertices, p["ellipse_rad"]
    )


def moo3(p):
    return hog.Oriented("MoO3", (0.0, 0.0, p["crystal_deg"]))


def reflectance(p, wavenumbers, max_order=DESIGN_ORDER, cell=None):
    """Zeroth-order R ``[F, out, in]`` at normal incidence (p = x, s = y)."""
    wavelengths = 1e4 / jnp.asarray(wavenumbers)  # [F]
    cell = cell or hog.UnitCell((PERIOD, PERIOD), (ellipse(p),))
    stack = [
        hog.StackLayer(p["moo3_um"], moo3(p), background=1.0),
        hog.StackLayer(p["spacer_um"], EPS_KBR),
        hog.StackLayer(0.2, gold),
    ]
    layers = hog.crossed_grating_layers(cell, stack, wavelengths)
    result = rcwa2d.solve_crossed_grating(
        layers, cell.period_um, wavelengths, 0.0, 1.0, EPS_SI, max_order
    )
    zeroth = int(np.flatnonzero((np.asarray(result.orders) == 0).all(axis=1))[0])
    return result.R[..., zeroth, :, :]


def optimise(start, steps=100, rate=0.06):
    def objective(raw):
        return reflectance(physical(raw), BAND)[:, 1, 0].mean()

    step_fn = jax.jit(jax.value_and_grad(objective))
    raw = unconstrained(start)
    first = {k: jnp.zeros(()) for k in raw}
    second = {k: jnp.zeros(()) for k in raw}
    history = []
    for step in range(1, steps + 1):
        value, grad = step_fn(raw)
        history.append(float(value))
        for k in raw:  # Adam, ascending
            first[k] = 0.9 * first[k] + 0.1 * grad[k]
            second[k] = 0.999 * second[k] + 0.001 * grad[k] ** 2
            m_hat, v_hat = first[k] / (1 - 0.9**step), second[k] / (1 - 0.999**step)
            raw[k] = raw[k] + rate * m_hat / (jnp.sqrt(v_hat) + 1e-12)
    return {k: float(v) for k, v in physical(raw).items()}, history


def layer_stack(design):
    """The PDK-side description: WAFER films, MoO₃ on LAYER, Si as the exit."""
    wafer = LogicalLayer(layer=gf.get_layer("WAFER"))
    spacer, film = design["spacer_um"], design["moo3_um"]
    return LayerStack(
        layers={
            "substrate": LayerLevel(
                layer=wafer, zmin=-1.2, thickness=1.0, material="si", mesh_order=10
            ),
            "mirror": LayerLevel(
                layer=wafer, zmin=-0.2, thickness=0.2, material="gold", mesh_order=5
            ),
            "spacer": LayerLevel(
                layer=wafer, zmin=0.0, thickness=spacer, material="kbr", mesh_order=5
            ),
            "moo3": LayerLevel(
                layer=LogicalLayer(layer=LAYER),
                zmin=spacer,
                thickness=film,
                material="moo3",
                mesh_order=1,
            ),  # fmt: skip
        }
    )


def _style(ax):
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b6a64")
    ax.tick_params(colors="#6b6a64", labelcolor="#1f1f1e", labelsize=8)


def main():
    OUT.mkdir(exist_ok=True)
    gf.gpdk.PDK.activate()
    start = {
        "a_um": 1.2, "b_um": 0.8, "ellipse_rad": 0.0, "crystal_deg": 20.0,
        "moo3_um": 0.5, "spacer_um": 1.5,
    }  # fmt: skip

    t0 = time.perf_counter()
    design, history = optimise(start)
    print(f"Optimised in {time.perf_counter() - t0:.0f} s ({len(history)} steps, "
          f"{(2 * DESIGN_ORDER[0] + 1) ** 2} harmonics):")  # fmt: skip
    for k, v in design.items():
        print(f"  {k:12s} {start[k]:.3f} → {v:.4f}")
    print(
        f"  mean R_ps over {BAND[0]:.0f}–{BAND[-1]:.0f} cm⁻¹: {history[0]:.3f} → {history[-1]:.4f}"
    )
    aligned = dict(design, crystal_deg=0.0, ellipse_rad=0.0)
    aligned_ps = float(reflectance(aligned, BAND)[:, 1, 0].mean())
    print(f"  with crystal and ellipse along x/y: {aligned_ps:.1e}")
    turned = dict(design, ellipse_rad=0.0)
    ellipse_only = dict(design, crystal_deg=0.0)
    print(f"  ellipse along x, crystal as designed: "
          f"{float(reflectance(turned, BAND)[:, 1, 0].mean()):.4f}; "
          f"crystal along x, ellipse as designed: "
          f"{float(reflectance(ellipse_only, BAND)[:, 1, 0].mean()):.1e}")  # fmt: skip

    # --- to GDS and back ---------------------------------------------------
    polygon = tuple(map(tuple, np.asarray(ellipse(design, 64)).tolist()))
    component = hog.periodic_array((PERIOD, PERIOD), (polygon,), (15, 15), layer=LAYER)
    gds = OUT / "moo3_metasurface.gds"
    component.write_gds(gds)
    component.delete()  # read the GDS back as itself, not as the cell in memory
    layout = gf.import_gds(gds)
    materials = {"moo3": moo3(design), "kbr": EPS_KBR, "gold": gold, "si": EPS_SI}
    tech = layer_stack(design)

    wavenumbers = np.linspace(850.0, 960.0, 45)
    fabricated, column, coarse = hog.simulate_crossed_grating(
        layout, LAYER, tech, 1e4 / wavenumbers, [0.0], max_order=DESIGN_ORDER,
        materials=materials,
    )  # fmt: skip
    print(f"\nWritten {gds.name}; read back {fabricated.n_cells[0]} × {fabricated.n_cells[1]} "
          f"cells, period {fabricated.period_um}, "
          f"{len(fabricated.polygons[0])}-vertex ellipse")  # fmt: skip
    print("  stack from the LayerStack:",
          [(layer.name, round(layer.thickness_um, 4)) for layer in column.layers])  # fmt: skip
    fine_wavenumbers = wavenumbers[::4]
    fine = []
    for w in fine_wavenumbers:  # one wavelength at a time: 225 harmonics is memory-heavy
        _, _, response = hog.simulate_crossed_grating(
            layout, LAYER, tech, [1e4 / w], [0.0], max_order=(7, 7), materials=materials
        )
        fine.append(response.R[0, 0, response.order(0, 0)])
    fine = np.array(fine)
    model = np.asarray(reflectance(design, wavenumbers))
    from_gds = coarse.R[:, 0, coarse.order(0, 0)]
    in_band = (wavenumbers >= BAND[0]) & (wavenumbers <= BAND[-1])
    fine_band = (fine_wavenumbers >= BAND[0]) & (fine_wavenumbers <= BAND[-1])
    print(f"  mean R_ps in band: optimiser model {model[in_band, 1, 0].mean():.4f}, "
          f"from the GDS {from_gds[in_band, 1, 0].mean():.4f} (81 harmonics); "
          f"225 harmonics at {fine_wavenumbers[fine_band].round(1)} cm⁻¹: "
          f"{fine[fine_band, 1, 0].round(4)}")  # fmt: skip

    # --- figure -----------------------------------------------------------
    ink, blue, orange, aqua, muted = "#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#6b6a64"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    _style(ax)
    band = np.linspace(800.0, 1010.0, 300)
    eps = np.asarray(dispersion.permittivity("MoO3", jnp.asarray(band)))
    ax.plot(band, eps[:, 0, 0].real, color=blue, linewidth=2, label="Re ε_x")
    ax.plot(band, eps[:, 1, 1].real, color=orange, linewidth=2, label="Re ε_y")
    ax.plot(band, eps[:, 2, 2].real, color=aqua, linewidth=2, label="Re ε_z")
    ax.axvspan(BAND[0], BAND[-1], color="#e4e3dd", alpha=0.8, linewidth=0,
               label="design band")  # fmt: skip
    ax.axhline(0.0, color=muted, linewidth=0.8)
    ax.set_ylim(-20, 20)
    ax.set_title("(a) α-MoO₃ dispersion (crystal frame)", fontsize=9, color=ink, loc="left")
    ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    ax.set_ylabel("permittivity", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[0, 1]
    _style(ax)
    ax.plot(np.arange(1, len(history) + 1), history, color=blue, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"(b) optimisation: mean R_ps, {BAND[0]:.0f}–{BAND[-1]:.0f} cm⁻¹",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("Adam step", fontsize=9, color=ink)
    ax.set_ylabel("cross-polarized reflectance", fontsize=9, color=ink)

    ax = axes[1, 0]
    _style(ax)
    ax.axvspan(BAND[0], BAND[-1], color="#e4e3dd", alpha=0.8, linewidth=0)
    ax.plot(wavenumbers, model[:, 1, 0], color=blue, linewidth=2,
            label="R_ps, optimiser model (81)")  # fmt: skip
    ax.plot(wavenumbers, from_gds[:, 1, 0], color=orange, linewidth=2, linestyle="--",
            label="R_ps, from the GDS (81)")  # fmt: skip
    ax.plot(fine_wavenumbers, fine[:, 1, 0], linestyle="none", marker="o", markersize=5,
            markerfacecolor="none", color=aqua, label="R_ps, from the GDS (225)")  # fmt: skip
    ax.plot(wavenumbers, from_gds[:, 0, 0], color=muted, linewidth=1.5,
            label="R_pp, from the GDS (81)")  # fmt: skip
    ax.set_ylim(0, 1)
    ax.set_title("(c) spectra, normal incidence, x-polarized in", fontsize=9, color=ink,
                 loc="left")  # fmt: skip
    ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    ax.set_ylabel("reflectance", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.14, 0.38))

    ax = axes[1, 1]
    _style(ax)
    polygons = layout.get_polygons_points(layers=[LAYER])[gf.get_layer(LAYER)]
    for shape in polygons:
        if shape[:, 0].max() < 3 * PERIOD and shape[:, 1].max() < 3 * PERIOD:
            ax.fill(shape[:, 0], shape[:, 1], color=blue, alpha=0.85, linewidth=0)
    angle = np.radians(design["crystal_deg"])
    centre = np.array([PERIOD * 1.5, PERIOD * 1.5])
    for label, (dx, dy), align in (
        ("ε_x < 0", (np.cos(angle), np.sin(angle)), "left"),
        ("ε_y", (-np.sin(angle), np.cos(angle)), "right"),
    ):
        tip = centre + 1.4 * np.array([dx, dy])
        arrow = {"arrowstyle": "->", "color": "white", "linewidth": 1.6}
        ax.annotate("", xy=tip, xytext=centre, arrowprops=arrow)
        box = {"boxstyle": "round,pad=0.2", "facecolor": "white", "linewidth": 0}
        ax.text(tip[0] + (0.15 if align == "left" else -0.15), tip[1] + 0.15, label,
                fontsize=8, color=ink, ha=align, va="bottom", bbox=box)  # fmt: skip
    ax.set_aspect("equal")
    ax.set_xlim(0, 3 * PERIOD)
    ax.set_ylim(0, 3 * PERIOD)
    ax.set_title("(d) the GDS, 3 × 3 of 15 × 15 cells; arrows: MoO₃ axes",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("x (µm)", fontsize=9, color=ink)
    ax.set_ylabel("y (µm)", fontsize=9, color=ink)

    fig.suptitle(
        f"α-MoO₃ metasurface on gold: gradient design → gdsfactory → GDS → re-simulation "
        f"(MoO₃ {design['moo3_um'] * 1e3:.0f} nm, KBr {design['spacer_um']:.2f} µm)",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "moo3_metasurface.png", dpi=160)
    print(f"\nFigure: {OUT / 'moo3_metasurface.png'}")


if __name__ == "__main__":
    main()
