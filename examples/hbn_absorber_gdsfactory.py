#!/usr/bin/env python3
"""Design a narrowband hBN ribbon absorber, export it with gdsfactory, verify the GDS.

hBN is hyperbolic between 1370 and 1610 cm⁻¹ (ε⊥ < 0 < ε∥). Ribbons cut from a
flake support phonon-polariton resonances there, and above a gold mirror they
can absorb nearly all of the incident light at one frequency -- the basis of
narrowband thermal emitters and mid-IR absorbers:

    air | hBN ribbons (h) | CaF₂ spacer (s) | gold | Si

The period, ribbon width (fill), hBN thickness and spacer thickness are
optimised by gradient ascent to maximise the absorption of TM (p) light at
1500 cm⁻¹, normal incidence. The gradients run through hBN's measured
dispersion (hyperbolic_optics.dispersion), so the design stays tied to the real
material across the band.

The optimised geometry is then drawn as a gdsfactory PCell and written to GDS;
the GDS is read back, the grating is extracted from its polygons, and the
design is re-simulated from that geometry at higher Fourier order -- so what is
verified is what would be fabricated, not the optimiser's own model.

Needs the gdsfactory extra:  uv sync --extra gdsfactory
Run:  python examples/hbn_absorber_gdsfactory.py   (writes to examples/output/)
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

from hyperbolic_optics import dispersion, rcwa  # noqa: E402
from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

OUT = Path(__file__).parent / "output"
TARGET_CM = 1500.0
LAYER = (1, 0)
DESIGN_ORDER = 20  # 41 orders while optimising; verified at 81 and 161 below
EPS_CAF2, EPS_SI = 1.85, 11.7  # transparent in this band


def gold(wavelength_um):
    """Drude gold, ω_p ≈ 7.28e4 cm⁻¹, γ ≈ 215 cm⁻¹ (Ordal et al.); approximate."""
    w = 1e4 / wavelength_um
    return 1.0 - 7.28e4**2 / (w**2 + 1j * 215.0 * w)


BOUNDS = {  # parameter: (low, high)
    "period_um": (1.0, 4.0),
    "fill": (0.1, 0.9),
    "hbn_um": (0.02, 0.3),
    "spacer_um": (0.1, 2.0),
}


def physical(raw):
    """Map unconstrained parameters into their bounds."""
    return {k: lo + (hi - lo) * jax.nn.sigmoid(raw[k]) for k, (lo, hi) in BOUNDS.items()}


def unconstrained(values):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        u = (values[k] - lo) / (hi - lo)
        out[k] = jnp.log(u / (1 - u))
    return out


def stack(p, ribbons="hBN"):
    return [
        hog.StackLayer(p["hbn_um"], ribbons, background=1.0),
        hog.StackLayer(p["spacer_um"], EPS_CAF2),
        hog.StackLayer(0.2, gold),
    ]


def absorption(p, wavenumber, max_order, grating=None, ribbons="hBN"):
    """Absorbed fraction of normally incident p (TM) light: 1 − R − T over all orders."""
    wavelength = 1e4 / wavenumber
    grating = grating or hog.LineGrating(period_um=p["period_um"], fill=p["fill"])
    layers = hog.grating_layers(grating, stack(p, ribbons), wavelength)
    res = rcwa.solve_grating(layers, grating.period_um, wavelength, 0.0, 1.0, EPS_SI, max_order)
    return 1.0 - jnp.sum(res.R[..., 0]) - jnp.sum(res.T[..., 0])


def optimise(start, steps=150, rate=0.08):
    objective = jax.jit(
        jax.value_and_grad(lambda r: -absorption(physical(r), TARGET_CM, DESIGN_ORDER))
    )
    raw = unconstrained(start)
    first = {k: jnp.zeros(()) for k in raw}
    second = {k: jnp.zeros(()) for k in raw}
    history = []
    for step in range(1, steps + 1):
        loss, grad = objective(raw)
        history.append(-float(loss))
        for k in raw:  # Adam
            first[k] = 0.9 * first[k] + 0.1 * grad[k]
            second[k] = 0.999 * second[k] + 0.001 * grad[k] ** 2
            m_hat, v_hat = first[k] / (1 - 0.9**step), second[k] / (1 - 0.999**step)
            raw[k] = raw[k] - rate * m_hat / (jnp.sqrt(v_hat) + 1e-12)
    return {k: float(v) for k, v in physical(raw).items()}, history


def spectrum(p, wavenumbers, max_order, grating=None, ribbons="hBN"):
    f = jax.jit(lambda w: absorption(p, w, max_order, grating, ribbons))
    return np.array([float(f(w)) for w in wavenumbers])


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
    start = {"period_um": 2.0, "fill": 0.5, "hbn_um": 0.1, "spacer_um": 0.8}

    t0 = time.perf_counter()
    design, history = optimise(start)
    elapsed, orders = time.perf_counter() - t0, 2 * DESIGN_ORDER + 1
    print(f"Optimised in {elapsed:.0f} s ({len(history)} steps, {orders} orders):")
    for k, v in design.items():
        print(f"  {k:10s} {start[k]:.3f} → {v:.4f}")
    print(f"  absorption at {TARGET_CM:.0f} cm⁻¹: {history[0]:.3f} → {history[-1]:.4f}")

    # --- to GDS and back ---------------------------------------------------
    component = hog.line_grating(
        period_um=design["period_um"],
        fill=design["fill"],
        n_periods=30,
        length_um=60.0,
        layer=LAYER,
    )
    gds = OUT / "hbn_absorber.gds"
    component.write_gds(gds)
    fabricated = hog.extract_line_grating(gf.import_gds(gds), LAYER)
    grid_nm = component.kcl.dbu * 1e3
    print(f"\nWritten {gds.name}; read back: period {fabricated.period_um:.4f} µm, "
          f"fill {fabricated.fill:.4f} (snapped to the {grid_nm:.0f} nm grid)")  # fmt: skip

    # --- re-simulate the fabricated geometry ------------------------------
    at_target = {n: float(absorption(design, TARGET_CM, n, fabricated)) for n in (20, 40, 80)}
    for n, a in at_target.items():
        print(
            f"  from the GDS, {2 * n + 1:3d} orders: absorption at {TARGET_CM:.0f} cm⁻¹ = {a:.4f}"
        )
    mirror = float(absorption(design, TARGET_CM, 0, fabricated, ribbons=1.0))
    print(f"  without the hBN (bare mirror and spacer): {mirror:.4f} -- the gold's own share")

    wavenumbers = np.linspace(1380.0, 1600.0, 111)
    spectra = {
        "initial design": spectrum(start, wavenumbers, DESIGN_ORDER),
        "optimised (41 orders)": spectrum(design, wavenumbers, DESIGN_ORDER),
        "from the GDS (81 orders)": spectrum(design, wavenumbers, 40, fabricated),
    }
    coarse = wavenumbers[::5]
    from_gds_161 = spectrum(design, coarse, 80, fabricated)

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    ink, blue, orange, aqua, muted = "#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#6b6a64"

    ax = axes[0, 0]
    _style(ax)
    band = np.linspace(1350.0, 1650.0, 301)
    eps = np.asarray(dispersion.permittivity("hBN", jnp.asarray(band)))
    ax.plot(
        band, np.real(eps[:, 0, 0]), color=blue, linewidth=2, label="Re ε⊥ (in the flake plane)"
    )
    ax.plot(band, np.real(eps[:, 2, 2]), color=orange, linewidth=2, label="Re ε∥ (out of plane)")
    hyperbolic = np.real(eps[:, 0, 0]) * np.real(eps[:, 2, 2]) < 0
    ax.fill_between(band, -40, 40, where=hyperbolic, color="#e4e3dd", alpha=0.6, linewidth=0,
                    label="hyperbolic (ε⊥ · ε∥ < 0)")  # fmt: skip
    ax.axvline(TARGET_CM, color=ink, linewidth=1, linestyle=":")
    ax.set_ylim(-40, 40)
    ax.set_title("(a) hBN dispersion used throughout", fontsize=9, color=ink, loc="left")
    ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    ax.set_ylabel("permittivity", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[0, 1]
    _style(ax)
    ax.plot(np.arange(1, len(history) + 1), history, color=blue, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"(b) optimisation: absorption at {TARGET_CM:.0f} cm⁻¹", fontsize=9, color=ink,
                 loc="left")  # fmt: skip
    ax.set_xlabel("Adam step", fontsize=9, color=ink)
    ax.set_ylabel("absorption, p in", fontsize=9, color=ink)

    ax = axes[1, 0]
    _style(ax)
    for (label, values), color, style in zip(
        spectra.items(), (muted, blue, orange), (":", "-", "--"), strict=True
    ):
        ax.plot(wavenumbers, values, color=color, linewidth=2, linestyle=style, label=label)
    ax.plot(coarse, from_gds_161, linestyle="none", marker="o", markersize=5,
            markerfacecolor="none", color=aqua, label="from the GDS (161 orders)")  # fmt: skip
    ax.axvline(TARGET_CM, color=ink, linewidth=1, linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_title(
        "(c) absorption spectra, normal incidence, p in", fontsize=9, color=ink, loc="left"
    )
    ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    ax.set_ylabel("absorption", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 1]
    _style(ax)
    polygons = gf.import_gds(gds).get_polygons_points(layers=[LAYER])[gf.get_layer(LAYER)]
    for polygon in sorted(polygons, key=lambda q: q[:, 0].min())[:6]:
        ax.fill(polygon[:, 0], polygon[:, 1], color=blue, alpha=0.85, linewidth=0)
    ax.set_xlim(-0.5, 6 * fabricated.period_um)
    ax.set_ylim(-2, fabricated.length_um + 2)
    ax.set_title(
        f"(d) {gds.name}, 6 of {fabricated.n_periods} ribbons: period "
        f"{fabricated.period_um:.3f} µm, width {fabricated.fill * fabricated.period_um:.3f} µm",
        fontsize=9, color=ink, loc="left",
    )  # fmt: skip
    ax.set_xlabel("x (µm)", fontsize=9, color=ink)
    ax.set_ylabel("y (µm)", fontsize=9, color=ink)

    fig.suptitle(
        f"hBN ribbon absorber: gradient design → gdsfactory → GDS → re-simulation "
        f"(hBN {design['hbn_um'] * 1e3:.0f} nm, CaF₂ {design['spacer_um']:.2f} µm, gold)",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "hbn_absorber.png", dpi=160)
    print(f"\nFigure: {OUT / 'hbn_absorber.png'}")


if __name__ == "__main__":
    main()
