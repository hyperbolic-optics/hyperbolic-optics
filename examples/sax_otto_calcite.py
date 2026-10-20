#!/usr/bin/env python3
"""Prism coupling to calcite's surface phonon polaritons, as a SAX circuit.

    KRS-5 prism | air gap (d) | x-cut calcite, 100 µm | air

In calcite's ordinary Reststrahlen band (ε_o < 0 < ε_e), a TM wave across the
optic axis binds to the surface as a polariton; along the axis the crystal is
hyperbolic. Each element is a planar SAX model sharing ``wl`` and ``kx``. The
script checks the circuit against the whole stack, maps the polariton dip,
shows p→s conversion with the axis turned, and tunes the gap to critical
coupling by gradient descent through the circuit.

Run:  uv run --with sax python examples/sax_otto_calcite.py
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import sax  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import bcm, dispersion  # noqa: E402
from hyperbolic_optics.integrations import sax as hsax  # noqa: E402

OUT = Path(__file__).parent / "output"
EPS_PRISM = 2.37**2  # KRS-5, transparent here
CALCITE_UM = 100.0
ACROSS, OBLIQUE = 90.0, 60.0  # optic-axis azimuth from the plane of incidence, degrees


def calcite(azimuth_deg):
    """x-cut: the optic axis in the surface, ``azimuth_deg`` from the plane of incidence."""
    rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(azimuth_deg))
    return lambda wl: bcm.rotate_tensor(dispersion.at_wavelength("Calcite", wl), rotation)


def otto(azimuth_deg=0.0, backend="default"):
    """The circuit: prism face → gap (thickness a setting) → calcite sample."""

    def gap(wl=6.8, kx=0.0, gap_um=2.0):
        return hsax.planar_model([(1.0, gap_um)], 1.0, 1.0)(wl=wl, kx=kx)

    models = {
        "prism": hsax.planar_model([], EPS_PRISM, 1.0),
        "gap": gap,
        "sample": hsax.planar_model([(calcite(azimuth_deg), CALCITE_UM)], 1.0, 1.0),
    }
    netlist = {
        "instances": {"prism": "prism", "gap": "gap", "sample": "sample"},
        "connections": {"prism,bottom": "gap,top", "gap,bottom": "sample,top"},
        "ports": {"top": "prism,top", "bottom": "sample,bottom"},
    }
    circuit, _ = sax.circuit(netlist, models, backend=backend)
    return circuit


def one_piece(wl, kx, gap_um, azimuth_deg=0.0):
    tensors = jnp.stack(
        jnp.broadcast_arrays(jnp.eye(3, dtype=complex), calcite(azimuth_deg)(wl)), axis=-3
    )
    return bcm.stack_smatrix(tensors, jnp.array([gap_um, CALCITE_UM]), wl, kx, EPS_PRISM, 1.0)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b6a64")
    ax.tick_params(colors="#6b6a64", labelcolor="#1f1f1e", labelsize=8)


def main():
    OUT.mkdir(exist_ok=True)
    ink = "#1f1f1e"

    # --- 1. the circuit is the stack --------------------------------------
    wl, kx, gap_um = 1e4 / 1470.17, 1.3, 1.5
    sd = otto(OBLIQUE)(wl=wl, kx=kx, gap={"gap_um": gap_um})
    whole = np.asarray(one_piece(wl, kx, gap_um, OBLIQUE).s)
    worst = max(
        abs(complex(sd[p_in, p_out]) - whole[i_out, i_in])
        for i_in, p_in in enumerate(hsax.PORTS)
        for i_out, p_out in enumerate(hsax.PORTS)
    )
    print(f"SAX circuit vs one-piece stack, all 16 entries: max |Δs| = {worst:.1e}")

    # --- 2 and 3. reflectance maps ----------------------------------------
    wavenumbers = np.linspace(1410.0, 1560.0, 151)
    kxs = np.linspace(1.02, 2.3, 129)
    grid_w, grid_k = np.meshgrid(wavenumbers, kxs, indexing="ij")
    maps = {}
    for azimuth in (ACROSS, OBLIQUE):
        sd = otto(azimuth)(wl=1e4 / grid_w.ravel(), kx=grid_k.ravel(), gap={"gap_um": gap_um})
        maps[azimuth] = {
            "pp": np.abs(np.asarray(sd["top@p", "top@p"])).reshape(grid_w.shape) ** 2,
            "ps": np.abs(np.asarray(sd["top@p", "top@s"])).reshape(grid_w.shape) ** 2,
        }
    across, oblique = maps[ACROSS]["ps"].max(), maps[OBLIQUE]["ps"].max()
    print(f"largest p→s conversion, axis across the plane of incidence: {across:.1e}; "
          f"at {OBLIQUE:.0f}°: {oblique:.3f}")  # fmt: skip

    # --- 4. critical coupling by gradient descent through the circuit ------
    target_w = 1470.17
    row = np.argmin(np.abs(wavenumbers - target_w))
    target_kx = float(kxs[np.argmin(maps[ACROSS]["pp"][row])])
    circuit = otto(ACROSS, backend="filipsson_gunnar")

    def reflectance(params):
        log_gap, kx = params
        sd = circuit(wl=1e4 / target_w, kx=kx, gap={"gap_um": jnp.exp(log_gap)})
        return jnp.abs(sd["top@p", "top@p"]) ** 2

    # gap and kx together: the resonance moves as the gap loads it
    step = jax.jit(jax.value_and_grad(reflectance))
    params = jnp.array([jnp.log(3.0), target_kx])  # start well away from critical coupling
    first, second, history = jnp.zeros(2), jnp.zeros(2), []
    for i in range(1, 201):  # Adam
        value, grad = step(params)
        history.append((float(jnp.exp(params[0])), float(params[1]), float(value)))
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        params = params - 0.02 * (first / (1 - 0.9**i)) / (
            jnp.sqrt(second / (1 - 0.999**i)) + 1e-12
        )
    best_gap, best_kx = float(jnp.exp(params[0])), float(params[1])
    best_r = float(reflectance(params))
    print(f"critical coupling at {target_w} cm⁻¹: gap 3.00 → {best_gap:.3f} µm, kx "
          f"{target_kx:.3f} → {best_kx:.4f}, R_pp {history[0][2]:.3f} → {best_r:.1e}")  # fmt: skip
    gaps = np.linspace(0.3, 4.0, 75)
    sweep = [float(reflectance(jnp.array([np.log(g), best_kx]))) for g in gaps]

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.6), constrained_layout=True)
    panels = (
        (maps[ACROSS]["pp"], "(a) R_pp, optic axis across the plane of incidence", "Greys_r"),
        (maps[OBLIQUE]["pp"], f"(b) R_pp, optic axis at {OBLIQUE:.0f}°", "Greys_r"),
        (maps[OBLIQUE]["ps"], f"(c) R_ps (p→s), optic axis at {OBLIQUE:.0f}°", "Blues"),
    )
    for ax, (values, title, cmap) in zip(axes[:3], panels, strict=True):
        _style(ax)
        image = ax.pcolormesh(kxs, wavenumbers, values, cmap=cmap, shading="auto")
        fig.colorbar(image, ax=ax)
        ax.set_title(title, fontsize=9, color=ink, loc="left")
        ax.set_xlabel("in-plane wavevector kx / k₀", fontsize=9, color=ink)
        ax.set_ylabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    axes[0].plot([target_kx], [target_w], marker="o", markersize=7, markerfacecolor="none",
                 color="#eb6834")  # fmt: skip

    ax = axes[3]
    _style(ax)
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.plot(gaps, sweep, color="#2a78d6", linewidth=2, label="R_pp against the gap, at best kx")
    ax.plot([h[0] for h in history[::5]], [h[2] for h in history[::5]], linestyle="none",
            marker="o", markersize=4, markerfacecolor="none", color="#eb6834",
            label="Adam on (gap, kx) through the SAX circuit")  # fmt: skip
    ax.set_title(f"(d) critical coupling at {target_w:.0f} cm⁻¹: R_pp at kx = {best_kx:.3f}",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("air gap (µm)", fontsize=9, color=ink)
    ax.set_ylabel("R_pp", fontsize=9, color=ink)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Otto coupling to calcite's surface phonon polaritons as a SAX circuit: KRS-5 prism | "
        f"air gap ({gap_um:g} µm in a–c) | x-cut calcite, {CALCITE_UM:.0f} µm",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "sax_otto_calcite.png", dpi=160)
    print(f"Figure: {OUT / 'sax_otto_calcite.png'}")


if __name__ == "__main__":
    main()
