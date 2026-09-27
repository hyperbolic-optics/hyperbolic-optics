#!/usr/bin/env python3
"""Negative refraction in tilted calcite at 1470 cm⁻¹, and a grating that uses it.

This mirrors FDTDX's ``examples/hyperbolic_asymmetric_refraction.py``
(https://github.com/ymahlau/fdtdx). In that example calcite is driven inside
its ordinary Reststrahlen band at 1470.17 cm⁻¹ (λ = 6.80 µm), where Re ε < 0
across the ordinary plane and Re ε > 0 along the optic axis. The axis is tilted
25° above the surface, in the plane of incidence. The tilt breaks the kx → −kx
mirror symmetry of the interface, so TM beams at +45° and −45° refract to +23°
and +53° -- one ordinarily, one negatively. There it takes a GPU FDTD run; here
it is an eigenmode of the 4×4 Berreman matrix, with this package's calcite
model (which FDTDX's two-pole fit was matched to).

The same asymmetry makes a *symmetric* grating diffract asymmetrically. With a
period of λ / sin 45° = 9.62 µm, a rectangular calcite grating at normal
incidence sends its ±1 orders into the crystal with exactly the wavevectors of
FDTDX's ±45° beams: the +1 order refracts positively and the −1 order
negatively, so both carry their energy toward +x. Nothing in the layout
distinguishes +x from −x; only the crystal does.

The script:

    1. Computes the energy-refraction angle in the tilted crystal against the
       angle of incidence, and checks the ±45° values against FDTDX's.
    2. Optimises an etched calcite grating on a calcite slab, by gradient
       ascent through calcite's dispersion, to send the transmitted light into
       the −1 order (the negatively refracted one) rather than the +1.
    3. Draws it as a gdsfactory PCell, writes the GDS, reads it back, and
       re-simulates it at higher order from the layout and a LayerStack, with
       the crystal cut stated as ``Oriented("Calcite", AXIS)``.

Needs the gdsfactory extra:  uv sync --extra gdsfactory
Run:  python benchmarks/calcite_negative_refraction_gdsfactory.py   (writes to benchmarks/output/)
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

from hyperbolic_optics import bcm, rcwa  # noqa: E402
from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

OUT = Path(__file__).parent / "output"
DRIVE_CM = 1470.17  # FDTDX's drive frequency, inside calcite's ordinary Reststrahlen band
WAVELENGTH = 1e4 / DRIVE_CM  # 6.80 µm
# The optic axis 65° from the normal (25° above the surface), in the plane of
# incidence. FDTDX's z points out of the crystal and this package's into the
# stack, so FDTDX's axis "towards +x and up" is rotationY = −65° here.
AXIS = (0.0, -65.0, 0.0)
CALCITE = hog.Oriented("Calcite", AXIS)
FDTDX_ANALYTIC = {45.0: 23.1, -45.0: 53.3}  # its closed form, Re ε, degrees
PERIOD = WAVELENGTH / np.sin(np.radians(45.0))  # ±1 orders at ±45°
LAYER = (1, 0)
DESIGN_ORDER = 20  # 41 orders while optimising; verified at 81 below


# ---------------------------------------------------------------------------
# 1. Refraction in the tilted crystal
# ---------------------------------------------------------------------------


def energy_angle(theta_deg, wavenumber=DRIVE_CM, real_part=False):
    """Direction of the time-averaged Poynting vector of the TM wave in the crystal.

    From the eigenmodes of the Berreman matrix at ``kx = sin θ`` (air above),
    with ``Ez`` and ``Hz`` restored from ``(εE)_z = −kx·Hy`` and ``Hz = kx·Ey``.
    Degrees from the inward normal, positive towards +x; the incident beam
    moves towards +x for θ > 0, so the sign differs from θ's under negative
    refraction.
    """
    eps = np.asarray(hog.material_tensor(CALCITE, 1e4 / wavenumber))
    if real_part:
        eps = eps.real + 0j
    kx = np.sin(np.radians(theta_deg))
    delta = np.asarray(bcm.system_matrix(jnp.asarray(eps), jnp.eye(3, dtype=complex), kx))
    kz, modes = np.linalg.eig(delta)
    best = None
    for i in range(4):
        ex, ey, hx, hy = modes[:, i]
        ez = -(eps[2, 0] * ex + eps[2, 1] * ey + kx * hy) / eps[2, 2]
        hz = kx * ey
        sz = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
        sx = 0.5 * np.real(ey * np.conj(hz) - ez * np.conj(hy))
        transverse_magnetic = abs(hy) > abs(hx)
        # the transmitted wave: energy into the stack (and decaying, if lossy)
        if transverse_magnetic and sz > 0 and (real_part or kz[i].imag >= 0):
            best = np.degrees(np.arctan2(sx, sz))
    if best is None:
        raise ValueError(f"no transmitted TM wave at {theta_deg}°")
    return float(best)


# ---------------------------------------------------------------------------
# 2. The grating: design
# ---------------------------------------------------------------------------

BOUNDS = {  # parameter: (low, high)
    "fill": (0.1, 0.9),
    "etch_um": (0.2, 4.0),
    "slab_um": (0.3, 10.0),  # unetched calcite below the grating
}


def physical(raw):
    return {k: lo + (hi - lo) * jax.nn.sigmoid(raw[k]) for k, (lo, hi) in BOUNDS.items()}


def unconstrained(values):
    return {k: jnp.log((values[k] - lo) / (hi - values[k])) for k, (lo, hi) in BOUNDS.items()}


def stack(p, material=CALCITE):
    return [
        hog.StackLayer(p["etch_um"], material, background=1.0, name="calcite ridges"),
        hog.StackLayer(p["slab_um"], material, name="calcite slab"),
    ]


def first_orders(p, wavenumber=DRIVE_CM, max_order=DESIGN_ORDER, grating=None, material=CALCITE):
    """(T₋₁, T₊₁) for normally incident p (TM) light; free-standing, air below."""
    wavelength = 1e4 / wavenumber
    grating = grating or hog.LineGrating(period_um=PERIOD, fill=p["fill"])
    layers = hog.grating_layers(grating, stack(p, material), wavelength)
    res = rcwa.solve_grating(layers, grating.period_um, wavelength, 0.0, 1.0, 1.0, max_order)
    minus, plus = max_order - 1, max_order + 1
    return res.T[minus, :, 0].sum(), res.T[plus, :, 0].sum()


def optimise(start, steps=150, rate=0.08):
    def objective(raw):
        minus, plus = first_orders(physical(raw))
        return minus - plus, (minus, plus)

    step_fn = jax.jit(jax.value_and_grad(objective, has_aux=True))
    raw = unconstrained(start)
    first = {k: jnp.zeros(()) for k in raw}
    second = {k: jnp.zeros(()) for k in raw}
    history = []
    for step in range(1, steps + 1):
        (_, (minus, plus)), grad = step_fn(raw)
        history.append((float(minus), float(plus)))
        for k in raw:  # Adam, ascending
            first[k] = 0.9 * first[k] + 0.1 * grad[k]
            second[k] = 0.999 * second[k] + 0.001 * grad[k] ** 2
            m_hat, v_hat = first[k] / (1 - 0.9**step), second[k] / (1 - 0.999**step)
            raw[k] = raw[k] + rate * m_hat / (jnp.sqrt(v_hat) + 1e-12)
    return {k: float(v) for k, v in physical(raw).items()}, np.array(history)


# ---------------------------------------------------------------------------
# 3. The layout and its stack
# ---------------------------------------------------------------------------


def layer_stack(design):
    """A free-standing calcite slab with ridges on LAYER; air below is the exit."""
    wafer = LogicalLayer(layer=gf.get_layer("WAFER"))
    slab = design["slab_um"]
    return LayerStack(
        layers={
            "air_below": LayerLevel(
                layer=wafer, zmin=-1.0, thickness=1.0, material="air", mesh_order=10
            ),
            "calcite_slab": LayerLevel(
                layer=wafer, zmin=0.0, thickness=slab, material="calcite", mesh_order=2
            ),
            "calcite_ridges": LayerLevel(
                layer=LogicalLayer(layer=LAYER),
                zmin=slab,
                thickness=design["etch_um"],
                material="calcite",
                mesh_order=2,
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

    # --- 1. refraction ------------------------------------------------------
    print(f"Calcite at {DRIVE_CM} cm⁻¹ (λ = {WAVELENGTH:.3f} µm), optic axis at {AXIS}:")
    eps = np.asarray(hog.material_tensor(CALCITE, WAVELENGTH))
    print(f"  lab-frame ε_xx {eps[0, 0]:.3f}, ε_zz {eps[2, 2]:.3f}, ε_xz {eps[0, 2]:.3f}")
    for theta, fdtdx in FDTDX_ANALYTIC.items():
        real, lossy = energy_angle(theta, real_part=True), energy_angle(theta)
        kind = "negative" if real * theta < 0 else "positive"
        print(f"  {theta:+.0f}° in → energy at {real:+.1f}° (Re ε; FDTDX {fdtdx:+.1f}°), "
              f"{lossy:+.1f}° with loss: {kind} refraction")  # fmt: skip
    angles = np.linspace(-80.0, 80.0, 161)
    refraction = np.array([energy_angle(a) for a in angles])

    # --- 2. design ----------------------------------------------------------
    start = {"fill": 0.5, "etch_um": 1.5, "slab_um": 1.5}
    t0 = time.perf_counter()
    design, history = optimise(start)
    print(f"\nOptimised in {time.perf_counter() - t0:.0f} s ({len(history)} steps, "
          f"{2 * DESIGN_ORDER + 1} orders), period fixed at {PERIOD:.3f} µm:")  # fmt: skip
    for k, v in design.items():
        print(f"  {k:8s} {start[k]:.3f} → {v:.4f}")
    (m0, p0), (m1, p1) = history[0], history[-1]
    print(f"  T₋₁ / T₊₁: {m0:.4f} / {p0:.4f} → {m1:.4f} / {p1:.4f}")
    print(f"  inside the crystal the −1 order's energy runs at {energy_angle(-45.0):+.1f}° and "
          f"the +1 order's at {energy_angle(45.0):+.1f}°: both towards +x")  # fmt: skip

    # --- 3. GDS → LayerStack → re-simulation --------------------------------
    component = hog.line_grating(
        period_um=PERIOD, fill=design["fill"], n_periods=20, length_um=200.0, layer=LAYER
    )
    gds = OUT / "calcite_grating.gds"
    component.write_gds(gds)
    component.delete()  # so the GDS is read back as itself, not as the cell in memory
    layout = gf.import_gds(gds)
    tech = layer_stack(design)
    wavenumbers = np.linspace(1420.0, 1540.0, 61)
    fabricated, column, response = hog.simulate_grating(
        layout, LAYER, tech, 1e4 / wavenumbers, [0.0], max_order=40,
        materials={"calcite": CALCITE},
    )  # fmt: skip
    print(f"\nWritten {gds.name}; read back: period {fabricated.period_um:.4f} µm, "
          f"fill {fabricated.fill:.4f}")  # fmt: skip
    print("  stack from the LayerStack:",
          [(layer.name, round(layer.thickness_um, 4)) for layer in column.layers])  # fmt: skip
    T = response.T[:, 0]  # [wavenumber, order, out, in]
    t_minus = T[:, response.order(-1), :, 0].sum(-1)
    t_plus = T[:, response.order(1), :, 0].sum(-1)
    s_total = T[:, :, :, 1].sum((-2, -1))
    at = int(np.argmin(np.abs(wavenumbers - DRIVE_CM)))
    print(f"  from the GDS, 81 orders, {wavenumbers[at]:.1f} cm⁻¹: "
          f"T₋₁ {t_minus[at]:.4f}, T₊₁ {t_plus[at]:.4f}")  # fmt: skip
    print(f"  s (TE) light, all orders, {wavenumbers[at]:.1f} cm⁻¹: T = {s_total[at]:.1e} -- "
          "ε_yy = ε_o < 0 blocks it (less so towards the LO edge, where ε_o → 0)")  # fmt: skip
    untilted = hog.Oriented("Calcite", hog.z_cut())
    flat = first_orders(design, material=untilted)
    print(f"  the same grating with the axis along the normal: T₋₁ {float(flat[0]):.1e}, "
          f"T₊₁ {float(flat[1]):.1e} (TM evanescent: ε_xx < 0 < ε_zz)")  # fmt: skip

    # --- figure -------------------------------------------------------------
    ink, blue, orange, muted = "#1f1f1e", "#2a78d6", "#eb6834", "#6b6a64"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    _style(ax)
    negative = refraction * angles < 0
    ax.fill_between(angles, -90, 90, where=negative, color="#dcdad2", linewidth=0,
                    label="negative refraction")  # fmt: skip
    ax.plot(angles, refraction, color=blue, linewidth=2, label="this package (Berreman modes)")
    ax.plot(list(FDTDX_ANALYTIC), list(FDTDX_ANALYTIC.values()), linestyle="none", marker="o",
            markersize=7, markerfacecolor="none", color=orange,
            label="FDTDX example, ±45° (analytic)")  # fmt: skip
    ax.plot(
        angles,
        angles,
        color=muted,
        linewidth=1,
        linestyle=":",
        label="undeviated (energy along the incident beam)",
    )
    ax.set_ylim(-90, 90)
    ax.set_title(f"(a) energy refraction in tilted calcite, {DRIVE_CM:.0f} cm⁻¹, TM",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("angle of incidence (°)", fontsize=9, color=ink)
    ax.set_ylabel("energy direction in the crystal (°)", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[0, 1]
    _style(ax)
    steps = np.arange(1, len(history) + 1)
    ax.plot(steps, history[:, 0], color=blue, linewidth=2, label="T₋₁ (refracted negatively)")
    ax.plot(steps, history[:, 1], color=orange, linewidth=2, label="T₊₁ (refracted positively)")
    ax.set_title(f"(b) optimisation, normal incidence, p in, Λ = {PERIOD:.2f} µm",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("Adam step", fontsize=9, color=ink)
    ax.set_ylabel("diffraction efficiency", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    _style(ax)
    ax.plot(wavenumbers, t_minus, color=blue, linewidth=2, label="T₋₁")
    ax.plot(wavenumbers, t_plus, color=orange, linewidth=2, label="T₊₁")
    ax.axvline(DRIVE_CM, color=ink, linewidth=1, linestyle=":")
    ax.set_title("(c) from the GDS and the LayerStack, 81 orders, p in",
                 fontsize=9, color=ink, loc="left")  # fmt: skip
    ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=9, color=ink)
    ax.set_ylabel("diffraction efficiency", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    _style(ax)
    polygons = layout.get_polygons_points(layers=[LAYER])[gf.get_layer(LAYER)]
    for polygon in sorted(polygons, key=lambda q: q[:, 0].min())[:5]:
        ax.fill(polygon[:, 0], polygon[:, 1], color=blue, alpha=0.85, linewidth=0)
    ax.set_xlim(-1.0, 5 * fabricated.period_um)
    ax.set_ylim(-5, fabricated.length_um + 5)
    ax.set_title(
        f"(d) {gds.name}, 5 of {fabricated.n_periods} ridges: width "
        f"{fabricated.fill * fabricated.period_um:.3f} µm",
        fontsize=9, color=ink, loc="left",
    )  # fmt: skip
    ax.set_xlabel("x (µm)", fontsize=9, color=ink)
    ax.set_ylabel("y (µm)", fontsize=9, color=ink)

    fig.suptitle(
        f"Calcite, optic axis 25° above the surface: a symmetric grating that diffracts one way "
        f"(etch {design['etch_um']:.2f} µm on {design['slab_um']:.2f} µm, free-standing)",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "calcite_negative_refraction.png", dpi=160)
    print(f"\nFigure: {OUT / 'calcite_negative_refraction.png'}")


if __name__ == "__main__":
    main()
