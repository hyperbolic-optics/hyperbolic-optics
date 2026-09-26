"""Full-tensor hyperbolic grating: tilted-axis hBN ridges across the Reststrahlen band.

hBN between 1370 and 1610 cm⁻¹ has ε⊥ < 0 < ε∥ (type-II hyperbolic). Its optic
axis is tilted 40° out of the plane of the layers and turned 25° about the
normal, so the permittivity has every off-diagonal component -- the case FMMAX
cannot represent. The dispersion is the package's own hBN model.

    air | hBN ridges, period 8 µm, fill 0.5, 0.5 µm thick | ε = 2 substrate
    incidence 20° in the x-z plane

Figure panels:

(a) Uniform-film limit (fill 1) against the NumPy engine, an independent
    eigendecomposition-based solver: total R and T for p and s input.
(b) The grating's zeroth-order R and T, the first diffracted order and the
    absorption, at 41 and 81 Fourier orders.
(c) Convergence in Fourier order at three wavenumbers.
(d, e) The first-order efficiency and its exact gradient against fill factor, at
    three truncations, with central finite differences. The gradient is exact
    for each truncated model, but hyperbolic ridges add a truncation ripple
    whose amplitude falls with more orders while its frequency rises, so the
    ripple in the slope does not fade.
(f) Energy conservation of the same grating made lossless (Im ε → 0), which has
    the largest rounding error of any case tested.

Run:  uv run python benchmarks/hyperbolic_grating/hbn_grating.py
"""

import argparse
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import bcm, rcwa  # noqa: E402
from hyperbolic_optics.fields import FieldProfile  # noqa: E402
from hyperbolic_optics.materials import create_material  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402

PERIOD_UM, FILL, THICKNESS_UM, THETA_DEG, EPS_SUB = 8.0, 0.5, 0.5, 20.0, 2.0
TILT_DEG, TURN_DEG = 40.0, 25.0
WAVENUMBERS = np.linspace(1360.0, 1620.0, 53)
EYE = jnp.eye(3, dtype=complex)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#1f1f1e", "#6b6a64", "#e4e3dd",
)  # fmt: skip
HBN = create_material("hBN")


def tensor(wavenumber: float, engine_offset: bool = False) -> jax.Array:
    """Lab-frame hBN tensor. ``engine_offset`` adds the 1e-8 / 1e-9 rad that a payload
    ``Crystal Layer`` adds to rotationY / rotationZ, so panel (a) compares solvers
    on identical inputs."""
    crystal = jnp.asarray(HBN.fetch_permittivity_tensor_for_freq(wavenumber)).reshape(3, 3)
    extra_y, extra_z = (1e-8, 1e-9) if engine_offset else (0.0, 0.0)
    rotation = bcm.rotation_matrix(
        0.0, jnp.radians(TILT_DEG) + extra_y, jnp.radians(TURN_DEG) + extra_z
    )
    return bcm.rotate_tensor(crystal, rotation)


def _solve(eps, wavelength, fill, max_order):
    layer = rcwa.lamellar_layer(eps, EYE, fill, THICKNESS_UM)
    return rcwa.solve_grating(
        [layer], PERIOD_UM, wavelength, jnp.radians(THETA_DEG), 1.0, EPS_SUB, max_order
    )


SOLVE = jax.jit(_solve, static_argnames="max_order")


def efficiencies(eps, wavelength, max_order, fill=FILL):
    """Per input polarization: R0, T0, T−1 (summed over output) and absorption."""
    res = SOLVE(eps, wavelength, fill, max_order)
    R, T = np.asarray(res.R).sum(axis=-2), np.asarray(res.T).sum(axis=-2)  # [order, in]
    centre = max_order
    return {
        "R0": R[centre], "T0": T[centre], "T-1": T[centre - 1],
        "A": 1.0 - R.sum(axis=0) - T.sum(axis=0),
        "R": R, "T": T,
    }  # fmt: skip


def numpy_engine_film(wavenumbers):
    """The unpatterned tilted-hBN film through Structure + FieldProfile."""
    payload = {
        "ScenarioData": {
            "type": "Incident",
            "incidentAngle": [THETA_DEG],
            "frequency": wavenumbers.tolist(),
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 1.0},
            {
                "type": "Crystal Layer",
                "material": "hBN",
                "thickness": THICKNESS_UM,
                "rotationY": TILT_DEG,
                "rotationZ": TURN_DEG,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": EPS_SUB},
        ],
    }
    structure = Structure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure.execute(payload, backend="scattering")
    fp = FieldProfile(structure)
    return {pol: (np.ravel(fp.reflectance(pol)), np.ravel(fp.transmittance(pol))) for pol in "ps"}


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def _label(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wavelengths = 1e4 / WAVENUMBERS
    eps_all = [tensor(v) for v in WAVENUMBERS]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.4), constrained_layout=True)

    # (a) uniform-film limit against the NumPy engine
    ax = axes[0, 0]
    _style(ax)
    reference = numpy_engine_film(WAVENUMBERS)
    engine_eps = [tensor(v, engine_offset=True) for v in WAVENUMBERS]
    film = [
        rcwa.solve_grating(
            [rcwa.uniform_layer(eps, THICKNESS_UM)], PERIOD_UM, w, jnp.radians(THETA_DEG),
            1.0, EPS_SUB, 0,
        )
        for eps, w in zip(engine_eps, wavelengths, strict=True)
    ]  # fmt: skip
    worst_film = 0.0
    for index, (pol, color) in enumerate((("p", BLUE), ("s", ORANGE))):
        for quantity, row, style in (("R", 0, "-"), ("T", 1, "--")):
            ours = np.array([np.asarray(getattr(f, quantity))[0, :, index].sum() for f in film])
            theirs = reference[pol][row]
            worst_film = max(worst_film, float(np.max(np.abs(ours - theirs))))
            ax.plot(WAVENUMBERS, ours, color=color, linestyle=style, linewidth=2,
                    label=f"{quantity}, {pol} in: RCWA")  # fmt: skip
            ax.plot(WAVENUMBERS[::2], theirs[::2], linestyle="none", marker="o", markersize=5,
                    markerfacecolor="none", color=color,
                    label=f"{quantity}, {pol} in: NumPy engine")  # fmt: skip
    _label(ax, "(a) unpatterned film: RCWA vs NumPy engine", "wavenumber (cm⁻¹)", "power")
    ax.text(0.02, 0.03, f"max |Δ| = {worst_film:.0e}", transform=ax.transAxes, fontsize=8,
            color=MUTED)  # fmt: skip
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))  # fmt: skip

    # (b) grating spectrum at 41 and 81 orders, p input
    ax = axes[0, 1]
    _style(ax)
    spectra = {n: [efficiencies(e, w, n) for e, w in zip(eps_all, wavelengths, strict=True)]
               for n in (20, 40)}  # fmt: skip
    for key, color, label in (("R0", ORANGE, "R₀"), ("T0", BLUE, "T₀"), ("T-1", AQUA, "T₋₁"),
                              ("A", YELLOW, "absorbed")):  # fmt: skip
        for n, style, width in ((40, "-", 2.0), (20, ":", 1.4)):
            values = [s[key][0] for s in spectra[n]]
            ax.plot(WAVENUMBERS, values, color=color, linestyle=style, linewidth=width,
                    label=f"{label}, {2 * n + 1} orders")  # fmt: skip
    _label(ax, "(b) tilted-hBN grating, p input", "wavenumber (cm⁻¹)", "efficiency")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # (c) convergence in Fourier order
    ax = axes[0, 2]
    _style(ax)
    counts = (10, 20, 40)
    for v, color in zip((1400.0, 1450.0, 1550.0), (BLUE, ORANGE, AQUA), strict=True):
        eps, w = tensor(v), 1e4 / v
        ref = efficiencies(eps, w, 80)
        errors = []
        for n in counts:
            e = efficiencies(eps, w, n)
            window = slice(80 - n, 80 + n + 1)
            errors.append(max(np.abs(e[q] - ref[q][window]).max() for q in ("R", "T")))
        ax.loglog([2 * n + 1 for n in counts], errors, color=color, marker="o", linewidth=2,
                  label=f"{v:.0f} cm⁻¹")  # fmt: skip
    ax.set_xticks([21, 41, 81], ["21", "41", "81"])
    ax.minorticks_off()
    _label(ax, "(c) change to the 161-order result", "Fourier orders kept",
           "max |Δ efficiency| over orders")  # fmt: skip
    ax.legend(frameon=False, fontsize=8)

    # (d, e) the first-order efficiency and its gradient against fill factor, 1450 cm⁻¹
    eps, w = tensor(1450.0), 1e4 / 1450.0
    fills = np.linspace(0.3, 0.5, 41)
    step, relative = 1e-5, []
    ax_value, ax_grad = axes[1, 0], axes[1, 1]
    for ax in (ax_value, ax_grad):
        _style(ax)
    for n, color in ((20, BLUE), (40, ORANGE), (80, AQUA)):

        def first_order(fill, n=n):
            res = _solve(eps, w, fill, n)
            return jnp.sum(res.T[n - 1, :, 0])  # order −1, p in, summed over out

        value, grad = jax.jit(first_order), jax.jit(jax.grad(first_order))
        values = np.array([float(value(f)) for f in fills])
        ad = np.array([float(grad(f)) for f in fills])
        fd = np.array(
            [(float(value(f + step)) - float(value(f - step))) / (2 * step) for f in fills]
        )
        relative.extend(np.abs(ad - fd) / np.abs(ad))
        ax_value.plot(fills, values, color=color, linewidth=2, label=f"{2 * n + 1} orders")
        ax_grad.plot(fills, ad, color=color, linewidth=2, label=f"jax.grad, {2 * n + 1} orders")
        ax_grad.plot(fills[::2], fd[::2], linestyle="none", marker="o", markersize=5,
                     markerfacecolor="none", color=color,
                     label=f"finite difference, {2 * n + 1} orders")  # fmt: skip
    _label(ax_value, "(d) T₋₁ vs fill, 1450 cm⁻¹: truncation ripple", "fill factor",
           "first-order efficiency, p in")  # fmt: skip
    ax_value.legend(frameon=False, fontsize=8)
    _label(ax_grad, "(e) d T₋₁ / d fill: jax.grad vs finite difference", "fill factor",
           "derivative")  # fmt: skip
    ax_grad.text(0.02, 0.03, f"median |AD − FD| / |AD| = {np.median(relative):.0e}",
                 transform=ax_grad.transAxes, fontsize=8, color=MUTED)  # fmt: skip
    ax_grad.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18))  # fmt: skip

    # (f) lossless variant: energy conservation vs order
    ax = axes[1, 2]
    _style(ax)
    for v, color in zip((1400.0, 1450.0, 1550.0), (BLUE, ORANGE, AQUA), strict=True):
        lossless = jnp.real(tensor(v)).astype(complex)
        errs = []
        for n in (10, 20, 40, 80):
            e = efficiencies(lossless, 1e4 / v, n)
            errs.append(float(np.max(np.abs(e["A"]))))
        ax.loglog([21, 41, 81, 161], errs, color=color, marker="o", linewidth=2,
                  label=f"{v:.0f} cm⁻¹")  # fmt: skip
    ax.set_xticks([21, 41, 81, 161], ["21", "41", "81", "161"])
    ax.minorticks_off()
    _label(ax, "(f) lossless hBN: |R + T − 1|", "Fourier orders kept", "energy error")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Tilted-axis hBN grating (every ε component nonzero, hyperbolic): RCWA checks",
                 fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    path = args.out / "hbn_grating.png"
    fig.savefig(path, dpi=160)
    print(f"uniform film vs NumPy engine: max |Δ| {worst_film:.1e}")
    print(f"gradient vs finite difference: median relative {np.median(relative):.1e}, "
          f"max {np.max(relative):.1e}")  # fmt: skip
    print(f"figure written to {path}")


if __name__ == "__main__":
    main()
