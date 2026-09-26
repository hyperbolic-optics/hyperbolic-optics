"""Liquid-crystal Bragg polarization grating of arXiv:2608.06185, Fig. 4 (transmissive).

A full-tensor case beyond FMMAX's reach: the director tilts out of the plane of
the layers, so the permittivity has xz and yz components, and it rotates along
both x and z. The layer is sliced in z (20 slices, as in the paper) and each
slice sampled in x.

    director  u = (cos Θ cos Φ, cos Θ sin Φ, sin Θ),   Φ(x, z) = π x / Λx + Ω z
    ε = n_o² I + (n_e² − n_o²) u uᵀ

with n_o = 1.525, n_e = 1.775, Λx = 400 nm, thickness 1 µm, Ω = 217°/µm, on a
substrate of index 1.7. The paper does not state the incident medium or the
incident polarization; this assumes normal incidence from air with circular
polarization, and reports the power diffracted into each first order.

Run:  uv run python benchmarks/paper_fig4/lc_polarization_grating.py
"""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import rcwa  # noqa: E402

N_O, N_E, N_SUB, PERIOD_UM = 1.525, 1.775, 1.7, 0.4
THICKNESS_UM, TWIST_DEG_PER_UM, SLICES, SAMPLES, MAX_ORDER = 1.0, 217.0, 20, 64, 8
CIRCULAR = {"RCP": np.array([1, -1j]) / np.sqrt(2), "LCP": np.array([1, 1j]) / np.sqrt(2)}


def grating(tilt_deg: float, slices: int = SLICES) -> list[rcwa.GratingLayer]:
    x = (jnp.arange(SAMPLES) + 0.5) / SAMPLES  # fractions of Λx
    tilt = jnp.radians(tilt_deg)
    dz = THICKNESS_UM / slices
    layers = []
    for k in range(slices):
        phi = jnp.pi * x + jnp.radians(TWIST_DEG_PER_UM) * (k + 0.5) * dz
        u = jnp.stack(
            [
                jnp.cos(tilt) * jnp.cos(phi),
                jnp.cos(tilt) * jnp.sin(phi),
                jnp.sin(tilt) * jnp.ones_like(phi),
            ],
            axis=-1,
        )
        eps = N_O**2 * jnp.eye(3) + (N_E**2 - N_O**2) * u[:, :, None] * u[:, None, :]
        layers.append(rcwa.sampled_layer(eps.astype(complex), dz))
    return layers


SOLVE = jax.jit(
    lambda layers, wavelength, angle: rcwa.solve_grating(
        layers, PERIOD_UM, wavelength, angle, 1.0, N_SUB**2, MAX_ORDER
    )
)


def first_orders(layers, wavelength_um: float, angle_deg: float = 0.0):
    """(+1, −1) transmitted efficiency for each circular input, incident from air."""
    angle = np.radians(angle_deg)
    res = SOLVE(layers, wavelength_um, angle)
    orders = np.asarray(res.orders)
    kx = np.sin(angle) + orders * wavelength_um / PERIOD_UM
    weight = np.real(np.sqrt(N_SUB**2 - kx**2 + 0j)) / np.cos(angle)  # flux ratio per |t|²
    out = {}
    for name, jones in CIRCULAR.items():
        power = np.sum(np.abs(np.asarray(res.t) @ jones) ** 2, axis=-1) * weight
        out[name] = (float(power[orders == 1][0]), float(power[orders == -1][0]))
    return out


def _style(ax):
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6b6a64")
    ax.tick_params(colors="#6b6a64", labelcolor="#1f1f1e", labelsize=8)


def plot(path: Path) -> None:
    ink, colors = "#1f1f1e", ("#2a78d6", "#eb6834", "#1baf7a")
    wavelengths = np.linspace(0.40, 0.70, 31)
    angles = np.linspace(-20.0, 20.0, 21)
    fig, (spectral, angular) = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for tilt, color in zip((0.0, 25.0, 50.0), colors, strict=True):
        layers = grating(tilt)
        by_wavelength = [first_orders(layers, w) for w in wavelengths]
        by_angle = [first_orders(layers, 0.53, a) for a in angles]
        for ax, x, rows in (
            (spectral, wavelengths * 1e3, by_wavelength),
            (angular, angles, by_angle),
        ):
            ax.plot(x, [r["RCP"][1] for r in rows], color=color, linewidth=2,
                    label=f"Θ = {tilt:.0f}°: RCP → −1")  # fmt: skip
            ax.plot(x, [r["LCP"][0] for r in rows], color=color, linewidth=1.4, linestyle=":",
                    label=f"Θ = {tilt:.0f}°: LCP → +1")  # fmt: skip
    for ax, xlabel, title in (
        (spectral, "wavelength (nm)", "normal incidence"),
        (angular, "angle of incidence (°)", "530 nm"),
    ):
        _style(ax)
        ax.set_xlabel(xlabel, fontsize=9, color=ink)
        ax.set_ylabel("first-order efficiency", fontsize=9, color=ink)
        ax.set_title(title, fontsize=9, color=ink, loc="left")
        ax.set_ylim(0, 1)
    spectral.legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle(
        "LC Bragg polarization grating (arXiv:2608.06185 Fig. 4, transmissive) — "
        "assumes incidence from air",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wavelengths = (0.45, 0.50, 0.53, 0.55, 0.60, 0.65)
    print("Transmitted first-order efficiency, normal incidence from air (+1 / −1)\n")
    print(f"{'Θ':>4}  " + "".join(f"{w * 1e3:>16.0f} nm" for w in wavelengths))
    for tilt in (0.0, 25.0, 50.0):
        layers = grating(tilt)
        rows = [first_orders(layers, w) for w in wavelengths]
        for name in CIRCULAR:
            cells = "".join(f"{r[name][0]:>9.3f} /{r[name][1]:>6.3f}" for r in rows)
            print(f"{tilt:>4.0f}° {name} {cells}")
    print("\nSlice convergence, Θ = 25°, 530 nm, RCP → −1:")
    for slices in (20, 40, 80):
        print(f"  {slices:>3} slices: {first_orders(grating(25.0, slices), 0.53)['RCP'][1]:.4f}")
    plot(args.out / "lc_grating.png")
    print(f"\nFigure written to {args.out / 'lc_grating.png'}")


if __name__ == "__main__":
    main()
