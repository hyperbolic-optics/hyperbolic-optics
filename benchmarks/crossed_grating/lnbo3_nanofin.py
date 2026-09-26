"""A LiNbO₃ nanofin metasurface with the optic axis tilted out of plane.

The nanofin array of arXiv:2608.06185 (Sec. 3A): LiNbO₃ fins, εₒ = 5.46,
εₑ = 5.06, optic axis tilted ξ = 45° from the layer normal in the xz plane, so
ε has an xz component, which FMMAX and most RCWA codes cannot represent. Fins
0.2 × 0.1 µm, 0.3 µm tall, on a 0.34 µm square lattice, on SiO₂ (n = 1.46),
normal incidence from air at 550 nm.

Two panels:

(a) T_xx and T_yy at 550 nm against the number of harmonics, for the long side
    of the fin along x and along y, with both factorisations, against the
    paper's values for the long side along x;
(b) T_xx and T_yy against the tilt ξ, at 225 harmonics: what the
    out-of-plane component does. The mirror y → −y maps both the fin and an
    axis in the xz plane to themselves, so x → y conversion must vanish at
    every ξ; the script prints it as a check.

    uv run python benchmarks/crossed_grating/lnbo3_nanofin.py
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import rcwa2d  # noqa: E402

OUT = Path(__file__).parent / "output"
EPS_O, EPS_E = 5.46, 5.06
PERIOD, HEIGHT, WAVELENGTH, EPS_SUBSTRATE = 0.34, 0.30, 0.55, 1.46**2
PAPER_T_XX = {"BCM (paper)": 0.9496, "FEM (paper)": 0.9483}  # long side along x, 550 nm
ORDERS = (3, 4, 5, 6, 7, 8)


def lnbo3(tilt_rad):
    """The optic axis in the xz plane, ``tilt_rad`` from z."""
    axis = jnp.array([jnp.sin(tilt_rad), 0.0, jnp.cos(tilt_rad)])
    return (EPS_O * jnp.eye(3) + (EPS_E - EPS_O) * jnp.outer(axis, axis)).astype(complex)


_solve = jax.jit(rcwa2d.solve_crossed_grating, static_argnames=("max_order", "factorization"))


def transmission(footprint, tilt_rad, order, factorization="normal"):
    """Zeroth-order T ``[out, in]`` in x/y (p = x, s = y at normal incidence)."""
    centre = (PERIOD / 2, PERIOD / 2)
    fin = rcwa2d.Shape(rcwa2d.rectangle(centre, footprint), lnbo3(tilt_rad))
    layer = rcwa2d.PatternedLayer(HEIGHT, jnp.eye(3, dtype=complex), (fin,))
    result = _solve(
        [layer], (PERIOD, PERIOD), WAVELENGTH, 0.0, 1.0, EPS_SUBSTRATE,
        max_order=(order, order), factorization=factorization,
    )  # fmt: skip
    z = int(np.flatnonzero((np.asarray(result.orders) == 0).all(axis=1))[0])
    return np.asarray(result.T)[z]


def _style(ax, ink, muted):
    ax.set_axisbelow(True)
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=muted, labelcolor=ink, labelsize=8)


def main():
    OUT.mkdir(exist_ok=True)
    ink, muted, blue, orange = "#1f1f1e", "#6b6a64", "#2a78d6", "#eb6834"
    harmonics = [(2 * m + 1) ** 2 for m in ORDERS]
    along_x, along_y = (0.2, 0.1), (0.1, 0.2)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    _style(ax_a, ink, muted)
    tilt = np.radians(45.0)
    for factorization, color in (("normal", blue), ("laurent", orange)):
        t_x = [transmission(along_x, tilt, m, factorization)[0, 0] for m in ORDERS]
        t_y = [transmission(along_y, tilt, m, factorization)[1, 1] for m in ORDERS]
        print(factorization, "long side along x, T_xx:", np.round(t_x, 4))
        print(factorization, "long side along y, T_yy:", np.round(t_y, 4))
        ax_a.plot(harmonics, t_x, color=color, marker="o", markersize=5, linewidth=2,
                  label=f"{factorization}: long side ∥ x, T_xx")  # fmt: skip
        ax_a.plot(harmonics, t_y, color=color, marker="o", markersize=5, linewidth=2,
                  linestyle="--", label=f"{factorization}: long side ∥ y, T_yy")  # fmt: skip
    for (label, value), style in zip(PAPER_T_XX.items(), (":", "-."), strict=True):
        ax_a.axhline(value, color=muted, linestyle=style, linewidth=1.2, label=label)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("harmonics", fontsize=9, color=ink)
    ax_a.set_ylabel("zeroth-order transmittance", fontsize=9, color=ink)
    ax_a.set_title("(a) convergence at 550 nm, ξ = 45°", fontsize=9, color=ink, loc="left")
    ax_a.legend(
        frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16)
    )

    _style(ax_b, ink, muted)
    tilts = np.linspace(0.0, 90.0, 19)
    rows = np.array([transmission(along_x, np.radians(t), 7) for t in tilts])
    ax_b.plot(tilts, rows[:, 0, 0], color=blue, linewidth=2, label="T_xx")
    ax_b.plot(tilts, rows[:, 1, 1], color=orange, linewidth=2, label="T_yy")
    ax_b.set_xlabel("tilt ξ of the optic axis from the normal (°)", fontsize=9, color=ink)
    ax_b.set_ylabel("zeroth-order transmittance", fontsize=9, color=ink)
    ax_b.set_title("(b) long side ∥ x, 225 harmonics: the out-of-plane axis",
                   fontsize=9, color=ink, loc="left")  # fmt: skip
    ax_b.legend(frameon=False, fontsize=8, loc="center left")
    print("x → y conversion, largest over ξ (zero by symmetry):", float(rows[:, 1, 0].max()))

    fig.suptitle(
        "LiNbO₃ nanofins (arXiv:2608.06185 Sec. 3A): 0.2 × 0.1 × 0.3 µm on a 0.34 µm lattice, "
        "SiO₂ substrate",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "lnbo3_nanofin.png", dpi=160)
    print(f"Figure: {OUT / 'lnbo3_nanofin.png'}")


if __name__ == "__main__":
    main()
