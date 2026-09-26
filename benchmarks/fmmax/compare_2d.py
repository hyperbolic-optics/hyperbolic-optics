"""Cross-check the crossed-grating RCWA (``rcwa2d``) against FMMAX.

Four unit cells, square lattice Λ = 0.6 µm, a 0.25 µm patterned layer in air
on ε = 2.1, λ = 1 µm:

- a silicon-like square pillar (ε = 12.25) and disc, 0.3 rad incidence;
- a birefringent disc (ε = 7.5 along an in-plane axis at 30° from x, 6.25
  across it) and a hyperbolic one (ε = 4 along the axis, −3 + 0.3i across),
  at 30° incidence, where only the crystal converts p to s.

FMMAX takes anisotropy only in the plane of the layers (no xz / yz terms), so
those are the cases compared; tilted axes are checked against the 1D solver in
tests/test_rcwa2d.py and against arXiv:2608.06185 in ../crossed_grating/.

Both codes run their normal-vector formulations (FMMAX: JONES_DIRECT, on a
1200² raster; here the exact polygon coefficients). The quantity is the
zeroth-order reflectance, per channel. Circular (FMMAX) and rectangular (here)
truncations differ, so the x axis is the number of harmonics.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import fmmax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import rcwa2d  # noqa: E402

OUT = Path(__file__).parent / "output"
PERIOD, HEIGHT, WAVELENGTH, EPS_EXIT = 0.6, 0.25, 1.0, 2.1
RASTER = 1200
FMMAX_TERMS = (200, 400, 800, 1600)
ORDERS = (3, 5, 8, 11)


def _in_plane(axis_eps, cross_eps, angle=np.pi / 6):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ np.diag([axis_eps, cross_eps, cross_eps]).astype(complex) @ rotation.T


SQUARE = ("square", 0.15)  # half-width
DISC = ("disc", 0.17)  # radius
CASES = {
    "silicon square pillar": (SQUARE, 12.25 * np.eye(3, dtype=complex), 0.3, (0, 0)),
    "silicon disc": (DISC, 12.25 * np.eye(3, dtype=complex), 0.3, (0, 0)),
    "birefringent disc": (DISC, _in_plane(7.5, 6.25), np.pi / 6, (1, 0)),
    "hyperbolic disc": (DISC, _in_plane(4.0, -3.0 + 0.3j), np.pi / 6, (1, 0)),
}
CHANNEL_NAMES = {(0, 0): "R_pp", (1, 0): "R_ps (p in, s out)"}


def _fmmax(geometry, eps, incidence, terms):
    lattice = fmmax.LatticeVectors(u=jnp.array([PERIOD, 0.0]), v=jnp.array([0.0, PERIOD]))
    x = (np.arange(RASTER) + 0.5) / RASTER * PERIOD
    xx, yy = np.meshgrid(x, x, indexing="ij")
    kind, size = geometry
    if kind == "square":
        mask = (np.abs(xx - PERIOD / 2) < size) & (np.abs(yy - PERIOD / 2) < size)
    else:
        mask = (xx - PERIOD / 2) ** 2 + (yy - PERIOD / 2) ** 2 < size**2
    expansion = fmmax.generate_expansion(
        lattice, approximate_num_terms=terms, truncation=fmmax.Truncation.CIRCULAR
    )
    wavelength = jnp.asarray(WAVELENGTH)
    k_in_plane = fmmax.plane_wave_in_plane_wavevector(
        wavelength, jnp.asarray(incidence), jnp.asarray(0.0), jnp.asarray(1.0)
    )
    formulation = fmmax.Formulation.JONES_DIRECT

    def ambient(value):
        return fmmax.eigensolve_isotropic_media(
            wavelength, k_in_plane, lattice, jnp.asarray([[value + 0j]]), expansion, formulation
        )

    def component(i, j):
        return jnp.asarray(np.where(mask, eps[i, j], 1.0 if i == j else 0.0) + 0j)

    one = jnp.ones((RASTER, RASTER), complex)
    zero = jnp.zeros((RASTER, RASTER), complex)
    layer = fmmax.eigensolve_general_anisotropic_media(
        wavelength, k_in_plane, lattice,
        component(0, 0), component(0, 1), component(1, 0), component(1, 1), component(2, 2),
        one, zero, zero, one, one, expansion, formulation,
    )  # fmt: skip
    first, last = ambient(1.0), ambient(EPS_EXIT)
    s_matrix = fmmax.stack_s_matrix(
        [first, layer, last], [jnp.asarray(0.0), jnp.asarray(HEIGHT), jnp.asarray(0.0)]
    )
    n = expansion.num_terms
    z = int(np.flatnonzero((np.asarray(expansion.basis_coefficients) == 0).all(axis=1))[0])
    reflectance = np.zeros((2, 2))
    blocks = (1, 0)  # at azimuth 0, the Hy block carries p and the Hx block s
    for j, block_in in enumerate(blocks):
        incident = jnp.zeros((2 * n, 1), complex).at[block_in * n + z, 0].set(1.0)
        flux_in, _ = fmmax.amplitude_poynting_flux(incident, jnp.zeros_like(incident), first)
        _, back = fmmax.amplitude_poynting_flux(
            jnp.zeros_like(incident), s_matrix.s21 @ incident, first
        )
        for i, block_out in enumerate(blocks):
            reflectance[i, j] = float(-back[block_out * n + z, 0] / jnp.sum(flux_in))
    return n, reflectance


_solve = jax.jit(rcwa2d.solve_crossed_grating, static_argnames=("max_order", "factorization"))


def _ours(geometry, eps, incidence, order, factorization):
    kind, size = geometry
    centre = (PERIOD / 2, PERIOD / 2)
    if kind == "square":
        vertices = rcwa2d.rectangle(centre, (2 * size, 2 * size))
    else:
        vertices = rcwa2d.ellipse(centre, (size, size))
    layer = rcwa2d.PatternedLayer(
        HEIGHT, jnp.eye(3, dtype=complex), (rcwa2d.Shape(vertices, jnp.asarray(eps)),)
    )
    result = _solve(
        [layer], (PERIOD, PERIOD), WAVELENGTH, incidence, 1.0, EPS_EXIT,
        max_order=(order, order), factorization=factorization,
    )  # fmt: skip
    z = int(np.flatnonzero((np.asarray(result.orders) == 0).all(axis=1))[0])
    energy = float(np.max(1 - np.asarray(result.R.sum((0, 1)) + result.T.sum((0, 1)))))
    return (2 * order + 1) ** 2, np.asarray(result.R)[z], energy


def main():
    OUT.mkdir(exist_ok=True)
    ink, muted, blue, orange = "#1f1f1e", "#6b6a64", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), constrained_layout=True)
    for ax, (name, (geometry, eps, incidence, channel)) in zip(axes, CASES.items(), strict=True):
        t0 = time.perf_counter()
        reference = [_fmmax(geometry, eps, incidence, terms) for terms in FMMAX_TERMS]
        runs = {
            f: [_ours(geometry, eps, incidence, m, f) for m in ORDERS]
            for f in ("normal", "laurent")
        }
        print(f"\n{name} ({time.perf_counter() - t0:.0f} s)")
        for n, r in reference:
            print(f"  FMMAX   {n:5d}  R_pp {r[0, 0]:.5f}  R_ss {r[1, 1]:.5f}  R_ps {r[1, 0]:.6f}")
        for f, rows in runs.items():
            for n, r, energy in rows:
                print(
                    f"  {f:7s} {n:5d}  R_pp {r[0, 0]:.5f}  R_ss {r[1, 1]:.5f}  "
                    f"R_ps {r[1, 0]:.6f}  1 − ΣR − ΣT {energy:.0e} (absorbed; 0 if lossless)"
                )

        ax.set_axisbelow(True)
        ax.grid(True, color="#e4e3dd", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelcolor=ink, labelsize=8)
        i, j = channel
        ax.plot([n for n, _ in reference], [r[i, j] for _, r in reference], color=muted,
                marker="s", markersize=5, linewidth=2, label="FMMAX (JONES_DIRECT)")  # fmt: skip
        for f, color in (("normal", blue), ("laurent", orange)):
            ax.plot([n for n, _, _ in runs[f]], [r[i, j] for _, r, _ in runs[f]], color=color,
                    marker="o", markersize=5, linewidth=2, label=f"rcwa2d, {f}")  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(name, fontsize=9, color=ink, loc="left")
        ax.set_xlabel("harmonics", fontsize=9, color=ink)
        ax.set_ylabel(CHANNEL_NAMES[channel], fontsize=9, color=ink)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Zeroth-order reflectance against truncation: rcwa2d and FMMAX "
        "(Λ = 0.6 µm, h = 0.25 µm, λ = 1 µm)",
        fontsize=10, color=ink, x=0.01, ha="left",
    )  # fmt: skip
    fig.savefig(OUT / "convergence_2d.png", dpi=160)
    print(f"\nFigure: {OUT / 'convergence_2d.png'}")


if __name__ == "__main__":
    main()
