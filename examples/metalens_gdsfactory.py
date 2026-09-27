#!/usr/bin/env python3
"""A geometric-phase metalens of TiO₂ nanofins: unit cell → lens → GDS.

A rectangular TiO₂ fin on glass is form-birefringent: its transmission
Jones matrix is ``diag(t_long, t_short)`` along its sides. Turned by ``θ``, it
converts circularly polarized light to the opposite handedness with the
Pancharatnam–Berry phase ``2θ``, whatever the wavelength, with amplitude
``(t_long − t_short)/2``. A lens is then one fin, turned site by site to
``θ = φ(x, y)/2``:

1. **Unit cell.** A scan of fin length and width, each point one 2D RCWA
   solve of the periodic array (660 nm, 600 nm tall, 325 nm pitch), for the
   largest conversion efficiency (the length capped at 300 nm, to leave a
   25 nm gap between neighbours). The fin's Jones matrix is the anisotropy
   the whole design rests on.
2. **Lens.** The hyperbolic phase for f = 30 µm over a 20 µm-diameter
   aperture (NA 0.32), realised by rotation. The converted field of each
   site comes from its turned Jones matrix, co-polarized leak and all,
   projected on the opposite handedness.
3. **Focus.** The designed transmission propagated by the angular spectrum:
   the focal spot against the diffraction limit, the on-axis maximum
   against the focal shift of a finite Fresnel number, and the focusing
   efficiency. This is local periodicity (each fin transmits as it would in
   an array of identical fins), not a full-wave simulation of the finite lens.
4. **GDS.** One fin cell, one reference to it per site, turned; written to
   GDS and read back.

Needs the gdsfactory extra:  uv sync --extra gdsfactory
Run:  python examples/metalens_gdsfactory.py   (writes to examples/output/)
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

from hyperbolic_optics import metasurfaces as ms  # noqa: E402
from hyperbolic_optics import rcwa2d  # noqa: E402
from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

OUT = Path(__file__).parent / "output"
WAVELENGTH, PITCH, HEIGHT = 0.66, 0.325, 0.6
N_TIO2, N_GLASS = 2.4, 1.46
RADIUS, FOCAL = 10.0, 30.0
LEFT, RIGHT = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)


def fin(dims):
    """A fin ``(length, width)`` along x, centred in its cell, in air on glass."""
    eye = jnp.eye(3, dtype=complex)
    shape = rcwa2d.Shape(rcwa2d.rectangle((PITCH / 2, PITCH / 2), dims), N_TIO2**2 * eye)
    return [rcwa2d.PatternedLayer(HEIGHT, eye, (shape,))]


def conversion(t):
    """Transmitted power converted to the opposite handedness, ``|(txx − tyy)/2|²·n_out/n_in``."""
    return np.abs((t[..., 0, 0] - t[..., 1, 1]) / 2) ** 2 / N_GLASS


def pixelated(sites, values, sub=3):
    """The metasurface's transmission on a padded grid, each cell ``sub × sub`` pixels."""
    n = int(np.max(np.abs(np.round(sites / PITCH))))
    size = (2 * n + 1) * sub
    pad = 4 * size
    grid = np.zeros((pad, pad), complex)
    off = (pad - size) // 2
    for (x, y), value in zip(sites, values):
        i, j = int(round(x / PITCH)) + n, int(round(y / PITCH)) + n
        grid[off + i * sub : off + (i + 1) * sub, off + j * sub : off + (j + 1) * sub] = value
    coords = (np.arange(pad) - (off + n * sub + (sub - 1) / 2)) * PITCH / sub
    return coords, grid


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
    lattice = rcwa2d.Lattice.square(PITCH)

    # 1. unit cell
    lengths, widths = np.linspace(0.20, 0.30, 6), np.linspace(0.07, 0.17, 6)
    dims = [(length, width) for length in lengths for width in widths]
    t0 = time.perf_counter()
    scan = ms.cell_library(fin, dims, lattice, WAVELENGTH, N_GLASS**2, 1.0, (3, 3))
    efficiency = conversion(scan).reshape(len(lengths), len(widths))
    best = int(np.argmax(efficiency))
    fin_dims = dims[best]
    jones = ms.cell_library(fin, [fin_dims], lattice, WAVELENGTH, N_GLASS**2, 1.0, (5, 5))[0]
    print(f"Unit-cell scan: {len(dims)} RCWA solves in {time.perf_counter() - t0:.0f} s")
    print(f"  best fin {fin_dims[0] * 1e3:.0f} × {fin_dims[1] * 1e3:.0f} nm: conversion "
          f"{efficiency.flat[best]:.3f} (3×3 orders), {conversion(jones):.3f} (5×5)")  # fmt: skip

    # 2. lens: each fin turned to half the target phase
    sites = ms.square_sites(RADIUS, PITCH)
    phase = ms.lens_phase(sites[:, 0], sites[:, 1], FOCAL, WAVELENGTH)
    rotations = np.mod(phase / 2, np.pi)
    turned = ms.geometric_phase_jones(jones, rotations)  # [K, 2, 2]
    converted = np.einsum("i,kij,j->k", np.conj(RIGHT), turned, LEFT)
    error = np.angle(converted * np.exp(-1j * phase) / ((jones[0, 0] - jones[1, 1]) / 2))
    print(f"\nLens: {len(sites)} fins, f = {FOCAL:.0f} µm, "
          f"NA {np.sin(np.arctan(RADIUS / FOCAL)):.2f}; "
          f"rms phase error {np.std(error):.1e} rad")  # fmt: skip

    # 3. focus
    coords, field = pixelated(sites, converted)
    _, incident = pixelated(sites, np.ones(len(sites)))
    depths = np.linspace(FOCAL - 12, FOCAL + 12, 97)
    axial = [np.abs(ms.focus(coords, coords, field, WAVELENGTH, z))[len(coords) // 2].max() ** 2
             for z in depths]  # fmt: skip
    focal = np.abs(ms.focus(coords, coords, field, WAVELENGTH, FOCAL)) ** 2
    centre = np.unravel_index(focal.argmax(), focal.shape)
    cut = focal[:, centre[1]]
    fine = np.linspace(coords[centre[0]] - 2, coords[centre[0]] + 2, 4001)
    fwhm = np.ptp(fine[np.interp(fine, coords, cut) >= cut.max() / 2])
    airy = 0.51 * WAVELENGTH / np.sin(np.arctan(RADIUS / FOCAL))
    x, y = np.meshgrid(coords, coords, indexing="ij")
    spot = np.hypot(x - coords[centre[0]], y - coords[centre[1]]) <= 1.5 * fwhm
    focusing = (focal[spot].sum() * 1.0) / (np.sum(np.abs(incident) ** 2) * N_GLASS)
    # the on-axis maximum of a lens of Fresnel number N sits short of f (Li & Wolf)
    fresnel = RADIUS**2 / (WAVELENGTH * FOCAL)
    shifted = FOCAL / (1 + 12 / (np.pi * fresnel) ** 2)
    print(
        f"  on-axis maximum at {depths[int(np.argmax(axial))]:.1f} µm "
        f"(Li–Wolf focal shift, Fresnel number {fresnel:.1f}: {shifted:.1f} µm)"
    )
    print(f"  in the plane z = f: FWHM {fwhm * 1e3:.0f} nm "
          f"(diffraction limit {airy * 1e3:.0f} nm); focusing efficiency {focusing:.2f} "
          f"(of the light incident on the aperture, in 3 FWHM)")  # fmt: skip

    # 4. GDS
    polygon = np.asarray(rcwa2d.rectangle((0.0, 0.0), fin_dims))
    component = hog.metasurface_component(sites, np.zeros(len(sites), int), [[polygon]],
                                          layer=(1, 0), rotations_rad=rotations)  # fmt: skip
    gds = OUT / "metalens.gds"
    component.write_gds(gds)
    read = gf.import_gds(gds)
    read.flatten()
    polygons = next(iter(read.get_polygons_points().values()))
    print(f"\nWritten {gds.name}: {len(component.insts)} references to one fin cell; "
          f"read back {len(polygons)} fins")  # fmt: skip

    ink, blue, orange, muted = "#1f1f1e", "#2a78d6", "#eb6834", "#6b6a64"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    ax = axes[0]
    image = ax.pcolormesh(widths * 1e3, lengths * 1e3, efficiency, cmap="Blues", shading="nearest")
    ax.plot(fin_dims[1] * 1e3, fin_dims[0] * 1e3, "o", markerfacecolor="none",
            markeredgecolor=orange, markersize=10, markeredgewidth=2)  # fmt: skip
    fig.colorbar(image, ax=ax, label="circular conversion efficiency")
    ax.set_xlabel("fin width (nm)", fontsize=9, color=ink)
    ax.set_ylabel("fin length (nm)", fontsize=9, color=ink)
    ax.set_title("(a) unit cell, 2D RCWA", fontsize=9, color=ink, loc="left")

    ax = axes[1]
    _style(ax)
    ax.plot(depths, np.asarray(axial) / max(axial), color=blue, linewidth=2)
    ax.axvline(FOCAL, color=muted, linewidth=1, linestyle=":", label="design f")
    ax.axvline(shifted, color=orange, linewidth=1.5, linestyle="--", label="Li–Wolf shift")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("distance from the metasurface (µm)", fontsize=9, color=ink)
    ax.set_ylabel("on-axis intensity (normalised)", fontsize=9, color=ink)
    ax.set_title(f"(b) along the axis, design f = {FOCAL:.0f} µm", fontsize=9, color=ink,
                 loc="left")  # fmt: skip

    ax = axes[2]
    _style(ax)
    near = np.abs(coords - coords[centre[0]]) < 3
    ax.plot(coords[near] - coords[centre[0]], cut[near] / cut.max(), color=blue, linewidth=2,
            label="designed lens")  # fmt: skip
    ax.axvspan(-airy / 2, airy / 2, color="#e4e3dd", alpha=0.8, linewidth=0,
               label="diffraction-limited FWHM")  # fmt: skip
    ax.set_xlabel("x in the focal plane (µm)", fontsize=9, color=ink)
    ax.set_ylabel("intensity (normalised)", fontsize=9, color=ink)
    ax.set_title(f"(c) focal spot, FWHM {fwhm * 1e3:.0f} nm", fontsize=9, color=ink, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(OUT / "metalens.png", dpi=160)
    print(f"\nFigure: {OUT / 'metalens.png'}")


if __name__ == "__main__":
    main()
