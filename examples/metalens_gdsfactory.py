#!/usr/bin/env python3
"""A geometric-phase metalens of TiO₂ nanofins: unit cell → lens → GDS → re-simulation.

A rectangular TiO₂ fin is form-birefringent. Turned by ``θ``, an ideal
half-wave fin converts circular light to the opposite handedness with the
Pancharatnam–Berry phase ``2θ``, so a lens is one fin turned site by site to
``θ = φ/2``. That ideal assumes the lattice turns with the fin. In a real
layout it does not: the fin turns in a fixed square lattice, and its
neighbours see it differently at every angle.

1. **Fin.** A scan of fin length and width (660 nm, 600 nm tall, 325 nm
   pitch, 2D RCWA), for the largest conversion at the worst angle (0° or 45°
   in the lattice).
2. **Angle library.** That fin turned through 48 angles in its lattice, each
   one RCWA solve: its real converted amplitude against angle, against the
   ideal ``e^{2iθ}``. Between about 25° and 65° it converts 0.68, not 0.95
   (converged to ±0.01 from 81 to 289 harmonics).
3. **Lens.** The hyperbolic phase for f = 30 µm over a 20 µm aperture (NA
   0.32). Each site takes the angle whose real converted field best matches
   the target, and the lens is propagated to the focus by the angular
   spectrum (local periodicity, not a full-wave simulation of the finite lens).
4. **GDS.** One fin cell, a turned reference per site; written, read back into
   per-site cells and re-simulated.

Needs the gdsfactory extra:  uv sync --extra gdsfactory
Run:  python examples/metalens_gdsfactory.py   (writes to examples/output/)
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import gdsfactory as gf  # noqa: E402
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
LATTICE = rcwa2d.Lattice.square(PITCH)
LEFT, RIGHT = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)


def fin(dims) -> hog.VerticalStack:
    """A TiO₂ fin ``(length, width)`` along x, on glass (light from the glass side)."""
    outline = (rcwa2d.rectangle((0.0, 0.0), dims),)
    layer = hog.StackLayer(HEIGHT, N_TIO2**2, background=1.0, polygons=outline, name="fin")
    return hog.VerticalStack((layer,), incident=N_GLASS**2, exit=1.0)


def library(cell_of, parameters, max_order=(4, 4)):
    """Jones matrices ``[P, 2, 2]`` of ``cell_of(p)`` for each ``p``."""
    return np.asarray(ms.cell_library(lambda p: hog.cell_layers(cell_of(p), WAVELENGTH),
                                      parameters, LATTICE, WAVELENGTH, N_GLASS**2, 1.0,
                                      max_order))  # fmt: skip


def converted(t):
    """The field converted to the opposite handedness, ``a†·t·p``."""
    return np.einsum("i,...ij,j->...", np.conj(RIGHT), t, LEFT)


def spot(sites, values):
    """Focal-plane cut, its FWHM, and the power within 1.5 FWHM of the peak."""
    coords, field = ms.sample_grid(sites, values, PITCH, pad=4)
    _, aperture = ms.sample_grid(sites, np.ones(len(sites)), PITCH, pad=4)
    intensity = np.abs(np.asarray(ms.focus(coords, coords, field, WAVELENGTH, FOCAL))) ** 2
    i, j = np.unravel_index(intensity.argmax(), intensity.shape)
    fine = np.linspace(coords[i] - 2, coords[i] + 2, 4001)
    cut = intensity[:, j]
    fwhm = np.ptp(fine[np.interp(fine, coords, cut) >= cut.max() / 2])
    x, y = np.meshgrid(coords - coords[i], coords - coords[j], indexing="ij")
    power = intensity[np.hypot(x, y) <= 1.5 * fwhm].sum()
    efficiency = power / (np.sum(np.abs(np.asarray(aperture)) ** 2) * N_GLASS)
    return coords - coords[i], cut, fwhm, efficiency


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

    # 1. fin: the best conversion at the worst of 0° and 45° in the lattice
    lengths, widths = np.linspace(0.20, 0.30, 6), np.linspace(0.07, 0.17, 6)
    dims = np.array([(length, width) for length in lengths for width in widths])
    t0 = time.perf_counter()
    at = {deg: np.abs(converted(library(lambda d, a=np.radians(deg): hog.turned(fin(d), a),
                                        dims, (3, 3)))) ** 2 / N_GLASS
          for deg in (0, 45)}  # fmt: skip
    worst = np.minimum(at[0], at[45]).reshape(len(lengths), len(widths))
    best = int(np.argmax(worst))
    fin_dims = tuple(float(v) for v in dims[best])
    print(f"Fin scan: {2 * len(dims)} RCWA solves in {time.perf_counter() - t0:.0f} s; "
          f"{fin_dims[0] * 1e3:.0f} × {fin_dims[1] * 1e3:.0f} nm converts "
          f"{at[0][best]:.3f} at 0° and {at[45][best]:.3f} at 45°")  # fmt: skip

    # 2. the angle library: the real turned fin, and the ideal rotation
    angles = np.linspace(0.0, np.pi, 48, endpoint=False)
    real = library(lambda a: hog.turned(fin(fin_dims), a), angles)  # [48, 2, 2]
    ideal = np.asarray(ms.geometric_phase_jones(real[0], angles))
    efficiencies = np.abs(converted(real)) ** 2 / N_GLASS
    print(
        f"  turned in its lattice it converts {efficiencies.min():.3f}–"
        f"{efficiencies.max():.3f}; the ideal rotator {efficiencies[0]:.3f} at every angle"
    )

    # 3. lens: each site's angle chosen from the real library
    sites = ms.square_sites(RADIUS, PITCH)
    phase = np.asarray(ms.lens_phase(sites[:, 0], sites[:, 1], FOCAL, WAVELENGTH))
    indices, realised = ms.choose_cells(phase, real, polarization=LEFT, analyzer=RIGHT)
    x_real, cut_real, fwhm, efficiency = spot(sites, realised)
    ideal_field = converted(real[0]) * np.exp(1j * phase)  # every fin the ideal rotator
    x_ideal, cut_ideal, _, efficiency_ideal = spot(sites, ideal_field)
    airy = 0.51 * WAVELENGTH / np.sin(np.arctan(RADIUS / FOCAL))
    print(f"\nLens: {len(sites)} fins, f = {FOCAL:.0f} µm; FWHM {fwhm * 1e3:.0f} nm "
          f"(diffraction limit {airy * 1e3:.0f} nm); focusing efficiency {efficiency:.2f}, "
          f"against {efficiency_ideal:.2f} if every fin were the ideal rotator")  # fmt: skip

    # 4. GDS: one fin cell, a turned reference per site; read back and re-simulated
    component = hog.metasurface_component(sites, [fin(fin_dims)] * len(sites), [(1, 0)],
                                          (LATTICE.a1, LATTICE.a2), angles[indices])  # fmt: skip
    gds = OUT / "metalens.gds"
    component.write_gds(gds)
    read_sites, read_cells, _ = hog.read_metasurface(gf.import_gds(gds))
    probe = np.arange(0, len(read_cells), len(read_cells) // 5)
    resolved = np.stack([library(lambda _, c=read_cells[k]: c, np.zeros(1))[0] for k in probe])
    nearest = [int(np.argmin(np.hypot(*(sites - read_sites[k]).T))) for k in probe]
    mismatch = np.abs(resolved - real[indices[nearest]]).max()
    print(f"\nWritten {gds.name}: {len(component.insts)} references to "
          f"{component.info['n_cells']} fin cell. Read back, {len(probe)} sites re-simulated "
          f"from the GDS match the library to {mismatch:.1e}")  # fmt: skip

    ink, blue, orange, muted = "#1f1f1e", "#2a78d6", "#eb6834", "#6b6a64"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    ax = axes[0]
    image = ax.pcolormesh(widths * 1e3, lengths * 1e3, worst, cmap="Blues", shading="nearest")
    ax.plot(fin_dims[1] * 1e3, fin_dims[0] * 1e3, "o", markerfacecolor="none",
            markeredgecolor=orange, markersize=10, markeredgewidth=2)  # fmt: skip
    fig.colorbar(image, ax=ax, label="conversion, worst of 0° and 45°")
    ax.set_xlabel("fin width (nm)", fontsize=9, color=ink)
    ax.set_ylabel("fin length (nm)", fontsize=9, color=ink)
    ax.set_title("(a) fin, 2D RCWA", fontsize=9, color=ink, loc="left")

    ax = axes[1]
    _style(ax)
    degrees = np.degrees(angles)
    ax.plot(degrees, efficiencies, color=blue, linewidth=2, label="conversion, in the lattice")
    ax.plot(degrees, np.abs(converted(ideal)) ** 2 / N_GLASS, color=muted, linewidth=1.5,
            linestyle="--", label="conversion, ideal rotator")  # fmt: skip
    ax.plot(degrees, np.angle(converted(real) / converted(ideal)), color=orange, linewidth=2,
            label="phase error (rad)")  # fmt: skip
    ax.set_xlabel("fin angle (°)", fontsize=9, color=ink)
    ax.set_title("(b) the fin turned in its lattice", fontsize=9, color=ink, loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    _style(ax)
    near_real, near_ideal = np.abs(x_real) < 3, np.abs(x_ideal) < 3
    ax.plot(x_ideal[near_ideal], cut_ideal[near_ideal] / cut_ideal.max(), color=muted,
            linewidth=1.5, linestyle="--", label="ideal rotators")  # fmt: skip
    ax.plot(x_real[near_real], cut_real[near_real] / cut_ideal.max(), color=blue, linewidth=2,
            label="designed from the library")  # fmt: skip
    ax.axvspan(-airy / 2, airy / 2, color="#e4e3dd", alpha=0.8, linewidth=0,
               label="diffraction-limited FWHM")  # fmt: skip
    ax.set_xlabel("x in the focal plane (µm)", fontsize=9, color=ink)
    ax.set_ylabel("intensity (ideal peak = 1)", fontsize=9, color=ink)
    ax.set_title(f"(c) focal spot at f = {FOCAL:.0f} µm", fontsize=9, color=ink, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(OUT / "metalens.png", dpi=160)
    print(f"\nFigure: {OUT / 'metalens.png'}")


if __name__ == "__main__":
    main()
