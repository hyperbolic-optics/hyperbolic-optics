"""Cross-check the 1D-grating RCWA against FMMAX, an independent JAX RCWA code.

FMMAX (https://github.com/mfschubert/fmmax, MIT) is not a dependency of this
package; README.md alongside gives the command that runs this with it.

FMMAX handles isotropic media and anisotropy confined to the plane of the
layers (xx, xy, yx, yy, zz components) -- including in-plane hyperbolic and
gyrotropic (non-symmetric) tensors -- so those are the cases compared here.
Full tensors with xz / yz components are validated in tests/test_rcwa.py and
benchmarks/hyperbolic_grating/ instead.

What is compared is the diffraction efficiency of every order, summed over
outgoing polarization, which does not depend on either code's polarization
basis. FMMAX's uniform-layer amplitudes are the tangential H components (Hx
block first -- checked against Fresnel below); the same incident states are
formed in hyperbolic-optics from its p and s waves.

The two codes represent the geometry differently (exact Fourier coefficients of
the step profile here, an FFT of a sampled grid in FMMAX) and, for anisotropic
media, factorise the Fourier products differently. They agree to the level at
which both have converged, so each code's own convergence is reported. For the
anisotropic cases -- where the factorisations, and so the truncated answers,
differ -- FMMAX is also extrapolated to infinitely many orders (Richardson,
assuming its error falls as 1/N). The isotropic cases need no extrapolation:
both codes use the same factorisation there and agree directly.
"""

from __future__ import annotations

import argparse
import sys
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

from hyperbolic_optics import rcwa  # noqa: E402
from hyperbolic_optics.materials import create_material  # noqa: E402
from hyperbolic_optics.rcwa import _harmonic_modes  # noqa: E402

GRID = 2000  # FMMAX samples per period; the fills below are multiples of 1/GRID
EYE = np.eye(3, dtype=complex)
TRUNCATIONS = (10, 20, 40)  # max_order; the reference is 80 (161 orders)
REFERENCE = 80
BLUE, ORANGE, AQUA, INK, MUTED, GRID_INK = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#1f1f1e",
    "#6b6a64",
    "#e4e3dd",
)


def rotate_z(tensor, angle):
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return r @ tensor @ r.T


def uniaxial_in_plane(eps_o, eps_e, angle):
    """Optic axis in the plane of the layers at ``angle`` from x (no xz / yz terms)."""
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    return eps_o * EYE + (eps_e - eps_o) * np.outer(axis, axis)


MOO3_900 = np.asarray(create_material("MoO3").fetch_permittivity_tensor_for_freq(900.0))

CASES = {
    "silicon, 20°": dict(ridge=3.5**2 * EYE, fill=0.4, thickness=0.35, period=0.9, theta=20.0),
    "silicon, conical": dict(
        ridge=3.5**2 * EYE, fill=0.4, thickness=0.35, period=0.9, theta=30.0, azimuth=40.0
    ),
    "gold (ε = −25 + 1.5i)": dict(
        ridge=(-25.0 + 1.5j) * EYE, fill=0.5, thickness=0.1, period=0.6, theta=15.0
    ),
    "LiNbO₃, axis in-plane, conical": dict(
        ridge=uniaxial_in_plane(5.46, 5.06, np.radians(30.0)),
        fill=0.45,
        thickness=0.5,
        period=0.8,
        theta=25.0,
        azimuth=35.0,
    ),
    "α-MoO₃ 900 cm⁻¹, in-plane hyperbolic": dict(
        ridge=rotate_z(MOO3_900.reshape(3, 3), np.radians(30.0)),
        fill=0.5,
        thickness=1.0,
        period=15.0,
        theta=20.0,
        azimuth=30.0,
        wavelength=1e4 / 900.0,
        eps_out=2.0,
    ),
    "gyrotropic (ε_xy = −ε_yx = 0.8i)": dict(
        ridge=np.array([[4.0, 0.8j, 0], [-0.8j, 4.0, 0], [0, 0, 3.0]]),
        fill=0.45,
        thickness=0.4,
        period=0.9,
        theta=25.0,
        azimuth=35.0,
    ),
}
DEFAULTS = dict(azimuth=0.0, wavelength=1.0, eps_in=1.0, eps_out=1.45**2)


# ---------------------------------------------------------------------------
# hyperbolic-optics
# ---------------------------------------------------------------------------


def ours(case, max_order):
    """Per-order efficiency ``[state (Hx, Hy incident), order]`` for R and T."""
    c = {**DEFAULTS, **case}
    layer = rcwa.lamellar_layer(
        jnp.asarray(c["ridge"]), jnp.asarray(EYE), c["fill"], c["thickness"]
    )
    theta, azimuth = np.radians(c["theta"]), np.radians(c["azimuth"])
    result = rcwa.solve_grating(
        [layer],
        c["period"],
        c["wavelength"],
        theta,
        c["eps_in"],
        c["eps_out"],
        max_order,
        azimuth_rad=azimuth,
    )

    # Incident (a_p, a_s) giving tangential H = x̂ or ŷ in the zeroth order.
    k_par = np.sqrt(c["eps_in"]) * np.sin(theta)
    qx, qy = jnp.array([k_par * np.cos(azimuth)]), k_par * np.sin(azimuth)
    w, _ = _harmonic_modes(c["eps_in"], qx, qy, (jnp.array([1.0]), jnp.array([0.0])))
    jones = np.linalg.solve(np.asarray(w)[2:, :2], np.eye(2))  # columns: Hx-only, Hy-only

    orders = np.asarray(result.orders)
    kx = k_par * np.cos(azimuth) + orders * c["wavelength"] / c["period"]
    kz_in0 = np.real(np.sqrt(c["eps_in"] - k_par**2 + 0j))
    out = {}
    for name, amp, eps in (("R", result.r, c["eps_in"]), ("T", result.t, c["eps_out"])):
        weight = np.real(np.sqrt(eps - kx**2 - qy**2 + 0j)) / kz_in0  # [order]
        combined = np.einsum("oji,ik->koj", np.asarray(amp), jones)  # [state, order, out]
        norm = np.sum(np.abs(jones) ** 2, axis=0)[:, None]
        out[name] = np.sum(np.abs(combined) ** 2, axis=-1) * weight / norm
    return out


# ---------------------------------------------------------------------------
# FMMAX
# ---------------------------------------------------------------------------


def theirs(case, max_order, formulation=fmmax.Formulation.NORMAL):
    c = {**DEFAULTS, **case}
    lattice = fmmax.LatticeVectors(u=jnp.array([c["period"], 0.0]), v=jnp.array([0.0, 1.0]))
    expansion = fmmax.Expansion(
        basis_coefficients=np.array([[m, 0] for m in range(-max_order, max_order + 1)])
    )
    wavelength = jnp.asarray(c["wavelength"])
    wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength,
        jnp.asarray(np.radians(c["theta"])),
        jnp.asarray(np.radians(c["azimuth"])),
        jnp.asarray(c["eps_in"]),
    )

    def ambient(eps):
        return fmmax.eigensolve_isotropic_media(
            wavelength, wavevector, lattice, jnp.asarray([[eps + 0j]]), expansion, formulation
        )

    inside = (np.arange(GRID) < int(round(c["fill"] * GRID)))[:, None]  # [GRID, 1]

    def grid(i, j):
        return jnp.asarray(np.where(inside, c["ridge"][i, j], EYE[i, j]))

    one, zero = jnp.ones((GRID, 1), dtype=complex), jnp.zeros((GRID, 1), dtype=complex)
    grating = fmmax.eigensolve_general_anisotropic_media(
        wavelength, wavevector, lattice,
        grid(0, 0), grid(0, 1), grid(1, 0), grid(1, 1), grid(2, 2),
        one, zero, zero, one, one,
        expansion=expansion, formulation=formulation,
    )  # fmt: skip
    first, last = ambient(c["eps_in"]), ambient(c["eps_out"])
    thicknesses = [jnp.asarray(0.0), jnp.asarray(c["thickness"]), jnp.asarray(0.0)]
    s = fmmax.stack_s_matrix([first, grating, last], thicknesses)

    size = 2 * max_order + 1
    out = {"R": [], "T": []}
    for half in (0, 1):  # Hx block, then Hy block
        incident = jnp.zeros((2 * size, 1), dtype=complex).at[half * size + max_order, 0].set(1.0)
        flux_in, _ = fmmax.amplitude_poynting_flux(incident, jnp.zeros_like(incident), first)
        _, back = fmmax.amplitude_poynting_flux(jnp.zeros_like(incident), s.s21 @ incident, first)
        fwd, _ = fmmax.amplitude_poynting_flux(s.s11 @ incident, jnp.zeros_like(incident), last)
        power = float(jnp.sum(flux_in))
        out["R"].append(-np.asarray(back[:size, 0] + back[size:, 0]) / power)
        out["T"].append(np.asarray(fwd[:size, 0] + fwd[size:, 0]) / power)
    return {k: np.array(v) for k, v in out.items()}


def fresnel_check():
    """FMMAX's first block is s-like (Hx) at planar incidence: R matches Fresnel r_s."""
    case = dict(ridge=EYE, fill=0.5, thickness=0.0, period=0.8, theta=30.0)
    res = theirs(case, 1, fmmax.Formulation.FFT)
    c1, c2 = np.cos(np.radians(30.0)), np.sqrt(1 - (np.sin(np.radians(30.0)) / 1.45) ** 2)
    rs = ((c1 - 1.45 * c2) / (c1 + 1.45 * c2)) ** 2
    rp = ((1.45 * c1 - c2) / (1.45 * c1 + c2)) ** 2
    return abs(res["R"][0, 1] - rs), abs(res["R"][1, 1] - rp)


def central(result, max_order, keep):
    """The orders ``−keep … keep`` of a result truncated at ``max_order``."""
    window = slice(max_order - keep, max_order + keep + 1)
    return {q: v[:, window] for q, v in result.items()}


def max_difference(a, b):
    return max(float(np.max(np.abs(a[q] - b[q]))) for q in "RT")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax):
    ax.grid(True, color=GRID_INK, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def plot_convergence(records, path):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, constrained_layout=True)
    counts = [2 * n + 1 for n in TRUNCATIONS]
    for ax, (name, rec) in zip(axes.flat, records.items(), strict=True):
        _style(ax)
        ax.loglog(counts, rec["fmmax_self"], color=ORANGE, marker="o", linewidth=3.5,
                  markersize=8, label="FMMAX: change to its 161-order result")  # fmt: skip
        ax.loglog(counts, rec["ours_self"], color=BLUE, marker="o", linewidth=1.5, markersize=4,
                  label="hyperbolic-optics: change to its 161-order result")  # fmt: skip
        ax.loglog(counts + [2 * REFERENCE + 1], rec["cross"], color=AQUA, marker="s",
                  linestyle="--", linewidth=2, label="hyperbolic-optics vs FMMAX")  # fmt: skip
        if rec["extrapolated"] is not None:
            label = "vs FMMAX extrapolated to ∞ orders"
            ax.loglog([2 * REFERENCE + 1], [rec["extrapolated"]], linestyle="none", marker="*",
                      markersize=12, color=INK, label=label)  # fmt: skip
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xticks([21, 41, 81, 161], ["21", "41", "81", "161"])
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("Fourier orders kept", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("max |Δ efficiency| over orders", fontsize=9, color=INK)
    handles, labels = axes[1, 0].get_legend_handles_labels()  # an anisotropic panel
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("1D-grating RCWA vs FMMAX: agreement and convergence",
                 fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.savefig(path, dpi=160)
    plt.close(fig)


SPECTRUM_ORDER = 20  # 41 orders in both codes


def moo3_at(wavenumber):
    tensor = np.asarray(create_material("MoO3").fetch_permittivity_tensor_for_freq(wavenumber))
    return rotate_z(tensor.reshape(3, 3), np.radians(30.0))


def spectrum_axis(name):
    """(x values, x label, per-point case overrides) for a case's sweep."""
    if name.startswith("α-MoO₃"):
        wavenumbers = np.linspace(850.0, 960.0, 31)
        overrides = [dict(wavelength=1e4 / v, ridge=moo3_at(v)) for v in wavenumbers]
        return wavenumbers, "wavenumber (cm⁻¹), dispersive MoO₃", overrides
    base = {**DEFAULTS, **CASES[name]}["wavelength"]
    wavelengths = np.linspace(0.8 * base, 1.25 * base, 31)
    return wavelengths, "wavelength (µm)", [dict(wavelength=w) for w in wavelengths]


def spectra():
    """Zeroth-order T, R and the −1 transmitted order vs wavelength, Hy incident."""
    out = {}
    centre = SPECTRUM_ORDER
    for name, case in CASES.items():
        x, label, overrides = spectrum_axis(name)
        curves = {"ours": [], "fmmax": []}
        for override in overrides:
            point = {**case, **override}
            for key, function in (("ours", ours), ("fmmax", theirs)):
                eff = function(point, SPECTRUM_ORDER)
                curves[key].append(
                    [eff["T"][1, centre], eff["R"][1, centre], eff["T"][1, centre - 1]]
                )
        out[name] = (x, label, {k: np.array(v) for k, v in curves.items()})
    return out


def plot_spectra(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    series = (("T₀ (zeroth order, transmitted)", BLUE), ("R₀ (zeroth order, reflected)", ORANGE),
              ("T₋₁ (first diffracted order)", AQUA))  # fmt: skip
    worst = {}
    for ax, (name, (x, label, curves)) in zip(axes.flat, data.items(), strict=True):
        _style(ax)
        for index, (series_label, color) in enumerate(series):
            ax.plot(x, curves["ours"][:, index], color=color, linewidth=2,
                    label=f"{series_label}: hyperbolic-optics")  # fmt: skip
            ax.plot(x, curves["fmmax"][:, index], linestyle="none", marker="o", markersize=5,
                    markerfacecolor="none", markeredgewidth=1.1, color=color,
                    label=f"{series_label}: FMMAX")  # fmt: skip
        worst[name] = float(np.max(np.abs(curves["ours"] - curves["fmmax"])))
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(label, fontsize=9, color=INK)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("efficiency (Hy incident)", fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Diffraction spectra at 41 orders: lines hyperbolic-optics, circles FMMAX",
                 fontsize=11, color=INK, x=0.01, ha="left")  # fmt: skip
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return worst


def _time(function, case, repeats=5):
    function(case, 20)  # warm up
    start = time.perf_counter()
    for _ in range(repeats):
        function(case, 20)
    return (time.perf_counter() - start) / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"fmmax {fmmax.__version__}, jax {jax.__version__}, grid {GRID} samples/period\n")
    ds, dp = fresnel_check()
    print(f"FMMAX basis check vs Fresnel (Hx→s, Hy→p): |ΔRs| = {ds:.1e}, |ΔRp| = {dp:.1e}\n")

    header = (
        f"{'case':<38}{'orders':>7}{'vs FMMAX':>11}{'ho Δ→161':>11}{'FMMAX Δ→161':>13}"
        f"{'ho |R+T−1|':>12}"
    )
    print(header)
    print("-" * len(header))
    records, worst = {}, 0.0
    for name, case in CASES.items():
        ours_ref, fmmax_ref = ours(case, REFERENCE), theirs(case, REFERENCE)
        ridge = case["ridge"]
        anisotropic = not np.allclose(ridge, ridge[0, 0] * EYE)
        lossless = np.allclose(ridge, np.conj(ridge).T)  # Hermitian ε: no loss
        rec = {"ours_ref": ours_ref, "fmmax_ref": fmmax_ref}
        rec.update(ours_self=[], fmmax_self=[], cross=[])
        for n in TRUNCATIONS:
            mine, other = ours(case, n), theirs(case, n)
            rec["ours_self"].append(max_difference(mine, central(ours_ref, REFERENCE, n)))
            rec["fmmax_self"].append(max_difference(other, central(fmmax_ref, REFERENCE, n)))
            rec["cross"].append(max_difference(mine, other))
            energy = float(np.max(np.abs(mine["R"].sum(-1) + mine["T"].sum(-1) - 1.0)))
            print(
                f"{name:<38}{2 * n + 1:>7}{rec['cross'][-1]:>11.1e}{rec['ours_self'][-1]:>11.1e}"
                f"{rec['fmmax_self'][-1]:>13.1e}{(energy if lossless else float('nan')):>12.1e}"
            )
        rec["cross"].append(max_difference(ours_ref, fmmax_ref))
        rec["extrapolated"], note = None, "(same factorisation: compared directly)"
        if anisotropic:
            # FMMAX's error falls as 1/N here, so 2·F(161) − F(81) removes the leading term.
            keep = REFERENCE // 2
            fmmax_half = theirs(case, keep)
            extrapolated = {
                q: 2 * central(fmmax_ref, REFERENCE, keep)[q] - fmmax_half[q] for q in "RT"
            }
            rec["extrapolated"] = max_difference(central(ours_ref, REFERENCE, keep), extrapolated)
            note = f"vs FMMAX extrapolated to ∞: {rec['extrapolated']:.1e}"
        worst = max(worst, rec["extrapolated"] if anisotropic else rec["cross"][-1])
        print(f"{'':<38}{2 * REFERENCE + 1:>7}{rec['cross'][-1]:>11.1e}   {note}")
        records[name] = rec

    plot_convergence(records, args.out / "convergence.png")
    print("\nSpectra, 41 orders, 31 points per case: max |Δ efficiency| hyperbolic-optics vs FMMAX")
    for name, difference in plot_spectra(spectra(), args.out / "spectra.png").items():
        print(f"  {name:<38}{difference:.1e}")

    case = CASES["LiNbO₃, axis in-plane, conical"]
    print("\nTiming, LiNbO₃ case, 41 orders, one wavelength, not jitted:")
    mine, other = _time(ours, case) * 1e3, _time(theirs, case) * 1e3
    print(f"  hyperbolic-optics {mine:.0f} ms, FMMAX {other:.0f} ms")
    print(f"\nFigures written to {args.out}")
    return 0 if worst < 5e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
