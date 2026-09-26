"""Cross-check hyperbolic-optics against DiffTMM on planar anisotropic stacks.

DiffTMM (https://github.com/AI4Optics/DiffTMM, Apache-2.0) is an independent,
PyTorch 4x4 Berreman transfer-matrix code. It is not a dependency of this
package; README.md alongside gives the command that runs this against the
DiffTMM commit it was checked with (``DIFFTMM_COMMIT``).

The two codes normalise amplitudes differently (DiffTMM's p amplitude is the
tangential Ex, not the total field) and name cross-polarised terms differently,
so everything is compared as *power*: R and T resolved by incident and outgoing
polarization, which is convention-free. Both solvers are given the same
lab-frame permittivity tensor: hyperbolic-optics receives it directly as an
arbitrary material, and DiffTMM's own rotation of the principal indices is first
checked to reproduce it, so neither package's Euler-angle convention can hide or
fake agreement.

Cases:

1. ``isotropic_film``   -- both solvers against the closed-form Airy result.
2. ``paper_fig3``       -- the anisotropic multilayer of Yang, Song & Cai,
   arXiv:2608.06185, Fig. 3 (in-plane LiNbO3 axis at 45 degrees, 30 degree incidence).
3. ``tilted_axis``      -- the same stack with the LiNbO3 axis tilted out of plane,
   so every off-diagonal tensor component is nonzero; swept in angle.
4. ``hyperbolic_ir``    -- a lossy hyperbolic film (Re eps of opposite signs) with a
   tilted axis, swept in frequency.
5. ``prism_gap``        -- prism-coupled (Otto-like) evanescent gap swept in
   thickness, the regime this package's scattering backend exists for. This one
   locates where each method stops being usable rather than asserting agreement.

DiffTMM computes in complex64, which sets the agreement floor (~1e-5 in power);
its own energy-conservation residual on the lossless cases is reported so that
floor is measured, not assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from difftmm.film_solver_anisotropic import (  # noqa: E402
    create_eps_matrix_XY,
    create_jones_matrix_AOIAz,
)

from hyperbolic_optics import bcm  # noqa: E402
from hyperbolic_optics.fields import FieldProfile  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402

DIFFTMM_COMMIT = "7671270bf6c08ba377c380d1c512def80e21659e"

# Power agreement required between the solvers on cases 1-4. Set by DiffTMM's
# complex64 arithmetic; the measured floor is printed alongside.
TOL_CROSS_SOLVER = 2e-4
# hyperbolic-optics against the Airy closed form, and its two backends against
# each other where both are well-conditioned (complex128 throughout).
TOL_EXACT = 1e-9

POLS = ("p", "s")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


# ---------------------------------------------------------------------------
# Stack description
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """A uniaxial (or isotropic, eps_o == eps_e) layer.

    The optic axis is ``c = (cos t cos a, cos t sin a, sin t)`` in the lab frame
    (x in the plane of incidence, z the stack normal), with ``t`` the tilt out of
    the interface plane and ``a`` the in-plane azimuth from x.
    """

    eps_o: complex
    eps_e: complex
    thickness_um: float
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0

    @property
    def axis(self) -> np.ndarray:
        t, a = np.radians(self.tilt_deg), np.radians(self.azimuth_deg)
        return np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])

    def expected_tensor(self) -> np.ndarray:
        c = self.axis
        return self.eps_o * np.eye(3) + (self.eps_e - self.eps_o) * np.outer(c, c)

    def difftmm_spec(self) -> tuple[list[complex], list[float]]:
        """Principal indices and DiffTMM rotation angles reproducing ``axis``.

        DiffTMM puts the principal z axis along
        ``(cos th cos ph, cos th sin ph, sin th)`` with ``ph = a0 + pi/2`` and
        ``th = a1 + pi/2`` (film_solver_anisotropic.create_eps_matrix_XY).
        """
        n_o, n_e = np.sqrt(complex(self.eps_o)), np.sqrt(complex(self.eps_e))
        angles = [
            np.radians(self.azimuth_deg) - np.pi / 2,
            np.radians(self.tilt_deg) - np.pi / 2,
            0.0,
        ]
        return [n_o, n_o, n_e], angles


def isotropic(eps: float, thickness_um: float) -> Layer:
    return Layer(eps, eps, thickness_um)


@dataclass
class Stack:
    name: str
    title: str
    n_in: float
    n_out: float
    layers: list[Layer]
    theta_deg: np.ndarray
    wavelength_um: np.ndarray
    lossless: bool = True

    @property
    def sweep(self) -> tuple[str, np.ndarray]:
        if self.theta_deg.size > 1:
            return "Incident angle (deg)", self.theta_deg
        return "Wavelength (nm)", self.wavelength_um * 1e3


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def check_difftmm_rotation(stack: Stack) -> None:
    """DiffTMM's rotated tensors must equal the intended lab-frame tensors.

    Checked to complex64 precision; hyperbolic-optics is handed the exact
    complex128 tensor, so DiffTMM's rounding does not leak into its result.
    """
    specs = [layer.difftmm_spec() for layer in stack.layers]
    n_2d = torch.tensor([[s[0] for s in specs]], dtype=torch.complex64)
    a_2d = torch.tensor([[s[1] for s in specs]], dtype=torch.complex64)
    eps = create_eps_matrix_XY(a_2d, n_2d.unsqueeze(1), torch.zeros(1, 1, 1))[0, 0, 0, 0]
    tensors = [e.numpy().astype(np.complex128) for e in eps]
    for layer, tensor in zip(stack.layers, tensors):
        expected = layer.expected_tensor()
        scale = np.max(np.abs(expected))
        if np.max(np.abs(tensor - expected)) > 1e-5 * scale:
            raise AssertionError(f"{stack.name}: DiffTMM rotation does not give the intended axis")


def _flux_weights(n: float, sin_theta_in: np.ndarray, n_in: float) -> np.ndarray:
    """Normal-flux weight per unit |amplitude|^2 in DiffTMM's basis, ``[..., (p, s)]``.

    DiffTMM's p amplitude is Ex (Hy = n/cos Ex) and its s amplitude Ey
    (Hx-like = n cos Ey), so S_z scales as n/cos for p and n cos for s. For an
    evanescent exit cos is imaginary and the weight's real part vanishes.
    """
    cos = np.sqrt(1 - (n_in * sin_theta_in / n) ** 2 + 0j)
    return np.stack([n / cos, n * cos], axis=-1)


def run_difftmm(stack: Stack) -> dict[str, np.ndarray]:
    specs = [layer.difftmm_spec() for layer in stack.layers]
    n_2d = torch.tensor([[s[0] for s in specs]], dtype=torch.complex64)
    a_2d = torch.tensor([[s[1] for s in specs]], dtype=torch.complex64)
    d_1d = torch.tensor([[layer.thickness_um for layer in stack.layers]], dtype=torch.complex64)
    wl = torch.tensor(stack.wavelength_um[None], dtype=torch.float32)
    theta = torch.tensor(np.radians(stack.theta_deg)[None], dtype=torch.float32)

    jt, jr = create_jones_matrix_AOIAz(
        a_2d, n_2d, d_1d, wl, stack.n_in, stack.n_out, theta, torch.zeros(1, 1)
    )
    # [1, F, A, 1, out, in] -> [N, out, in]; one of F, A is 1. Index 0 = p, 1 = s.
    jt = jt[0, :, :, 0].numpy().astype(np.complex128).reshape(-1, 2, 2)
    jr = jr[0, :, :, 0].numpy().astype(np.complex128).reshape(-1, 2, 2)

    sin_in = np.sin(np.radians(stack.theta_deg))
    sin_in = np.broadcast_to(sin_in[None, :], (stack.wavelength_um.size, sin_in.size)).ravel()
    w_in = np.real(_flux_weights(stack.n_in, sin_in, stack.n_in))  # [N, 2]
    w_out = np.real(_flux_weights(stack.n_out, sin_in, stack.n_in))

    out = {}
    for j, pin in enumerate(POLS):
        for i, pout in enumerate(POLS):
            out[f"R_{pin}{pout}"] = np.abs(jr[:, i, j]) ** 2 * w_in[:, i] / w_in[:, j]
            out[f"T_{pin}{pout}"] = np.abs(jt[:, i, j]) ** 2 * w_out[:, i] / w_in[:, j]
        out[f"R_{pin}"] = out[f"R_{pin}p"] + out[f"R_{pin}s"]
        out[f"T_{pin}"] = out[f"T_{pin}p"] + out[f"T_{pin}s"]
    return out


def _material(tensor: np.ndarray) -> dict:
    def c(z):
        return {"real": float(np.real(z)), "imag": float(np.imag(z))}

    return {
        "eps_xx": c(tensor[0, 0]),
        "eps_yy": c(tensor[1, 1]),
        "eps_zz": c(tensor[2, 2]),
        "eps_xy": c(tensor[0, 1]),
        "eps_xz": c(tensor[0, 2]),
        "eps_yz": c(tensor[1, 2]),
    }


def run_hyperbolic_optics(stack: Stack, backend: str) -> dict[str, np.ndarray]:
    payload = {
        "ScenarioData": {
            "type": "Incident",
            "incidentAngle": stack.theta_deg.tolist(),
            "frequency": (1e4 / stack.wavelength_um).tolist(),  # cm^-1
        },
        "Layers": [{"type": "Ambient Incident Layer", "permittivity": stack.n_in**2}]
        + [
            {
                "type": "Crystal Layer",
                "material": _material(layer.expected_tensor()),
                "thickness": layer.thickness_um,
            }
            for layer in stack.layers
        ]
        + [{"type": "Semi Infinite Isotropic Layer", "permittivity": stack.n_out**2}],
    }
    structure = Structure()
    with warnings.catch_warnings():
        # Arbitrary tensors carry no validity range, so every frequency is
        # "out of range"; that warning is about material data, not the solver.
        warnings.filterwarnings("ignore", message=".*outside.*range.*")
        structure.execute(payload, backend=backend)
    fp = FieldProfile(structure)

    out = {}
    for pin in POLS:
        out[f"R_{pin}"] = np.ravel(fp.reflectance(pin))
        out[f"T_{pin}"] = np.ravel(fp.transmittance(pin))
        if backend == "transfer":
            # The per-channel split needs the transfer product (see FieldProfile).
            split = fp.polarization_resolved(pin)
            cross = "s" if pin == "p" else "p"
            out[f"R_{pin}{pin}"] = np.ravel(split["R_co"])
            out[f"R_{pin}{cross}"] = np.ravel(split["R_cross"])
            out[f"T_{pin}{pin}"] = np.ravel(split["T_co"])
            out[f"T_{pin}{cross}"] = np.ravel(split["T_cross"])
    return out


def run_bcm(stack: Stack) -> dict[str, np.ndarray]:
    """The differentiable JAX solver, fed the tensors the NumPy engine simulates.

    A ``Crystal Layer`` offsets rotationY by 1e-8 rad and rotationZ by 1e-9 rad
    (layers.Layer.__init__), so the engine's tensor differs from the specified
    one at ~1e-8. Applying the same offset here compares solvers, not inputs.
    """
    offset = np.asarray(bcm.rotation_matrix(0.0, 1e-8, 1e-9))
    tensors = np.stack([offset @ layer.expected_tensor() @ offset.T for layer in stack.layers])
    thickness = np.array([layer.thickness_um for layer in stack.layers])
    result = _BCM(
        jnp.asarray(tensors),
        jnp.asarray(thickness),
        jnp.asarray(stack.wavelength_um)[:, None],  # [F, A]; one of the two is 1
        jnp.radians(jnp.asarray(stack.theta_deg))[None, :],
        stack.n_in**2,
        stack.n_out**2,
    )
    power = {
        "R": np.asarray(result.R).reshape(-1, 2, 2),
        "T": np.asarray(result.T).reshape(-1, 2, 2),
    }
    out = {}
    for j, pin in enumerate(POLS):
        for q in "RT":
            for i, pout in enumerate(POLS):
                out[f"{q}_{pin}{pout}"] = power[q][:, i, j]
            out[f"{q}_{pin}"] = out[f"{q}_{pin}p"] + out[f"{q}_{pin}s"]
    return out


_BCM = jax.jit(bcm.solve_stack, static_argnames="doublings")


def airy(stack: Stack) -> dict[str, np.ndarray]:
    """Closed-form R, T of a single lossless isotropic film (case 1 only)."""
    (film,) = stack.layers
    n0, n1, n2 = stack.n_in, np.sqrt(film.eps_o), stack.n_out
    s0 = n0 * np.sin(np.radians(stack.theta_deg))
    c0, c1, c2 = (np.sqrt(1 - (s0 / n) ** 2 + 0j) for n in (n0, n1, n2))
    beta = 2 * np.pi / stack.wavelength_um * n1 * c1 * film.thickness_um

    def fresnel(na, ca, nb, cb, pol):
        if pol == "s":
            return (na * ca - nb * cb) / (na * ca + nb * cb)
        return (nb * ca - na * cb) / (nb * ca + na * cb)

    out = {}
    for pol in POLS:
        r01, r12 = fresnel(n0, c0, n1, c1, pol), fresnel(n1, c1, n2, c2, pol)
        phase = np.exp(2j * beta)
        r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
        out[f"R_{pol}"] = np.abs(r) ** 2
        out[f"T_{pol}"] = 1 - out[f"R_{pol}"]  # lossless film
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

LN_O, LN_E = 5.46, 5.06  # arXiv:2608.06185 Sec. 3A, dispersion neglected


def paper_stack(ln: Layer) -> list[Layer]:
    return [isotropic(1.38**2, 0.070), isotropic(2.40**2, 0.190), ln, isotropic(1.46**2, 0.080)]


def build_cases() -> list[Stack]:
    visible = np.linspace(0.45, 1.0, 221)
    ir = 1e4 / np.linspace(800.0, 1200.0, 201)
    return [
        Stack(
            "isotropic_film",
            "Isotropic film vs closed form",
            1.0,
            1.52,
            [isotropic(2.40**2, 0.200)],
            np.array([30.0]),
            visible,
        ),
        Stack(
            "paper_fig3",
            "arXiv:2608.06185 Fig. 3 multilayer",
            1.0,
            1.52,
            paper_stack(Layer(LN_O, LN_E, 2.8, tilt_deg=0.0, azimuth_deg=45.0)),
            np.array([30.0]),
            visible,
        ),
        Stack(
            "tilted_axis",
            "Tilted LiNbO3 axis (full tensor)",
            1.0,
            1.52,
            paper_stack(Layer(LN_O, LN_E, 2.8, tilt_deg=45.0, azimuth_deg=30.0)),
            np.linspace(0.0, 85.0, 171),
            np.array([0.633]),
        ),
        Stack(
            "hyperbolic_ir",
            "Lossy hyperbolic film, tilted axis",
            1.0,
            2.4,
            [Layer(-4.84 + 0.755j, 2.27 + 0.001j, 1.0, tilt_deg=30.0, azimuth_deg=20.0)],
            np.array([45.0]),
            ir,
            lossless=False,
        ),
    ]


def prism_gap_stack(gap_um: float) -> Stack:
    return Stack(
        "prism_gap",
        "Prism-coupled evanescent gap",
        np.sqrt(5.5),
        2.4,
        [
            isotropic(1.0, gap_um),
            Layer(-4.84 + 0.755j, 2.27 + 0.001j, 0.5, tilt_deg=30.0, azimuth_deg=20.0),
        ],
        np.array([45.0]),
        np.array([10.0]),  # 1000 cm^-1
        lossless=False,
    )


# ---------------------------------------------------------------------------
# Comparison, reporting, plotting
# ---------------------------------------------------------------------------


def max_diff(a: dict, b: dict, keys) -> float:
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in keys)


CHANNELS = [f"{q}_{i}{o}" for q in "RT" for i in POLS for o in POLS]
TOTALS = [f"{q}_{i}" for q in "RT" for i in POLS]


def compare_case(stack: Stack) -> dict:
    check_difftmm_rotation(stack)
    dt = run_difftmm(stack)
    ho = run_hyperbolic_optics(stack, "transfer")
    ho_sc = run_hyperbolic_optics(stack, "scattering")
    jx = run_bcm(stack)
    row = {
        "case": stack.name,
        "points": dt["R_p"].size,
        "ho_vs_difftmm": max_diff(ho, dt, CHANNELS),
        "transfer_vs_scattering": max_diff(ho, ho_sc, TOTALS),
        "bcm_vs_ho": max_diff(jx, ho, CHANNELS),
        "difftmm_energy": (
            max(float(np.max(np.abs(dt[f"R_{p}"] + dt[f"T_{p}"] - 1))) for p in POLS)
            if stack.lossless
            else float("nan")
        ),
        "ho_energy": (
            max(float(np.max(np.abs(ho[f"R_{p}"] + ho[f"T_{p}"] - 1))) for p in POLS)
            if stack.lossless
            else float("nan")
        ),
    }
    if stack.name == "isotropic_film":
        exact = airy(stack)
        row["ho_vs_exact"] = max_diff(ho, exact, TOTALS)
        row["difftmm_vs_exact"] = max_diff(dt, exact, TOTALS)
    row["passed"] = (
        row["ho_vs_difftmm"] <= TOL_CROSS_SOLVER
        and row["transfer_vs_scattering"] <= TOL_EXACT
        and row["bcm_vs_ho"] <= TOL_EXACT
        and row.get("ho_vs_exact", 0.0) <= TOL_EXACT
    )
    return {"row": row, "stack": stack, "ho": ho, "difftmm": dt}


def gap_sweep(gaps_um: np.ndarray) -> dict[str, np.ndarray]:
    """1 - R_p for each solver versus gap thickness (the frustrated coupling)."""
    res = {"transfer": [], "scattering": [], "bcm": [], "difftmm": []}
    for gap in gaps_um:
        stack = prism_gap_stack(gap)
        check_difftmm_rotation(stack)
        res["difftmm"].append(run_difftmm(stack)["R_p"][0])
        res["bcm"].append(run_bcm(stack)["R_p"][0])
        for backend in ("transfer", "scattering"):
            res[backend].append(run_hyperbolic_optics(stack, backend)["R_p"][0])
    return {k: 1.0 - np.array(v) for k, v in res.items()}


def timing(stack: Stack, repeats: int = 7) -> dict[str, float]:
    def best(fn):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return float(np.median(times))

    return {
        "hyperbolic-optics (NumPy, complex128)": best(
            lambda: run_hyperbolic_optics(stack, "transfer")
        ),
        "DiffTMM (PyTorch CPU, complex64)": best(lambda: run_difftmm(stack)),
        "bcm (JAX CPU, complex128, jitted)": best(lambda: run_bcm(stack)),
    }


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def plot_channels(result: dict, path: Path) -> None:
    """Paper-style 2x2: T and R for p- and s-incidence, co (blue) and cross (orange)."""
    stack, ho, dt = result["stack"], result["ho"], result["difftmm"]
    xlabel, x = stack.sweep
    every = max(1, x.size // 45)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, constrained_layout=True)
    for col, pin in enumerate(POLS):
        cross = "s" if pin == "p" else "p"
        for row, q in enumerate("TR"):
            ax = axes[row, col]
            _style(ax)
            for pout, color, label in ((pin, BLUE, "co-pol"), (cross, ORANGE, "cross-pol")):
                key = f"{q}_{pin}{pout}"
                ax.plot(x, ho[key], color=color, linewidth=2, label=f"{label}: hyperbolic-optics")
                ax.plot(
                    x[::every],
                    dt[key][::every],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    color=color,
                    label=f"{label}: DiffTMM",
                )
            ax.set_title(
                f"{'Transmittance' if q == 'T' else 'Reflectance'}, {pin} incident",
                fontsize=9,
                color=INK,
                loc="left",
            )
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(stack.title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gap(gaps: np.ndarray, res: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    _style(ax)
    series = (
        ("scattering", "hyperbolic-optics, scattering backend", BLUE, "-", None),
        ("transfer", "hyperbolic-optics, transfer backend", ORANGE, "--", None),
        ("bcm", "hyperbolic-optics, JAX solver (bcm)", AQUA, ":", None),
        ("difftmm", "DiffTMM", INK, "none", "o"),
    )
    for key, label, color, ls, marker in series:
        y = np.abs(res[key])
        ax.semilogy(
            gaps,
            np.where(y > 0, y, np.nan),
            color=color,
            linestyle=ls,
            linewidth=2,
            marker=marker,
            markersize=5,
            markerfacecolor="none",
            label=label,
        )
    ax.set_xlabel("Air-gap thickness (µm)", fontsize=9, color=INK)
    ax.set_ylabel("|1 − R|, p incident", fontsize=9, color=INK)
    ax.set_title(
        "Prism-coupled gap: coupled power should fall exponentially",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import difftmm

    print(
        f"DiffTMM {getattr(difftmm, '__version__', '?')} @ {DIFFTMM_COMMIT[:12]}, "
        f"torch {torch.__version__}, numpy {np.__version__}\n"
    )

    results = [compare_case(stack) for stack in build_cases()]
    header = (
        f"{'case':<16}{'pts':>5}{'ho vs DiffTMM':>15}{'xfer vs scat':>14}{'bcm vs ho':>11}"
        f"{'DiffTMM |R+T-1|':>17}{'ho |R+T-1|':>12}  ok"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        r = res["row"]
        print(
            f"{r['case']:<16}{r['points']:>5}{r['ho_vs_difftmm']:>15.1e}"
            f"{r['transfer_vs_scattering']:>14.1e}{r['bcm_vs_ho']:>11.1e}"
            f"{r['difftmm_energy']:>17.1e}"
            f"{r['ho_energy']:>12.1e}  {'yes' if r['passed'] else 'NO'}"
        )
        plot_channels(res, args.out / f"{r['case']}.png")
    exact = results[0]["row"]
    print(
        f"\nisotropic_film vs Airy closed form: hyperbolic-optics {exact['ho_vs_exact']:.1e}, "
        f"DiffTMM {exact['difftmm_vs_exact']:.1e}"
    )
    print(
        f"(max |dR|, |dT| over all resolved channels; tolerance {TOL_CROSS_SOLVER:g} "
        f"cross-solver, {TOL_EXACT:g} exact)"
    )

    gaps = np.geomspace(0.1, 20.0, 41)
    gap = gap_sweep(gaps)
    plot_gap(gaps, gap, args.out / "prism_gap.png")
    print("\nprism_gap: |1 - R_p| (frustrated coupling through the gap)")
    print(f"{'gap (um)':>9}{'scattering':>13}{'transfer':>13}{'bcm':>13}{'DiffTMM':>13}")
    for idx in range(0, gaps.size, 5):
        print(
            f"{gaps[idx]:>9.2f}{gap['scattering'][idx]:>13.2e}"
            f"{gap['transfer'][idx]:>13.2e}{gap['bcm'][idx]:>13.2e}{gap['difftmm'][idx]:>13.2e}"
        )

    print("\nTiming, paper_fig3 (221 wavelengths, all 8 channels), median wall time:")
    for name, seconds in timing(results[1]["stack"]).items():
        print(f"  {name:<40}{seconds * 1e3:8.1f} ms")

    print(f"\nFigures written to {args.out}")
    return 0 if all(r["row"]["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
