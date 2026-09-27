"""Tests for grating couplers (``hyperbolic_optics.couplers``, ``resonances.grating_leaky_modes``).

- the exponential–Gaussian overlap's closed form against quadrature, and its
  maximum: the 80 % of a uniform grating, at ``2αw₀ ≈ 1.37``;
- the leaky mode found in the complex plane is a pole of the grating's own
  reflection spectrum, at the radiating order's complex ``kx``;
- its directionality equals the one read from the spectrum: the ratio of the
  transmission and reflection residues at that pole;
- an x-cut LiNbO₃ grating emits pure TE with its axis along the grating lines,
  and both polarizations with it at 45°;
- the SAX model carries the efficiencies, reciprocally.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from hyperbolic_optics import bcm, couplers, dispersion, rcwa, resonances  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
WAVELENGTH, PERIOD, N_SIN, N_OX = 1.55, 1.0, 2.0, 1.45
SIN_GRATING = [
    rcwa.lamellar_layer(N_SIN**2 * EYE, EYE, 0.5, 0.2),  # half-etched SiN, in air
    rcwa.uniform_layer(N_SIN**2 * EYE, 0.2),
]


class TestOverlap:
    def test_closed_form_against_quadrature(self):
        a, w, x0, length, angle = 0.12, 5.2, 6.0, 20.0, 10.0
        x = np.linspace(0, length, 200001)
        footprint = w / np.cos(np.radians(angle))
        emitted = np.exp(-a * x)
        mode = np.exp(-(((x - x0) / footprint) ** 2))
        wide = np.linspace(-60, 80, 400001)
        norm = np.trapezoid(np.exp(-2 * ((wide - x0) / footprint) ** 2), wide)
        numeric = np.trapezoid(emitted * mode, x) ** 2 / (np.trapezoid(emitted**2, x) * norm)
        closed = couplers.exponential_gaussian_overlap(a, w, x0, length, angle)
        assert closed == pytest.approx(numeric, rel=1e-8)

    def test_the_uniform_grating_limit(self):
        best = minimize_scalar(
            lambda a: -couplers.optimal_overlap(a, 5.2)[0], bounds=(0.01, 1.0), method="bounded"
        )
        assert -best.fun == pytest.approx(0.801, abs=1e-3)
        assert 2 * best.x * 5.2 == pytest.approx(1.37, abs=0.01)


@pytest.fixture(scope="module")
def sin_mode():
    modes = resonances.grating_leaky_modes(
        SIN_GRATING, PERIOD, WAVELENGTH, 1.0, N_OX**2, (1.52, 1.72), max_order=10, max_loss=0.02
    )
    return modes[0]


class TestLeakyMode:
    def test_it_radiates_through_the_minus_first_order(self, sin_mode):
        assert sin_mode.alpha_per_um > 0 and -1 in sin_mode.emission_deg
        assert 0 < sin_mode.directionality < 1
        p, s = sin_mode.top_polarization[-1]
        assert abs(s) > 0.999  # an isotropic grating: TE in, TE out

    def test_it_is_a_pole_of_the_reflection_spectrum(self, sin_mode):
        target = sin_mode.bloch_index - WAVELENGTH / PERIOD  # the −1st order's kx
        kxs = np.linspace(target.real - 0.03, target.real + 0.03, 161)
        solve = jax.jit(
            lambda angle: rcwa.solve_grating(
                SIN_GRATING, PERIOD, WAVELENGTH, angle, 1.0, N_OX**2, 10
            )  # fmt: skip
        )
        r, t = [], []
        for kx in kxs:
            result = solve(np.arcsin(kx))
            r.append(complex(result.r[10, 1, 1]))
            t.append(complex(result.t[10, 1, 1]))
        poles_r, residues_r = resonances.spectrum_poles(kxs, np.array(r))
        poles_t, residues_t = resonances.spectrum_poles(kxs, np.array(t))
        i, j = np.argmin(np.abs(poles_r - target)), np.argmin(np.abs(poles_t - target))
        assert (
            abs(poles_r[i] - target) < 1e-5 and abs(poles_t[j] - target) < 1e-5
        )  # a broad resonance

        # the directionality, read from the spectrum: |c_down/c_up|² from the residues,
        # power-normalised by the two half-spaces' kz
        kz_top, kz_bottom = np.sqrt(1 - target.real**2), np.sqrt(N_OX**2 - target.real**2)
        ratio = abs(residues_t[j]) ** 2 * (kz_bottom / kz_top) / abs(residues_r[i]) ** 2
        assert 1 / (1 + ratio) == pytest.approx(sin_mode.directionality, abs=1e-3)


class TestCoupler:
    def test_efficiency_bookkeeping(self):
        coupler = couplers.grating_coupler(SIN_GRATING, PERIOD, WAVELENGTH, 1.0, N_OX**2,
                                           (1.52, 1.72), length_um=15.0, max_loss=0.02)  # fmt: skip
        assert coupler.efficiency == pytest.approx(
            coupler.directionality * coupler.radiated * coupler.overlap
        )
        assert coupler.efficiency_te + coupler.efficiency_tm == pytest.approx(coupler.efficiency)
        assert coupler.radiated == pytest.approx(1 - np.exp(-2 * coupler.mode.alpha_per_um * 15.0))
        assert 0 < coupler.efficiency < 0.8

    def test_sax_model(self):
        model = couplers.grating_coupler_model(SIN_GRATING, PERIOD, 1.0, N_OX**2, (1.52, 1.72),
                                               length_um=15.0, max_loss=0.02)  # fmt: skip
        sd = model(wl=np.array([1.55, 1.56]))
        assert sd["waveguide@TE", "fiber@TE"].shape == (2,)
        np.testing.assert_array_equal(
            sd["waveguide@TE", "fiber@TE"], sd["fiber@TE", "waveguide@TE"]
        )
        single = couplers.grating_coupler(SIN_GRATING, PERIOD, 1.56, 1.0, N_OX**2, (1.52, 1.72),
                                          length_um=15.0, max_loss=0.02)  # fmt: skip
        assert abs(sd["waveguide@TE", "fiber@TE"][1]) ** 2 == pytest.approx(
            single.efficiency_te, rel=1e-6
        )


@pytest.mark.parametrize(("azimuth", "mixed"), [(90.0, False), (45.0, True)])
def test_x_cut_lithium_niobate_polarization(azimuth, mixed):
    """An x-cut LiNbO₃ grating: its axis along the lines (y) keeps TE pure; at 45° it mixes."""
    rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(azimuth))
    ln = bcm.rotate_tensor(dispersion.at_wavelength("LiNbO3", WAVELENGTH), rotation)
    layers = [rcwa.lamellar_layer(ln, EYE, 0.5, 0.2), rcwa.uniform_layer(ln, 0.2)]
    slab = resonances.slab_modes(ln[None], jnp.array([0.3]), WAVELENGTH, 1.0, N_OX**2)
    guess = slab[0].effective_index.real
    period = WAVELENGTH / (guess - np.sin(np.radians(10.0)))
    mode = resonances.grating_leaky_modes(
        layers, period, WAVELENGTH, 1.0, N_OX**2, (guess - 0.12, guess + 0.12), max_order=10,
        max_loss=0.06,  # LiNbO3 in air etches a strong grating: Im kx₀/k₀ ≈ 0.02
    )[0]  # fmt: skip
    p, s = mode.top_polarization[-1]
    if mixed:
        assert min(abs(p), abs(s)) > 1e-2
    else:
        assert abs(p) < 1e-8


class TestLocalModel:
    def test_uniform_profile_is_the_closed_form(self):
        x = np.linspace(0, 30, 6001)
        alpha, directionality, angle = 0.12, 0.7, 10.0
        te, tm, offset, _ = couplers.coupling_from_profile(
            x, np.full_like(x, alpha), np.full_like(x, directionality),
            np.full_like(x, np.sin(np.radians(angle))), np.tile([0.0, 1.0], (x.size, 1)),
            WAVELENGTH, 5.2, angle,
        )  # fmt: skip
        overlap, best = couplers.optimal_overlap(alpha, 5.2, 30.0, angle)
        radiated = 1 - np.exp(-2 * alpha * 30)
        assert te == pytest.approx(directionality * radiated * overlap, rel=1e-6)
        assert tm == pytest.approx(0.0, abs=1e-15) and offset == pytest.approx(best, abs=0.01)

    def test_the_ideal_apodization_couples_everything(self):
        """α(x) = G²/(2(1 − ∫G²)) turns the emission into the fibre's own Gaussian."""
        angle = np.radians(10.0)
        footprint = 5.2 / np.cos(angle)
        x = np.linspace(-3 * footprint, 3 * footprint, 20001)
        target = np.exp(-2 * (x / footprint) ** 2)
        target /= np.trapezoid(target, x)
        alpha = target / (2 * np.clip(1 - np.cumsum(target) * (x[1] - x[0]), 1e-12, None))
        te, _, offset, fitted = couplers.coupling_from_profile(
            x, alpha, np.ones_like(x), np.full_like(x, np.sin(angle)),
            np.tile([0.0, 1.0], (x.size, 1)), WAVELENGTH, 5.2,
        )  # fmt: skip
        assert te > 0.9999 and fitted == pytest.approx(10.0, abs=1e-3) and abs(offset) < 0.01

    def test_an_apodized_grating_follows_its_mode_tooth_by_tooth(self):
        fills = np.linspace(0.35, 0.65, 6)
        layers = [
            [
                rcwa.lamellar_layer(N_SIN**2 * EYE, EYE, float(f), 0.2),
                rcwa.uniform_layer(N_SIN**2 * EYE, 0.2),
            ]
            for f in fills
        ]
        result = couplers.apodized_grating_coupler(
            lambda i: layers[i], np.arange(6) * PERIOD, np.full(6, PERIOD), WAVELENGTH, 1.0,
            N_OX**2, (1.52, 1.72), max_loss=0.03,
        )  # fmt: skip
        alphas = [m.alpha_per_um for m in result.modes]
        assert len(set(np.round(alphas, 8))) == 6  # every tooth its own leaky mode
        assert 0 < result.efficiency < 1
        # a refined root is the one a fresh search finds
        fresh = resonances.grating_leaky_modes(
            layers[3], PERIOD, WAVELENGTH, 1.0, N_OX**2, (1.52, 1.72), max_loss=0.03
        )[0]
        assert abs(fresh.bloch_index - result.modes[3].bloch_index) < 1e-9
