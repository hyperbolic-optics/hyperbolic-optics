"""Tests for resonances: band-edge modes, guided modes, spectral poles.

The anchors, none of which share code with the root finder:

- closed forms: the Fabry–Pérot poles of a uniform slab, its guided modes
  folded to Γ by a lattice, and the TE/TM (and uniaxial TM) slab dispersion;
- the real-frequency reflection spectrum, whose poles (by AAA) must be the
  bright band-edge modes found in the complex plane;
- symmetry: the degeneracy pattern at Γ of square and triangular lattices,
  lifted off Γ;
- energy: a lossy mode's Im n_eff against the absorbed power integrated over
  its profile;
- the NumPy payload engine: a hybrid LiNbO₃ mode's prism-coupling resonance.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.integrate import simpson  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from hyperbolic_optics import bcm, dispersion, rcwa2d, resonances  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
N_SLAB, T_SLAB = 3.5, 0.5  # a membrane, in units of the lattice constant


def _holes(lattice_centre, radius=0.15):
    shape = rcwa2d.Shape(rcwa2d.ellipse(lattice_centre, (radius, radius), 60), EYE)
    return rcwa2d.PatternedLayer(T_SLAB, N_SLAB**2 * EYE, (shape,))


class TestSpectrumPoles:
    def test_recovers_the_poles_of_a_rational_function(self):
        poles = np.array([1.2 - 0.01j, 1.5 - 0.002j])
        x = np.linspace(1.0, 2.0, 400)
        values = 0.3 + 0.1 / (x - poles[0]) - 0.02j / (x - poles[1])
        found, residues = resonances.spectrum_poles(x, values)
        for pole, residue in zip(poles, (0.1, -0.02j), strict=True):
            i = np.argmin(np.abs(found - pole))
            assert abs(found[i] - pole) < 1e-9 and abs(residues[i] - residue) < 1e-8


class TestUniformSlab:
    """No pattern: every resonance is known in closed form."""

    def test_fabry_perot_poles(self):
        r = (N_SLAB - 1) / (N_SLAB + 1)
        exact = (0.5 + 1j * np.log(r) / (2 * np.pi)) / (N_SLAB * T_SLAB)
        modes = resonances.band_edge_modes(
            [rcwa2d.uniform(N_SLAB**2 * EYE, T_SLAB)], rcwa2d.Lattice.square(1.0), (0.25, 0.32),
            1.0, 1.0, (0, 0), max_loss=0.07,
        )  # fmt: skip
        assert len(modes) == 1 and modes[0].multiplicity == 2  # p and s
        assert abs(modes[0].frequency - exact) < 1e-10
        assert modes[0].q_factor == pytest.approx(exact.real / (2 * abs(exact.imag)), rel=1e-9)

    def test_guided_modes_folded_to_gamma(self):
        beta = 2 * np.pi  # |b| on a unit square lattice

        def te_even(f):
            k0 = 2 * np.pi * f
            kappa, gamma = np.sqrt(N_SLAB**2 * k0**2 - beta**2), np.sqrt(beta**2 - k0**2)
            return kappa * np.tan(kappa * T_SLAB / 2) - gamma

        exact = brentq(te_even, 1 / N_SLAB + 1e-9, np.sqrt(2) / N_SLAB - 1e-9)
        modes = resonances.band_edge_modes(
            [rcwa2d.uniform(N_SLAB**2 * EYE, T_SLAB)], rcwa2d.Lattice.square(1.0),
            (exact - 0.01, exact + 0.01), 1.0, 1.0, (1, 1), max_loss=0.005,
        )  # fmt: skip
        assert len(modes) == 1
        mode = modes[0]
        assert mode.multiplicity == 4  # the four first-shell harmonics
        assert abs(mode.frequency - exact) < 1e-9 and mode.q_factor > 1e10


class TestPhotonicCrystalSlabs:
    def test_square_lattice_band_edge(self):
        """Two dark singlets and a bright pair, the pair a pole of the reflection spectrum."""
        lattice, order = rcwa2d.Lattice.square(1.0), (3, 3)
        layer = _holes((0.5, 0.5))
        modes = resonances.band_edge_modes(
            [layer], lattice, (0.33, 0.37), 1.0, 1.0, order, truncation="circular",
            max_loss=0.01, probes=24,
        )  # fmt: skip
        dark = [m for m in modes if m.q_factor > 1e8]
        bright = [m for m in modes if m.q_factor < 1e4]
        assert [m.multiplicity for m in dark] == [1, 1]
        assert len(bright) == 1 and bright[0].multiplicity == 2
        assert 100 < bright[0].q_factor < 2000

        # the same pole in the reflection spectrum at normal incidence
        f0 = bright[0].frequency.real
        frequencies = np.linspace(f0 - 0.004, f0 + 0.004, 161)
        reflection = []
        for f in frequencies:
            result = rcwa2d.solve_crossed_grating(
                [layer], lattice, 1.0 / f, 0.0, 1.0, 1.0, order, truncation="circular"
            )
            zeroth = int(np.flatnonzero((result.orders == 0).all(axis=1))[0])
            reflection.append(complex(result.r[zeroth, 0, 0]))
        poles, _ = resonances.spectrum_poles(frequencies, np.array(reflection))
        assert np.min(np.abs(poles - bright[0].frequency)) < 1e-7
        # and no pole at the dark modes: they do not couple to the incident plane wave
        for mode in dark:
            if frequencies[0] < mode.frequency.real < frequencies[-1]:
                assert np.min(np.abs(poles - mode.frequency)) > 1e-5

    def test_triangular_lattice_degeneracies(self):
        """C6v at Γ: singlets and doublets (Laurent's rule keeps the symmetry exact)."""
        modes = resonances.band_edge_modes(
            [_holes((0.0, 0.0))], rcwa2d.Lattice.hexagonal(1.0), (0.40, 0.44), 1.0, 1.0,
            (3, 3), truncation="circular", factorization="laurent", max_loss=0.01, probes=24,
        )  # fmt: skip
        assert [m.multiplicity for m in modes] == [1, 2, 2]
        assert modes[0].q_factor > 1e8 and modes[1].q_factor > 1e8  # dark
        assert modes[2].q_factor < 1e4  # the bright pair

    def test_off_gamma_the_pair_splits_and_dark_modes_leak(self):
        common = dict(
            frequency_range=(0.33, 0.37), eps_top=1.0, eps_bottom=1.0, max_order=(3, 3),
            truncation="circular", max_loss=0.01, probes=24, factorization="laurent",
        )  # fmt: skip
        layer, lattice = [_holes((0.5, 0.5))], rcwa2d.Lattice.square(1.0)
        at_gamma = resonances.band_edge_modes(layer, lattice, **common)
        off = resonances.band_edge_modes(layer, lattice, bloch=(0.03, 0.0), **common)
        assert sum(m.multiplicity for m in at_gamma) == sum(m.multiplicity for m in off) == 4
        assert all(m.multiplicity == 1 for m in off)
        assert all(np.isfinite(m.q_factor) and m.q_factor < 1e8 for m in off)


class TestSlabModes:
    WAVELENGTH, T, N_CLAD = 1.55, 0.6, 1.45

    def _analytic(self, eps_x, eps_y, eps_z, polarization, parity):
        k0 = 2 * np.pi / self.WAVELENGTH
        n2 = self.N_CLAD

        def condition(n):
            beta = n * k0
            gamma = np.sqrt(beta**2 - n2**2 * k0**2)
            if polarization == "te":
                kappa, scale = np.sqrt(eps_y * k0**2 - beta**2), 1.0
                clad = 1.0
            else:
                kappa, scale = np.sqrt(eps_x * k0**2 - eps_x / eps_z * beta**2), 1 / eps_x
                clad = 1 / n2**2
            phase = kappa * self.T / 2
            if parity == "even":
                return scale * kappa * np.sin(phase) - clad * gamma * np.cos(phase)
            return scale * kappa * np.cos(phase) + clad * gamma * np.sin(phase)

        grid = np.linspace(
            n2 + 1e-9, np.sqrt(eps_y if polarization == "te" else eps_z) - 1e-9, 4000
        )
        values = [condition(n) for n in grid]
        return [
            brentq(condition, grid[i], grid[i + 1])
            for i in range(len(grid) - 1)
            if np.sign(values[i]) != np.sign(values[i + 1])
        ]

    @pytest.mark.parametrize(
        "diagonal", [(4.0, 4.0, 4.0), (4.6, 4.0, 3.8)], ids=["isotropic", "uniaxial"]
    )
    def test_dispersion(self, diagonal):
        eps = jnp.asarray(np.diag(diagonal).astype(complex))[None]
        found = resonances.slab_modes(eps, jnp.array([self.T]), self.WAVELENGTH, self.N_CLAD**2,
                                      self.N_CLAD**2)  # fmt: skip
        indices = sorted(m.effective_index.real for m in found)
        expected = sorted(
            n
            for polarization in ("te", "tm")
            for parity in ("even", "odd")
            for n in self._analytic(*diagonal, polarization, parity)
        )
        np.testing.assert_allclose(indices, expected, atol=1e-9)
        assert all(abs(m.effective_index.imag) < 1e-10 for m in found)

    def test_lossy_mode_against_the_absorbed_power(self):
        eps = jnp.asarray(np.diag([4.6, 4.0, 3.8]).astype(complex) + 0.01j * np.eye(3))[None]
        args = (eps, jnp.array([self.T]), self.WAVELENGTH, self.N_CLAD**2, self.N_CLAD**2)
        mode = resonances.slab_modes(*args)[0]
        assert mode.loss_db_per_cm > 0
        z, fields = resonances.slab_mode_fields(
            *args, mode.effective_index, samples_per_layer=200, cladding_um=(3.0, 3.0),
            cladding_samples=800,
        )  # fmt: skip
        order = np.argsort(z, kind="stable")
        z, fields = z[order], fields[order]
        flux = 0.5 * np.real(
            fields[:, 1] * np.conj(fields[:, 5]) - fields[:, 2] * np.conj(fields[:, 4])
        )
        assert simpson(flux, x=z) == pytest.approx(1.0, abs=1e-4)
        inside = (z >= 0) & (z <= self.T)
        anti = (np.asarray(eps[0]) - np.asarray(eps[0]).conj().T) / 2j
        density = np.real(
            np.einsum("zi,ij,zj->z", fields[inside, :3].conj(), anti, fields[inside, :3])
        )
        absorbed = 0.25 * simpson(density, x=z[inside]) / simpson(flux, x=z)
        assert absorbed == pytest.approx(mode.effective_index.imag, rel=1e-6)

    def test_tangential_fields_are_continuous(self):
        eps = jnp.stack([2.0 * EYE, jnp.asarray(np.diag([4.6, 4.0, 3.8]).astype(complex))])
        args = (eps, jnp.array([0.3, 0.5]), self.WAVELENGTH, 1.0, self.N_CLAD**2)
        mode = resonances.slab_modes(*args)[0]
        z, fields = resonances.slab_mode_fields(*args, mode.effective_index, samples_per_layer=30)
        for boundary in (0.0, 0.3, 0.8):
            at = np.flatnonzero(np.isclose(z, boundary))
            tangential = fields[at][:, [0, 1, 3, 4]]  # Ex, Ey, Hx, Hy
            if len(at) > 1:
                np.testing.assert_allclose(tangential[0], tangential[-1], atol=1e-10)

    def test_hybrid_mode_against_prism_coupling_in_the_payload_engine(self):
        """x-cut LiNbO₃, axis 45° in the plane: TE and TM mix, and the NumPy engine agrees.

        Prism coupling through a 1.4 µm air gap converts p to s only at the
        mode's effective index (the gap shifts it by ~1e-10).
        """
        from hyperbolic_optics.structure import Structure

        wavelength, thickness, eps_sio2 = 1.55, 0.6, 1.444**2
        rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(45.0))
        ln = bcm.rotate_tensor(dispersion.at_wavelength("LiNbO3", wavelength), rotation)
        mode = resonances.slab_modes(ln[None], jnp.array([thickness]), wavelength, 1.0, eps_sio2)[1]
        n0 = mode.effective_index.real
        n_axis = np.linspace(n0 - 5e-7, n0 + 5e-7, 2001)
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {
                    "type": "Incident",
                    "incidentAngle": list(np.degrees(np.arcsin(n_axis / 2.4))),
                    "frequency": 1e4 / wavelength,
                },  # fmt: skip
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": 2.4**2},
                    {"type": "Isotropic Middle-Stack Layer", "thickness": 1.4, "permittivity": 1.0},
                    {
                        "type": "Crystal Layer",
                        "material": "LiNbO3",
                        "thickness": thickness,
                        "rotationY": 90,
                        "rotationZ": 45,
                    },  # fmt: skip
                    {"type": "Semi Infinite Isotropic Layer", "permittivity": eps_sio2},
                ],
            },
            backend="scattering",
        )
        conversion = np.abs(np.squeeze(structure.r_ps)) ** 2
        assert conversion.max() > 1e-3
        assert abs(n_axis[np.argmax(conversion)] - n0) < 5e-9
