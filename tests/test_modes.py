"""Tests for guided modes of anisotropic stacks (``hyperbolic_optics.modes``).

The anchors, none of which share code with the root finder:

- closed forms: the TE/TM (and uniaxial TM) slab dispersion;
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

from hyperbolic_optics import bcm, dispersion, modes  # noqa: E402

EYE = jnp.eye(3, dtype=complex)


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
        found = modes.slab_modes(eps, jnp.array([self.T]), self.WAVELENGTH, self.N_CLAD**2,
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
        mode = modes.slab_modes(*args)[0]
        assert mode.loss_db_per_cm > 0
        z, fields = modes.slab_mode_fields(
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
        mode = modes.slab_modes(*args)[0]
        z, fields = modes.slab_mode_fields(*args, mode.effective_index, samples_per_layer=30)
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
        mode = modes.slab_modes(ln[None], jnp.array([thickness]), wavelength, 1.0, eps_sio2)[1]
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
