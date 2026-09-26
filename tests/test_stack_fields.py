"""Tests for differentiable fields and absorption (``bcm.stack_fields``).

- R and T are :func:`bcm.solve_stack`'s;
- absorption, R, T and the six field components against depth match the NumPy
  engine's ``FieldProfile``, which reconstructs them by a different route;
- each layer's absorption equals the constitutive loss density integrated over
  its depth -- electric and magnetic -- an independent check, where summing
  interface fluxes alone would only telescope;
- thick, strongly absorbing and evanescent stacks stay finite and passive;
- gradients (thickness, wavelength) match finite differences.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.integrate import simpson  # noqa: E402

from hyperbolic_optics import bcm, dispersion  # noqa: E402
from hyperbolic_optics.fields import FieldProfile  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
WAVELENGTH = 1e4 / 1470.17
CALCITE = bcm.rotate_tensor(
    dispersion.at_wavelength("Calcite", WAVELENGTH),
    bcm.rotation_matrix(0.0, np.radians(-65.0), np.radians(20.0)),
)
EPS = jnp.stack([2.0 * EYE, CALCITE, (3.0 + 0.4j) * EYE])
THICKNESS = jnp.array([0.8, 1.7, 0.6])


def _fields(samples=0, **kwargs):
    args = {"eps": EPS, "thickness_um": THICKNESS, "wavelength_um": WAVELENGTH,
            "incidence_rad": 0.6, "eps_incident": 2.25, "eps_exit": 1.0}  # fmt: skip
    args.update(kwargs)
    return bcm.stack_fields(**args, samples_per_layer=samples)


def test_power_is_solve_stacks():
    fields = _fields()
    planar = bcm.solve_stack(EPS, THICKNESS, WAVELENGTH, 0.6, 2.25, 1.0)
    np.testing.assert_allclose(fields.R, np.asarray(planar.R).sum(0), atol=1e-14)
    np.testing.assert_allclose(fields.T, np.asarray(planar.T).sum(0), atol=1e-14)
    total = fields.R + fields.T + fields.absorption.sum(-2)
    np.testing.assert_allclose(total, 1.0, atol=1e-13)


class TestAgainstNumpyEngine:
    @pytest.fixture(scope="class")
    def structure(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {
                    "type": "Simple",
                    "incidentAngle": 40.0,
                    "azimuthal_angle": 0.0,
                    "frequency": 1470.17,
                },  # fmt: skip
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": 2.25},
                    {
                        "type": "Isotropic Middle-Stack Layer",
                        "thickness": 0.8,
                        "permittivity": 2.0,
                    },  # fmt: skip
                    {
                        "type": "Crystal Layer",
                        "material": "Calcite",
                        "thickness": 1.7,
                        "rotationY": 70,
                        "rotationZ": 40,
                    },  # fmt: skip
                    {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
                ],
            },
            backend="transfer",
        )
        return structure

    def _ours(self, structure, samples):
        layers = structure.layers[1:-1]
        eps = np.stack([np.squeeze(layer.eps_tensor) for layer in layers])
        thickness = np.array([layer.thickness * 1e4 for layer in layers])  # cm -> µm
        wavelength = float(np.squeeze(2 * np.pi / structure.k_0 * 1e4))
        eps_prism = float(structure.eps_prism)
        incidence = float(np.squeeze(np.arcsin(np.real(structure.k_x) / np.sqrt(eps_prism))))
        return bcm.stack_fields(
            eps, thickness, wavelength, incidence, eps_prism,
            float(structure.layers[-1].eps_exit), samples_per_layer=samples,
        )  # fmt: skip

    @pytest.mark.parametrize(("index", "polarization"), [(0, "p"), (1, "s")])
    def test_power(self, structure, index, polarization):
        ours = self._ours(structure, 0)
        reference = FieldProfile(structure)
        absorbed = [np.squeeze(a["absorptance"]) for a in reference.layer_absorption(polarization)]
        np.testing.assert_allclose(ours.absorption[:, index], absorbed, atol=1e-12)
        assert float(ours.R[index]) == pytest.approx(
            float(np.squeeze(reference.reflectance(polarization))), abs=1e-12
        )
        assert float(ours.T[index]) == pytest.approx(
            float(np.squeeze(reference.transmittance(polarization))), abs=1e-12
        )

    @pytest.mark.parametrize(("index", "polarization"), [(0, "p"), (1, "s")])
    def test_fields_against_depth(self, structure, index, polarization):
        ours = self._ours(structure, 50)
        profile = FieldProfile(structure).field_profile(polarization, n_points=50)
        n = ours.z_um.shape[-1]  # the NumPy profile continues into the exit medium
        np.testing.assert_allclose(ours.z_um, np.asarray(profile["z"])[:n], atol=1e-12)
        for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
            np.testing.assert_allclose(
                np.abs(ours.fields[:, component, index]),
                np.abs(np.squeeze(profile[name]))[:n],
                atol=1e-12,
                err_msg=name,
            )


class TestLossDensity:
    @staticmethod
    def _integrated(fields, eps, mu, k0, incident_flux, samples):
        """``½ ∫ (E†·ε_a·E + H†·μ_a·H) k₀ dz`` per layer, ``ε_a = (ε − ε†)/2i``."""
        z, f = np.asarray(fields.z_um), np.asarray(fields.fields)
        out = []
        for i in range(eps.shape[0]):
            window = slice(samples * i, samples * (i + 1))
            density = 0.0
            for tensor, rows in ((eps[i], slice(0, 3)), (mu[i], slice(3, 6))):
                tensor = np.asarray(tensor)
                anti = (tensor - tensor.conj().T) / 2j
                vec = f[window, rows, :]
                density = density + 0.5 * np.real(
                    np.einsum("zin,ij,zjn->zn", vec.conj(), anti, vec)
                )
            out.append(simpson(density, x=z[window] * k0, axis=0) / incident_flux)
        return np.array(out)

    def test_electric_loss(self):
        samples = 401
        fields = _fields(samples)
        kz = np.sqrt(2.25 - (1.5 * np.sin(0.6)) ** 2)
        integrated = self._integrated(
            fields, EPS, np.broadcast_to(np.eye(3), EPS.shape), 2 * np.pi / WAVELENGTH,
            0.5 * kz, samples,
        )  # fmt: skip
        np.testing.assert_allclose(integrated, fields.absorption, atol=1e-8)
        assert np.abs(fields.absorption[0]).max() < 1e-13  # the lossless spacer

    def test_magnetic_loss(self):
        samples = 401
        mu = jnp.stack([EYE, EYE, jnp.array([[1.2 + 0.3j, 0.1, 0], [0.1, 1.0 + 0.1j, 0],
                                             [0, 0, 1.1 + 0.2j]])])  # fmt: skip
        fields = _fields(samples, mu=mu)
        kz = np.sqrt(2.25 - (1.5 * np.sin(0.6)) ** 2)
        integrated = self._integrated(fields, EPS, mu, 2 * np.pi / WAVELENGTH, 0.5 * kz, samples)
        np.testing.assert_allclose(integrated, fields.absorption, atol=1e-8)


class TestRegimes:
    def test_lossless_flux_is_constant_through_each_layer(self):
        lossless = jnp.stack([2.0 * EYE, jnp.real(CALCITE) + 0j, 3.0 * EYE])
        fields = _fields(21, eps=lossless)
        ex, ey, hx, hy = (fields.fields[:, i, :] for i in (0, 1, 3, 4))
        flux = 0.5 * jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))
        np.testing.assert_allclose(np.ptp(np.asarray(flux), axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(fields.absorption, 0.0, atol=1e-12)

    def test_thick_absorbing_and_evanescent(self):
        """A 60 µm lossy calcite slab behind an evanescent gap, beyond the critical angle."""
        eps = jnp.stack([EYE, CALCITE])
        fields = bcm.stack_fields(eps, jnp.array([1.5, 60.0]), WAVELENGTH, 0.9, 5.76, 1.0)
        values = np.concatenate([np.ravel(v) for v in (fields.R, fields.T, fields.absorption)])
        assert np.all(np.isfinite(values))
        assert np.all(fields.absorption > -1e-12) and np.all(fields.R <= 1 + 1e-12)
        np.testing.assert_allclose(fields.T, 0.0, atol=1e-12)  # evanescent exit
        np.testing.assert_allclose(fields.R + fields.absorption.sum(-2), 1.0, atol=1e-12)

    def test_no_layers(self):
        fields = bcm.stack_fields(jnp.zeros((0, 3, 3)), jnp.zeros(0), 1.0, 0.3, 2.25, 1.0)
        assert fields.absorption.shape == (0, 2) and fields.interface_flux.shape == (1, 2)
        np.testing.assert_allclose(fields.R + fields.T, 1.0, atol=1e-14)

    def test_batches_and_jit(self):
        angles = jnp.array([0.2, 0.5, 0.8])
        run = jax.jit(bcm.stack_fields, static_argnames=("samples_per_layer", "doublings"))
        batched = run(EPS, THICKNESS, WAVELENGTH, angles, 2.25, 1.0, samples_per_layer=5)
        assert batched.absorption.shape == (3, 3, 2) and batched.fields.shape == (3, 15, 6, 2)
        single = _fields(5, incidence_rad=0.5)
        np.testing.assert_allclose(batched.fields[1], single.fields, atol=1e-12)


def test_gradients_match_finite_differences():
    def absorbed(params):
        thickness, wavelength = params
        eps = EPS.at[1].set(
            bcm.rotate_tensor(
                dispersion.at_wavelength("Calcite", wavelength),
                bcm.rotation_matrix(0.0, np.radians(-65.0), np.radians(20.0)),
            )
        )
        fields = bcm.stack_fields(eps, THICKNESS.at[1].set(thickness), wavelength, 0.6, 2.25, 1.0,
                                  samples_per_layer=3)  # fmt: skip
        return fields.absorption[1, 0] + 0.1 * jnp.abs(fields.fields[4, 2, 1]) ** 2

    params = jnp.array([1.7, WAVELENGTH])
    grad = np.asarray(jax.grad(absorbed)(params))
    for i, step in enumerate((1e-6, 1e-7)):
        e = jnp.zeros(2).at[i].set(step)
        fd = float((absorbed(params + e) - absorbed(params - e)) / (2 * step))
        assert grad[i] == pytest.approx(fd, rel=1e-5), i
