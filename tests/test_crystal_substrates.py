"""Tests for stacks on semi-infinite anisotropic substrates (``bcm.solve_stack_on_crystal``).

- the NumPy engine's ``Semi Infinite Anisotropic Layer``, fed the same tensors:
  hyperbolic lossy calcite behind an Otto gap, lossless LiNbO₃ under a film;
- an isotropic "crystal" is :func:`bcm.solve_stack`'s isotropic exit;
- a lossless stack on a lossless crystal conserves energy (R + T = 1), and a
  lossy crystal absorbs;
- degenerate substrate modes (the optic axis along z at normal incidence),
  where eigenvector-based methods fail, are exact, with finite gradients;
- gradients match finite differences.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, dispersion  # noqa: E402
from hyperbolic_optics.scattering import scattering_coefficients  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402

EYE = jnp.eye(3, dtype=complex)


def test_zero_trace_hyperbolic_substrate_has_a_regular_limit():
    def solve(zz):
        return bcm.solve_stack_on_crystal(
            jnp.zeros((0, 3, 3), complex),
            jnp.zeros(0),
            1.55,
            0.3,
            1.0,
            jnp.diag(jnp.array([2.0, 2.0, zz], complex)),
        )

    result = solve(-4.0)
    expected = (solve(-4.000001).r + solve(-3.999999).r) / 2
    np.testing.assert_allclose(result.r, expected, atol=1e-12)
    np.testing.assert_allclose(result.R.sum(-2) + result.T, 1.0, atol=1e-12)
    assert np.isfinite(jax.grad(lambda zz: solve(zz).R.sum())(-4.0))


def _against_engine(scenario, layers, substrate, atol):
    structure = Structure()
    structure.execute(
        {
            "ScenarioData": scenario,
            "Layers": [{"type": "Ambient Incident Layer", "permittivity": 5.5}, *layers, substrate],
        },
        backend="scattering",
    )
    middle = structure.layers[1:-1]
    batch = np.broadcast_shapes(
        *(np.shape(layer.eps_tensor)[:-2] for layer in structure.layers[1:])
    )
    eps = np.stack([np.broadcast_to(layer.eps_tensor, batch + (3, 3)) for layer in middle], -3)
    thickness = np.array([layer.thickness * 1e4 for layer in middle])
    crystal = np.broadcast_to(structure.layers[-1].eps_tensor, batch + (3, 3))
    incidence = np.arcsin(np.real(structure.k_x) / np.sqrt(5.5))
    wavelength = 2 * np.pi / structure.k_0 * 1e4
    result = bcm.solve_stack_on_crystal(eps, thickness, wavelength, incidence, 5.5, crystal)
    reference = scattering_coefficients(structure.layers, structure.k_0)
    for pair, (out, inc) in {"pp": (0, 0), "ps": (1, 0), "sp": (0, 1), "ss": (1, 1)}.items():
        want = np.asarray(reference[f"r_{pair}"])
        got = np.broadcast_to(np.asarray(result.r[..., out, inc]), want.shape)
        np.testing.assert_allclose(got, want, atol=atol, err_msg=f"r_{pair}")
    return result


def test_hyperbolic_calcite_behind_an_otto_gap():
    result = _against_engine(
        {"type": "Incident", "incidentAngle": [10.0, 30.0, 50.0, 70.0], "frequency": 1470.0},
        [{"type": "Isotropic Middle-Stack Layer", "thickness": 1.0, "permittivity": 1.0}],
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationY": 65,
            "rotationZ": 30,
        },  # fmt: skip
        atol=1e-12,
    )
    absorbed = 1 - np.asarray(result.R).sum(-2) - np.asarray(result.T)
    np.testing.assert_allclose(absorbed, 0.0, atol=1e-12)  # a lossless gap: T is the absorption
    assert np.all(np.asarray(result.T) > -1e-12)


def test_lossless_lithium_niobate_under_a_film():
    result = _against_engine(
        {"type": "Incident", "incidentAngle": [10.0, 30.0, 50.0], "frequency": 1e4 / 1.55},
        [{"type": "Isotropic Middle-Stack Layer", "thickness": 0.3, "permittivity": 2.1}],
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "LiNbO3",
            "rotationY": 90,
            "rotationZ": 40,
        },  # fmt: skip
        atol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(result.R).sum(-2) + result.T, 1.0, atol=1e-12)


def test_an_isotropic_crystal_is_the_isotropic_exit():
    eps = jnp.stack([jnp.diag(jnp.array([4.6, 5.0, 4.2], dtype=complex))])
    crystal = bcm.solve_stack_on_crystal(eps, jnp.array([0.3]), 1.0, 0.4, 1.0, 2.25 * EYE)
    planar = bcm.solve_stack(eps, jnp.array([0.3]), 1.0, 0.4, 1.0, 2.25)
    np.testing.assert_allclose(crystal.r, planar.r, atol=1e-13)
    np.testing.assert_allclose(crystal.T, np.asarray(planar.T).sum(-2), atol=1e-13)


def test_degenerate_modes_and_their_gradients():
    """Optic axis along z at normal incidence: the two ordinary waves are degenerate."""
    eps_o, eps_e = 2.28, 2.13

    def reflectance(tilt):
        axis = jnp.array([jnp.sin(tilt), 0.0, jnp.cos(tilt)])
        crystal = (eps_o * jnp.eye(3) + (eps_e - eps_o) * jnp.outer(axis, axis)).astype(complex)
        result = bcm.solve_stack_on_crystal(
            jnp.zeros((0, 3, 3)), jnp.zeros(0), 1.0, 0.0, 1.0, crystal
        )
        return result.R[0, 0]

    n = np.sqrt(eps_o)
    assert float(reflectance(0.0)) == pytest.approx(((n - 1) / (n + 1)) ** 2, rel=1e-12)
    grad = jax.grad(reflectance)(0.0)
    assert np.isfinite(float(grad)) and abs(float(grad)) < 1e-8  # even in the tilt
    tilted = float(jax.grad(reflectance)(0.3))
    fd = float((reflectance(0.3 + 1e-6) - reflectance(0.3 - 1e-6)) / 2e-6)
    assert tilted == pytest.approx(fd, rel=1e-6)


def test_gradients_through_a_dispersive_hyperbolic_substrate():
    rotation = bcm.rotation_matrix(0.0, np.radians(65.0), np.radians(30.0))

    def conversion(wavenumber):
        calcite = bcm.rotate_tensor(dispersion.permittivity("Calcite", wavenumber), rotation)
        result = bcm.solve_stack_on_crystal(
            jnp.stack([EYE]), jnp.array([1.0]), 1e4 / wavenumber, 0.6, 5.5, calcite
        )
        return result.R[1, 0]

    grad = float(jax.grad(conversion)(1470.0))
    fd = float((conversion(1470.0 + 1e-3) - conversion(1470.0 - 1e-3)) / 2e-3)
    assert grad == pytest.approx(fd, rel=1e-5)
