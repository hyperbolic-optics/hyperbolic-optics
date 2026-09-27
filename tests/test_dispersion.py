"""Tests for the JAX material models (``hyperbolic_optics.dispersion``).

Every built-in material must reproduce its NumPy class to rounding across its
default frequency window, and its frequency derivative must match finite
differences -- that derivative is what the NumPy classes cannot provide.
The Pockels tensor gives the textbook bulk index changes of LiNbO₃.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, dispersion  # noqa: E402
from hyperbolic_optics.materials import create_material  # noqa: E402


@pytest.mark.parametrize("name", dispersion.materials())
def test_matches_numpy_material(name):
    low, high = dispersion.frequency_range(name)
    frequencies = np.linspace(low, high, 23)
    material = create_material(name)
    expected = np.stack(
        [
            np.asarray(material.fetch_permittivity_tensor_for_freq(f)).reshape(3, 3)
            for f in frequencies
        ]
    )
    got = np.asarray(dispersion.permittivity(name, jnp.asarray(frequencies)))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("name", ["hBN", "MoO3", "GalliumOxide", "Calcite"])
def test_frequency_derivative(name):
    low, high = dispersion.frequency_range(name)
    w0 = 0.37 * low + 0.63 * high
    for i, j in ((0, 0), (0, 1), (2, 2)):

        def component(w, i=i, j=j):
            return dispersion.permittivity(name, w)[i, j]

        grad_re = float(jax.grad(lambda w: jnp.real(component(w)))(w0))
        grad_im = float(jax.grad(lambda w: jnp.imag(component(w)))(w0))
        h = 1e-6 * w0  # close to a phonon resonance a coarser step is not converged
        fd = (complex(component(w0 + h)) - complex(component(w0 - h))) / (2 * h)
        assert grad_re == pytest.approx(fd.real, rel=1e-6, abs=1e-10)
        assert grad_im == pytest.approx(fd.imag, rel=1e-6, abs=1e-10)


def test_hbn_is_hyperbolic_in_its_upper_band():
    """ε⊥ < 0 < ε∥ inside 1370–1610 cm⁻¹: the property this package is named for."""
    eps = np.asarray(dispersion.permittivity("hBN", jnp.array([1450.0, 1550.0])))
    assert (np.real(eps[:, 0, 0]) < 0).all() and (np.real(eps[:, 2, 2]) > 0).all()


def test_wavelength_form_and_shapes():
    w = jnp.array([[1400.0, 1500.0], [1450.0, 1600.0]])
    np.testing.assert_allclose(
        dispersion.at_wavelength("hBN", 1e4 / w), dispersion.permittivity("hBN", w), rtol=1e-14
    )
    assert dispersion.permittivity("MoO3", w).shape == (2, 2, 3, 3)


def test_unknown_material():
    with pytest.raises(ValueError, match="Unknown material"):
        dispersion.permittivity("unobtainium", 1000.0)


_LN = dispersion.at_wavelength("LiNbO3", 1.55)


def test_pockels_bulk_index_changes_are_the_textbook_ones():
    field = 1.0  # V/µm
    perturbed = dispersion.pockels_tensor(
        _LN, dispersion.LITHIUM_NIOBATE, jnp.array([0.0, 0.0, field])
    )
    ne, no = np.sqrt(np.real(_LN[2, 2])), np.sqrt(np.real(_LN[0, 0]))
    dne = np.sqrt(np.real(perturbed[2, 2])) - ne
    dno = np.sqrt(np.real(perturbed[0, 0])) - no
    assert dne == pytest.approx(-0.5 * ne**3 * 30.8e-6 * field, rel=2e-4)
    assert dno == pytest.approx(-0.5 * no**3 * 8.6e-6 * field, rel=2e-4)
    assert abs(perturbed[0, 1]) < 1e-15  # no off-diagonal term for a field along z


def test_pockels_the_3m_tensor_and_rotations():
    r = np.asarray(dispersion.LITHIUM_NIOBATE)
    np.testing.assert_array_equal(r, np.swapaxes(r, 0, 1))  # symmetric in (i, j)
    assert r[0, 0, 1] == -3.4 and r[1, 1, 1] == 3.4 and r[0, 2, 0] == 28.0
    rotation = bcm.rotation_matrix(0.3, 0.9, -0.4)
    field = jnp.array([0.4, -0.7, 1.1])
    lab = dispersion.pockels_tensor(_LN, dispersion.LITHIUM_NIOBATE, rotation @ field, rotation)
    crystal = dispersion.pockels_tensor(_LN, dispersion.LITHIUM_NIOBATE, field)
    np.testing.assert_allclose(lab, bcm.rotate_tensor(crystal, rotation), atol=1e-14)


def test_pockels_differentiable_in_the_field():
    def eps_zz(field_z):
        return jnp.real(
            dispersion.pockels_tensor(
                _LN, dispersion.LITHIUM_NIOBATE, jnp.array([0.0, 0.0, field_z])
            )[2, 2]
        )

    # dε/dE = −ε·(dη/dE)·ε = −ε_zz²·r₃₃ at zero field
    assert float(jax.grad(eps_zz)(0.0)) == pytest.approx(
        -(float(np.real(_LN[2, 2])) ** 2) * 30.8e-6, rel=1e-10
    )
