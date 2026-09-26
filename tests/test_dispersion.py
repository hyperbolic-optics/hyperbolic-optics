"""Tests for the JAX material models (``hyperbolic_optics.dispersion``).

Every built-in material must reproduce its NumPy class to rounding across its
default frequency window, and its frequency derivative must match finite
differences -- that derivative is what the NumPy classes cannot provide.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import dispersion  # noqa: E402
from hyperbolic_optics.materials import create_material  # noqa: E402


@pytest.mark.parametrize("name", dispersion.materials())
def test_matches_numpy_material(name):
    low, high = dispersion.reststrahlen_range(name)
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
    low, high = dispersion.reststrahlen_range(name)
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
