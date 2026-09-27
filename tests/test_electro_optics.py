"""Tests for the Pockels effect (``hyperbolic_optics.electro_optics``).

- bulk LiNbO₃ with a field along the optic axis: the textbook
  ``Δn_e = −½n_e³r₃₃E`` and ``Δn_o = −½n_o³r₁₃E``;
- the 3m tensor has the crystal's symmetry, and rotating the crystal and the
  field together changes nothing;
- a guided mode's first-order index shift matches the finite difference of
  exact mode roots;
- the tensor is differentiable in the field, with the analytic derivative.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, dispersion, resonances  # noqa: E402
from hyperbolic_optics import electro_optics as eo  # noqa: E402

WAVELENGTH = 1.55
LN = dispersion.at_wavelength("LiNbO3", WAVELENGTH)


def test_bulk_index_changes_are_the_textbook_ones():
    field = 1.0  # V/µm
    perturbed = eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, jnp.array([0.0, 0.0, field]))
    ne, no = np.sqrt(np.real(LN[2, 2])), np.sqrt(np.real(LN[0, 0]))
    dne = np.sqrt(np.real(perturbed[2, 2])) - ne
    dno = np.sqrt(np.real(perturbed[0, 0])) - no
    assert dne == pytest.approx(-0.5 * ne**3 * 30.8e-6 * field, rel=2e-4)
    assert dno == pytest.approx(-0.5 * no**3 * 8.6e-6 * field, rel=2e-4)
    assert abs(perturbed[0, 1]) < 1e-15  # no off-diagonal term for a field along z


def test_the_3m_tensor_and_rotations():
    r = np.asarray(eo.LITHIUM_NIOBATE)
    np.testing.assert_array_equal(r, np.swapaxes(r, 0, 1))  # symmetric in (i, j)
    assert r[0, 0, 1] == -3.4 and r[1, 1, 1] == 3.4 and r[0, 2, 0] == 28.0
    rotation = bcm.rotation_matrix(0.3, 0.9, -0.4)
    field = jnp.array([0.4, -0.7, 1.1])
    lab = eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, rotation @ field, rotation)
    crystal = eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, field)
    np.testing.assert_allclose(lab, bcm.rotate_tensor(crystal, rotation), atol=1e-14)


def test_mode_index_shift_against_exact_roots():
    """An x-cut film, optic axis along y: the TE mode's Ey sees r₃₃."""
    rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(90.0))
    eps = bcm.rotate_tensor(LN, rotation)
    thickness, cladding = jnp.array([0.6]), 1.444**2
    mode = resonances.slab_modes(eps[None], thickness, WAVELENGTH, 1.0, cladding)[0]
    z, fields = resonances.slab_mode_fields(
        eps[None], thickness, WAVELENGTH, 1.0, cladding, mode.effective_index,
        samples_per_layer=300, cladding_um=(3.0, 3.0), cladding_samples=900,
    )  # fmt: skip
    field = jnp.array([0.0, 1.0, 0.0])
    delta = eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, field, rotation) - eps
    shift = complex(eo.mode_index_shift(z, fields, thickness, delta[None]))

    def index(sign):
        tensor = eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, sign * field, rotation)
        return resonances.slab_modes(tensor[None], thickness, WAVELENGTH, 1.0, cladding)[0]

    exact = (index(1.0).effective_index - index(-1.0).effective_index) / 2
    assert shift.real == pytest.approx(exact.real, rel=5e-4)
    assert abs(shift.imag) < 1e-12
    assert eo.vpi_length(shift.real / 5.0, WAVELENGTH) == pytest.approx(
        WAVELENGTH * 1e-4 * 5.0 / (2 * abs(shift.real))
    )


def test_differentiable_in_the_field():
    def eps_zz(field_z):
        return jnp.real(
            eo.pockels_tensor(LN, eo.LITHIUM_NIOBATE, jnp.array([0.0, 0.0, field_z]))[2, 2]
        )

    # dε/dE = −ε·(dη/dE)·ε = −ε_zz²·r₃₃ at zero field
    assert float(jax.grad(eps_zz)(0.0)) == pytest.approx(
        -(float(np.real(LN[2, 2])) ** 2) * 30.8e-6, rel=1e-10
    )
