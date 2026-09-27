"""Tests for Gaussian beams on planar stacks (``hyperbolic_optics.beams``).

- at total internal reflection the reflected beam's centroid moves by Artmann's
  Goos–Hänchen shift, ``Δx = −dφ/dq_x``, from the plane-wave phase alone;
- the transverse (Imbert–Fedorov) shift vanishes for linear polarization and
  changes sign with the helicity of a circular one.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, beams  # noqa: E402

NONE = (jnp.zeros((0, 3, 3)), jnp.zeros(0))
N_GLASS, WAVELENGTH, ANGLE = 1.5, 1.0, np.radians(50.0)  # beyond the critical angle


@pytest.mark.parametrize("polarization", [0, 1], ids=["p", "s"])
def test_goos_hanchen_shift_is_artmanns(polarization):
    k0 = 2 * np.pi / WAVELENGTH
    q0 = N_GLASS * np.sin(ANGLE)

    def phase(q):
        s = bcm.stack_smatrix(*NONE, WAVELENGTH, q, N_GLASS**2, 1.0).s
        return np.angle(complex(s[polarization, polarization]))

    artmann = -(phase(q0 + 1e-6) - phase(q0 - 1e-6)) / 2e-6 / k0
    jones = (1.0, 0.0) if polarization == 0 else (0.0, 1.0)
    waist = 20.0
    x, y, field = beams.reflected_beam(*NONE, WAVELENGTH, ANGLE, N_GLASS**2, 1.0, waist, jones)
    dx, dy = beams.beam_shift(x, y, field)
    assert float(dx) == pytest.approx(artmann, rel=2e-3)
    assert abs(float(dy)) < 1e-12


def test_imbert_fedorov_shift_follows_the_helicity():
    shifts = []
    for helicity in (1, -1):
        jones = (1 / np.sqrt(2), helicity * 1j / np.sqrt(2))
        x, y, field = beams.reflected_beam(*NONE, WAVELENGTH, ANGLE, N_GLASS**2, 1.0, 10.0, jones)
        shifts.append(float(beams.beam_shift(x, y, field)[1]))
    assert abs(shifts[0]) > 1e-3 * WAVELENGTH
    assert shifts[0] == pytest.approx(-shifts[1], rel=1e-6)
