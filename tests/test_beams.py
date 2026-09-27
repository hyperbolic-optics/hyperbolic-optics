"""Tests for Gaussian beams on planar stacks (``hyperbolic_optics.beams``).

- a wide beam is a plane wave: ``S_beam → S(q₀)``, the difference falling as 1/w₀²;
- at total internal reflection the reflected beam's centroid moves by Artmann's
  Goos–Hänchen shift, ``Δx = −dφ/dq_x``, from the plane-wave phase alone;
- a shifted Gaussian couples into the unshifted mode by ``exp(−Δ²cos²θ/2w₀²)``;
- the transverse (Imbert–Fedorov) shift vanishes for linear polarization and
  changes sign with the helicity of a circular one;
- the beam S-matrix is passive.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, beams  # noqa: E402

LN = bcm.rotate_tensor(
    jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex)), bcm.rotation_matrix(0.0, 0.7, 0.4)
)
FILM = (jnp.stack([LN]), jnp.array([0.4]))
NONE = (jnp.zeros((0, 3, 3)), jnp.zeros(0))
N_GLASS, WAVELENGTH, ANGLE = 1.5, 1.0, np.radians(50.0)  # beyond the critical angle


def test_a_wide_beam_is_a_plane_wave():
    plane = np.asarray(bcm.stack_smatrix(*FILM, 0.8, np.sin(0.5), 1.0, 2.25).s)
    errors = [
        np.abs(np.asarray(beams.beam_smatrix(*FILM, 0.8, 0.5, 1.0, 2.25, waist)) - plane).max()
        for waist in (50.0, 500.0)
    ]
    assert errors[1] < 1e-6
    assert errors[0] / errors[1] == pytest.approx(100.0, rel=0.05)  # ∝ 1/w₀²


def test_the_beam_smatrix_is_passive():
    s = np.asarray(beams.beam_smatrix(*FILM, 0.8, 0.5, 1.0, 2.25, 5.0))
    assert np.linalg.eigvalsh(np.eye(4) - s.conj().T @ s).min() > -1e-12


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

    # the shift is why the specular beam mode is not reached in full
    s = np.asarray(beams.beam_smatrix(*NONE, WAVELENGTH, ANGLE, N_GLASS**2, 1.0, waist, samples=41))
    predicted = np.exp(-((artmann * np.cos(ANGLE)) ** 2) / (2 * waist**2))
    assert abs(s[polarization, polarization]) == pytest.approx(predicted, abs=1e-4)


def test_imbert_fedorov_shift_follows_the_helicity():
    shifts = []
    for helicity in (1, -1):
        jones = (1 / np.sqrt(2), helicity * 1j / np.sqrt(2))
        x, y, field = beams.reflected_beam(*NONE, WAVELENGTH, ANGLE, N_GLASS**2, 1.0, 10.0, jones)
        shifts.append(float(beams.beam_shift(x, y, field)[1]))
    assert abs(shifts[0]) > 1e-3 * WAVELENGTH
    assert shifts[0] == pytest.approx(-shifts[1], rel=1e-6)
