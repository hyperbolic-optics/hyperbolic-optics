"""Tests for the planar scattering-matrix contract (``bcm.stack_smatrix``, ``bcm.cascade``).

- the top-incidence block is :func:`bcm.solve_stack`'s r and t, with its R and T;
- a lossless stack's matrix is unitary, a lossy one's is passive;
- reciprocity takes the documented form, ``s(kx) = D s(−kx)ᵀ D``;
- evanescent ports are flagged, and total internal reflection is lossless;
- cascading two stacks equals the stack of both, including through an
  evanescent gap (frustrated total internal reflection), against the closed
  form for a single isotropic gap;
- gradients match finite differences.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm  # noqa: E402
from tests._helpers import uniaxial as _uniaxial  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
LN = _uniaxial(5.46, 5.06, 0.7, 0.4)  # every off-diagonal component nonzero
EPS = jnp.stack([LN, 2.1 * EYE])
THICKNESS = jnp.array([0.4, 0.2])
D = np.diag([-1.0, 1.0, -1.0, 1.0])


def _s(eps=EPS, thickness=THICKNESS, kx=0.479, top=1.0, bottom=2.25, wavelength=0.8):
    return bcm.stack_smatrix(eps, thickness, wavelength, kx, top, bottom)


class TestContract:
    def test_top_incidence_is_solve_stack(self):
        theta = 0.5
        smatrix = _s(kx=np.sin(theta))
        planar = bcm.solve_stack(EPS, THICKNESS, 0.8, theta, 1.0, 2.25)
        s = np.asarray(smatrix.s)
        np.testing.assert_allclose(s[:2, :2], planar.r, atol=1e-14)
        np.testing.assert_allclose(np.abs(s[:2, :2]) ** 2, planar.R, atol=1e-14)
        np.testing.assert_allclose(np.abs(s[2:, :2]) ** 2, planar.T, atol=1e-14)
        assert bcm.PORTS == ("top@p", "top@s", "bottom@p", "bottom@s")

    def test_lossless_is_unitary(self):
        s = np.asarray(_s().s)
        np.testing.assert_allclose(s.conj().T @ s, np.eye(4), atol=1e-13)

    def test_lossy_is_passive(self):
        lossy = EPS.at[0].add(0.3j * EYE)
        s = np.asarray(_s(eps=lossy).s)
        assert np.linalg.eigvalsh(np.eye(4) - s.conj().T @ s).min() > 0

    def test_reciprocity(self):
        forward, backward = np.asarray(_s(kx=0.479).s), np.asarray(_s(kx=-0.479).s)
        np.testing.assert_allclose(forward, D @ backward.T @ D, atol=1e-13)

    def test_nonreciprocal_media_break_it(self):
        gyrotropic = EPS.at[1].set(jnp.array([[2.1, 0.4j, 0], [-0.4j, 2.1, 0], [0, 0, 2.1]]))
        forward, backward = (
            np.asarray(_s(eps=gyrotropic).s),
            np.asarray(_s(eps=gyrotropic, kx=-0.479).s),
        )
        assert np.abs(forward - D @ backward.T @ D).max() > 1e-3

    def test_total_internal_reflection(self):
        smatrix = _s(kx=1.3, top=2.25, bottom=1.0)
        np.testing.assert_array_equal(smatrix.propagating, [True, True, False, False])
        top = np.asarray(smatrix.s)[:2, :2]
        np.testing.assert_allclose(top.conj().T @ top, np.eye(2), atol=1e-13)

    def test_grazing_cutoff_is_finite(self):
        """kx exactly at the top medium's index: that port carries no power, and no NaN."""
        smatrix = _s(kx=1.5, top=2.25, bottom=1.0)
        assert np.all(np.isfinite(np.asarray(smatrix.s)))
        np.testing.assert_array_equal(smatrix.propagating, [False, False, False, False])

    def test_batches(self):
        wavelengths = jnp.array([0.7, 0.8, 0.9])
        batched = bcm.stack_smatrix(EPS, THICKNESS, wavelengths, 0.3, 1.0, 2.25)
        assert batched.s.shape == (3, 4, 4) and batched.propagating.shape == (3, 4)
        np.testing.assert_allclose(batched.s[1], _s(kx=0.3).s, atol=1e-14)


class TestCascade:
    def test_two_stacks_are_the_stack_of_both(self):
        upper = bcm.stack_smatrix(EPS[:1], THICKNESS[:1], 0.8, 0.4, 1.0, 2.1)
        lower = bcm.stack_smatrix(EPS[1:], THICKNESS[1:], 0.8, 0.4, 2.1, 2.25)
        # the shared half-space has the second layer's ε, so it may be
        # dropped from the lower stack's own list without changing anything
        whole = bcm.stack_smatrix(EPS, THICKNESS, 0.8, 0.4, 1.0, 2.25)
        np.testing.assert_allclose(bcm.cascade(upper, lower).s, whole.s, atol=1e-13)

    def test_three_stacks_associate(self):
        a = bcm.stack_smatrix(EPS[:1], THICKNESS[:1], 0.8, 0.4, 1.0, 1.5)
        b = bcm.stack_smatrix(jnp.stack([3.0 * EYE]), jnp.array([0.3]), 0.8, 0.4, 1.5, 1.5)
        c = bcm.stack_smatrix(EPS[1:], THICKNESS[1:], 0.8, 0.4, 1.5, 2.25)
        left = bcm.cascade(bcm.cascade(a, b), c)
        right = bcm.cascade(a, bcm.cascade(b, c))
        np.testing.assert_allclose(left.s, right.s, atol=1e-13)

    def test_frustrated_total_internal_reflection(self):
        """glass | air gap (d) | glass, beyond the critical angle, assembled from pieces."""
        eps_glass, gap, wavelength, kx = 2.25, 0.3, 1.0, 1.2  # kx > 1: evanescent in air
        upper = bcm.stack_smatrix(
            jnp.zeros((0, 3, 3)), jnp.zeros(0), wavelength, kx, eps_glass, 1.0
        )
        middle = bcm.stack_smatrix(jnp.stack([EYE]), jnp.array([gap]), wavelength, kx, 1.0, 1.0)
        lower = bcm.stack_smatrix(
            jnp.zeros((0, 3, 3)), jnp.zeros(0), wavelength, kx, 1.0, eps_glass
        )
        tunnel = bcm.cascade(bcm.cascade(upper, middle), lower)
        assert not bool(middle.propagating.any())
        np.testing.assert_array_equal(tunnel.propagating, [True] * 4)

        # s polarization, closed form: t = 4·k1·κ... via the Airy sum with kz2 = iκ
        k0 = 2 * np.pi / wavelength
        kz1, kz2 = np.sqrt(eps_glass - kx**2 + 0j), np.sqrt(1.0 - kx**2 + 0j)
        r12 = (kz1 - kz2) / (kz1 + kz2)
        phase = np.exp(1j * kz2 * k0 * gap)
        t_s = (1 - r12**2) * phase / (1 - r12**2 * phase**2)
        assert abs(complex(tunnel.s[3, 1])) ** 2 == pytest.approx(abs(t_s) ** 2, rel=1e-10)
        s = np.asarray(tunnel.s)
        np.testing.assert_allclose(s.conj().T @ s, np.eye(4), atol=1e-12)


def test_gradient_matches_finite_differences():
    def cross_coupling(thickness):
        s = bcm.stack_smatrix(EPS, jnp.array([thickness, 0.2]), 0.8, 0.4, 1.0, 2.25).s
        return jnp.abs(s[3, 0]) ** 2 + jnp.real(s[0, 2] * jnp.conj(s[2, 1]))

    grad = float(jax.grad(cross_coupling)(0.4))
    fd = float((cross_coupling(0.4 + 1e-6) - cross_coupling(0.4 - 1e-6)) / 2e-6)
    assert grad == pytest.approx(fd, rel=1e-6)
