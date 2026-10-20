"""Tests for the differentiable 1D-grating RCWA (``hyperbolic_optics.rcwa``).

The anchors, none of which share code with the grating solver's own results:

- uniform layers must reproduce the planar solver exactly, at any truncation
  and for conical incidence;
- the doubling engine must agree with a matrix exponential (SciPy) of the same
  Fourier-space operator on a patterned layer;
- lossless gratings must conserve energy across all orders, symmetric ones must
  diffract symmetrically, and Li's factorisation must converge faster than
  Laurent's for TM light;
- gradients, including with respect to the geometry, must match finite
  differences.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import scipy.linalg  # noqa: E402

from hyperbolic_optics import bcm, rcwa  # noqa: E402
from hyperbolic_optics.bcm import _interface, _layer, _star  # noqa: E402
from hyperbolic_optics.rcwa import (  # noqa: E402
    _factorize,
    _fourier_system_matrix,
    _harmonic_modes,
)
from tests._helpers import total as _total  # noqa: E402
from tests._helpers import uniaxial as _uniaxial  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
LN_TILTED = _uniaxial(5.46, 5.06, 0.7, 0.4)  # every off-diagonal component nonzero
SILICON = 3.5**2 * EYE
GLASS = 1.45**2


# ---------------------------------------------------------------------------
# Fourier coefficients
# ---------------------------------------------------------------------------


class TestFourierCoefficients:
    def test_against_quadrature(self):
        boundaries = jnp.array([0.1, 0.35, 0.8, 1.1])
        values = jnp.array([2.0 + 0.5j, -1.0, 4.0])
        coefficients = rcwa.fourier_coefficients(values, boundaries, 6)

        x = np.linspace(0.1, 1.1, 400_001)[:-1] + 0.5 / 400_000  # midpoints
        f = np.select(
            [x < 0.35, x < 0.8], [complex(values[0]), complex(values[1])], complex(values[2])
        )
        for index, m in enumerate(range(-6, 7)):
            quadrature = np.mean(f * np.exp(-2j * np.pi * m * x))
            assert complex(coefficients[index]) == pytest.approx(quadrature, abs=1e-9)

    def test_differentiable_in_the_boundaries(self):
        """d c_m / d b = −(jump in f) · exp(−2πi m b): the edge is not pixelated."""
        values = jnp.array([3.0, 1.0])

        def coefficient(edge, m):
            boundaries = jnp.stack([jnp.asarray(0.0), edge, jnp.asarray(1.0)])
            return rcwa.fourier_coefficients(values, boundaries, 3)[m + 3]

        for m in (-2, 0, 1, 3):
            grad = jax.grad(lambda b, m=m: jnp.real(coefficient(b, m)))(0.4)
            expected = np.real((3.0 - 1.0) * np.exp(-2j * np.pi * m * 0.4))
            assert float(grad) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_single_harmonic_is_the_planar_berreman_matrix(self):
        rng = np.random.default_rng(1)
        eps = jnp.asarray(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) + 4 * np.eye(3))
        mu = jnp.asarray(0.2 * rng.normal(size=(3, 3)) + np.eye(3) + 0j)
        boundaries = jnp.array([0.0, 1.0])
        fourier = _fourier_system_matrix(
            _factorize(eps[None], boundaries, 0, "li"),
            _factorize(mu[None], boundaries, 0, "li"),
            jnp.array([0.6]),
            0.0,
        )
        np.testing.assert_allclose(fourier, bcm.system_matrix(eps, mu, 0.6), atol=1e-14)

    def test_uniform_isotropic_dispersion_with_conical_wavevector(self):
        """Eigenvalues of Δ for each harmonic are ±√(ε − kx_m² − ky²), twice."""
        eps, kx, ky = 2.3, jnp.array([-1.4, -0.2, 1.0]), 0.35
        boundaries = jnp.array([0.0, 1.0])
        delta = _fourier_system_matrix(
            _factorize((eps * EYE)[None], boundaries, 1, "li"),
            _factorize(EYE[None], boundaries, 1, "li"),
            kx,
            ky,
        )
        got = np.sort_complex(np.linalg.eigvals(np.asarray(delta)))
        kz = np.sqrt(eps - np.asarray(kx) ** 2 - ky**2 + 0j)
        want = np.sort_complex(np.concatenate([kz, kz, -kz, -kz]))
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_uniform_factorization_is_the_tensor(self):
        """Li's rearrangement is an identity when nothing varies in x."""
        blocks = _factorize(LN_TILTED[None], jnp.array([0.0, 1.0]), 2, "li")
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(blocks[i, j], LN_TILTED[i, j] * np.eye(5), atol=1e-14)

    def test_reference_and_ambient_modes_are_power_normalised(self):
        qx, qy = jnp.array([-2.0, -0.3, 0.9]), 0.4
        k_par_sq = np.asarray(qx) ** 2 + qy**2
        _, flux = _harmonic_modes(jnp.asarray(k_par_sq) + 1.0, qx, qy, (1.0, 0.0))
        np.testing.assert_allclose(flux, np.repeat([0.5, 0.5, -0.5, -0.5], 3), atol=1e-14)
        _, flux = _harmonic_modes(2.25, qx, qy, (1.0, 0.0))
        kz = np.real(np.sqrt(2.25 - k_par_sq + 0j))
        np.testing.assert_allclose(flux, 0.5 * np.concatenate([kz, kz, -kz, -kz]), atol=1e-14)


# ---------------------------------------------------------------------------
# Reduction to the planar solver
# ---------------------------------------------------------------------------


STACK_EPS = jnp.stack([2.0 * EYE, _uniaxial(5.46, 5.06 + 0.01j, 0.7, 0.4), 2.25 * EYE])
STACK_D = jnp.array([0.1, 0.8, 0.2])


class TestUniformReduction:
    @pytest.mark.parametrize("max_order", [0, 3])
    def test_matches_planar_solver(self, max_order):
        layers = [rcwa.uniform_layer(STACK_EPS[i], STACK_D[i]) for i in range(3)]
        grating = rcwa.solve_grating(layers, 0.5, 0.633, 0.5, 1.0, 2.25, max_order=max_order)
        planar = bcm.solve_stack(STACK_EPS, STACK_D, 0.633, 0.5, 1.0, 2.25)
        np.testing.assert_allclose(grating.r[max_order], planar.r, atol=1e-13)
        np.testing.assert_allclose(grating.t[max_order], planar.t, atol=1e-13)
        others = np.delete(np.asarray(grating.r), max_order, axis=0)
        np.testing.assert_allclose(others, 0.0, atol=1e-14)

    def test_conical_incidence_is_a_rotated_planar_problem(self):
        """A plane of incidence at azimuth φ is the planar case with ε rotated by −φ."""
        phi = 0.6
        layers = [rcwa.uniform_layer(STACK_EPS[i], STACK_D[i]) for i in range(3)]
        grating = rcwa.solve_grating(layers, 0.5, 0.633, 0.5, 1.0, 2.25, 2, azimuth_rad=phi)
        rotated = bcm.rotate_tensor(STACK_EPS, bcm.rotation_matrix(0.0, 0.0, -phi))
        planar = bcm.solve_stack(rotated, STACK_D, 0.633, 0.5, 1.0, 2.25)
        np.testing.assert_allclose(grating.r[2], planar.r, atol=1e-13)
        np.testing.assert_allclose(grating.t[2], planar.t, atol=1e-13)

    def test_magnetic_uniform_layer(self):
        mu = jnp.array([[1.2, 0.3j, 0.0], [-0.3j, 1.2, 0.0], [0.0, 0.0, 1.0]])
        eps = _uniaxial(3.0, 2.5, 0.3, 0.2)
        grating = rcwa.solve_grating([rcwa.uniform_layer(eps, 1.1, mu)], 0.7, 1.0, 0.4, 2.0, 1.5, 2)
        planar = bcm.solve_stack(eps[None], jnp.array([1.1]), 1.0, 0.4, 2.0, 1.5, mu=mu[None])
        np.testing.assert_allclose(grating.r[2], planar.r, atol=1e-13)
        np.testing.assert_allclose(grating.t[2], planar.t, atol=1e-13)


# ---------------------------------------------------------------------------
# The doubling engine on a patterned layer
# ---------------------------------------------------------------------------


def test_doubling_matches_matrix_exponential_on_a_patterned_layer():
    """Same Fourier operator, independent propagation: SciPy's expm, then T → S."""
    max_order, size = 4, 9
    wavelength, period, theta, thickness = 1.0, 0.8, 0.3, 0.15
    layer = rcwa.lamellar_layer(LN_TILTED, EYE, 0.4, thickness)
    k_par = np.sin(theta)
    orders = np.arange(-max_order, max_order + 1)
    kx = jnp.asarray(k_par + orders * wavelength / period)
    direction = (jnp.ones(1), jnp.zeros(1))

    w_ref, _ = _harmonic_modes(kx**2 + 1.0, kx, 0.0, direction)
    delta = _fourier_system_matrix(
        _factorize(layer.eps, layer.boundaries, max_order, "li"),
        _factorize(jnp.stack([EYE, EYE]), layer.boundaries, max_order, "li"),
        kx,
        0.0,
    )
    delta_ref = np.linalg.solve(np.asarray(w_ref), np.asarray(delta @ w_ref))
    k0d = 2 * np.pi / wavelength * thickness

    # Transfer matrix over the whole (thin) layer, converted to a scattering matrix.
    transfer = scipy.linalg.expm(1j * k0d * delta_ref)
    t11, t12 = transfer[: 2 * size, : 2 * size], transfer[: 2 * size, 2 * size :]
    t21, t22 = transfer[2 * size :, : 2 * size], transfer[2 * size :, 2 * size :]
    inv22 = np.linalg.inv(t22)
    expected = (t11 - t12 @ inv22 @ t21, t12 @ inv22, -inv22 @ t21, inv22)

    got = _layer(jnp.asarray(delta_ref), jnp.asarray(k0d), bcm.DEFAULT_DOUBLINGS)
    for block_got, block_want in zip(got, expected, strict=True):
        np.testing.assert_allclose(np.asarray(block_got), block_want, atol=1e-11)
    # and the same interface/star machinery the solver uses is exercised
    assert _star(_interface(w_ref, w_ref), got)[0].shape == (2 * size, 2 * size)


# ---------------------------------------------------------------------------
# Physics of patterned layers
# ---------------------------------------------------------------------------


class TestGratingPhysics:
    @pytest.mark.parametrize(
        "ridge,azimuth",
        [(SILICON, 0.0), (SILICON, 0.7), (_uniaxial(5.46, 5.06, 0.7, 0.4), 0.3)],
        ids=["silicon", "silicon_conical", "tilted_ln_conical"],
    )
    def test_lossless_gratings_conserve_energy(self, ridge, azimuth):
        layers = [rcwa.lamellar_layer(ridge, EYE, 0.4, 0.35), rcwa.uniform_layer(2.1 * EYE, 0.1)]
        result = rcwa.solve_grating(layers, 0.9, 1.0, 0.35, 1.0, GLASS, 12, azimuth_rad=azimuth)
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-11)
        assert np.asarray(result.T[11:14].sum()) > 0.05, "power should reach several orders"

    def test_energy_holds_at_high_truncation(self):
        """161 orders: the default doubling count scales with max_order (see default_doublings)."""
        layers = [rcwa.lamellar_layer(SILICON, EYE, 0.4, 0.35)]
        result = rcwa.solve_grating(layers, 0.9, 1.0, 0.5, 1.0, GLASS, 80, azimuth_rad=0.7)
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-8)
        assert rcwa.default_doublings(80) == 28

    def test_absorbing_grating_is_passive(self):
        gold = (-25.0 + 1.5j) * EYE
        result = rcwa.solve_grating(
            [rcwa.lamellar_layer(gold, EYE, 0.5, 0.1)], 0.6, 0.8, 0.2, 1.0, GLASS, 15
        )
        total = _total(result)
        assert (total < 1.0).all() and (total > 0.5).all()
        assert (np.asarray(result.R) >= -1e-14).all() and (np.asarray(result.T) >= -1e-14).all()

    def test_symmetric_grating_diffracts_symmetrically_at_normal_incidence(self):
        result = rcwa.solve_grating(
            [rcwa.lamellar_layer(SILICON, EYE, 0.3, 0.4, offset=0.35)],
            1.3,
            1.0,
            0.0,
            1.0,
            GLASS,
            10,
        )
        R, T = np.asarray(result.R).sum(axis=-2), np.asarray(result.T).sum(axis=-2)
        np.testing.assert_allclose(R, R[::-1], atol=1e-12)
        np.testing.assert_allclose(T, T[::-1], atol=1e-12)
        assert T[11].max() > 0.01  # ±1 orders propagate and carry power

    def test_lateral_shift_only_changes_phases(self):
        """Shifting the only patterned layer by δ multiplies order m by exp(∓2πi·m·δ)."""

        def run(offset):
            layer = rcwa.lamellar_layer(LN_TILTED, EYE, 0.4, 0.3, offset=offset)
            return rcwa.solve_grating([layer], 0.8, 1.0, 0.3, 1.0, GLASS, 6)

        base, shifted = run(0.0), run(0.13)
        np.testing.assert_allclose(shifted.R, base.R, atol=1e-13)
        np.testing.assert_allclose(shifted.T, base.T, atol=1e-13)
        phase = np.exp(-2j * np.pi * np.asarray(base.orders) * 0.13)[:, None, None]
        np.testing.assert_allclose(shifted.t, base.t * phase, atol=1e-12)

    def test_propagating_orders_only(self):
        """Orders evanescent in a medium carry no power there."""
        result = rcwa.solve_grating(
            [rcwa.lamellar_layer(SILICON, EYE, 0.5, 0.3)], 0.7, 1.0, 0.2, 1.0, GLASS, 6
        )
        kx = np.sin(0.2) + np.asarray(result.orders) * 1.0 / 0.7
        np.testing.assert_allclose(np.asarray(result.R)[np.abs(kx) > 1.0], 0.0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(result.T)[np.abs(kx) > 1.45], 0.0, atol=1e-15)

    def test_li_converges_faster_than_laurent_for_tm(self):
        layers = [rcwa.lamellar_layer(SILICON, EYE, 0.4, 0.3)]

        def r0(max_order, rule):
            res = rcwa.solve_grating(
                layers, 0.8, 1.0, 0.3, 1.0, GLASS, max_order, factorization=rule
            )
            return float(res.R[max_order, 0, 0])  # zeroth order, p → p (TM)

        reference = r0(60, "li")
        li_error = abs(r0(10, "li") - reference)
        laurent_error = abs(r0(10, "laurent") - reference)
        assert li_error < 0.2 * laurent_error, (li_error, laurent_error)
        assert abs(r0(60, "laurent") - reference) < laurent_error  # same limit, slowly

    def test_jit_and_vmap(self):
        def run(wavelength):
            layers = [rcwa.lamellar_layer(LN_TILTED, EYE, 0.4, 0.3)]
            return rcwa.solve_grating(layers, 0.8, wavelength, 0.3, 1.0, GLASS, 5).T

        wavelengths = jnp.array([0.9, 1.0, 1.1])
        eager = jnp.stack([run(w) for w in wavelengths])
        np.testing.assert_allclose(jax.jit(jax.vmap(run))(wavelengths), eager, atol=1e-13)


# ---------------------------------------------------------------------------
# Hyperbolic, non-symmetric and near-zero tensors
# ---------------------------------------------------------------------------


def _hbn_tilted():
    """hBN at 1450 cm⁻¹ (ε⊥ ≈ −9.7, ε∥ ≈ +2.75), axis tilted 40° and turned 25°."""
    from hyperbolic_optics.materials import create_material

    hbn = jnp.asarray(create_material("hBN").fetch_permittivity_tensor_for_freq(1450.0))
    return bcm.rotate_tensor(
        hbn.reshape(3, 3), bcm.rotation_matrix(0.0, jnp.radians(40.0), jnp.radians(25.0))
    )


HBN_WAVELENGTH = 1e4 / 1450.0


def _hbn_grating(eps, max_order, azimuth=0.4):
    layers = [rcwa.lamellar_layer(eps, EYE, 0.5, 0.5)]
    return rcwa.solve_grating(
        layers, 3.0, HBN_WAVELENGTH, 0.35, 1.0, 2.0, max_order, azimuth_rad=azimuth
    )


def _eigenmode_scattering(eps, max_order):
    """Independent layer solution: numpy eigenmodes, propagated as decaying phases."""
    layer = rcwa.lamellar_layer(eps, EYE, 0.5, 0.5)
    orders = np.arange(-max_order, max_order + 1)
    k_par, azimuth = np.sin(0.35), 0.4
    kx = jnp.asarray(k_par * np.cos(azimuth) + orders * HBN_WAVELENGTH / 3.0)
    ky = k_par * np.sin(azimuth)
    direction = (jnp.array([np.cos(azimuth)]), jnp.array([np.sin(azimuth)]))
    delta = np.asarray(
        _fourier_system_matrix(
            _factorize(layer.eps, layer.boundaries, max_order, "li"),
            _factorize(jnp.stack([EYE, EYE]), layer.boundaries, max_order, "li"),
            kx,
            ky,
        )
    )
    kz, modes = np.linalg.eig(delta)
    n = delta.shape[0] // 4
    flux = 0.5 * np.real(
        np.sum(
            modes[:n] * np.conj(modes[3 * n :]) - modes[n : 2 * n] * np.conj(modes[2 * n : 3 * n]),
            0,
        )
    )
    forward = np.where(np.abs(kz.imag) > 1e-9 * np.abs(kz), kz.imag > 0, flux > 0)
    order = np.concatenate([np.where(forward)[0], np.where(~forward)[0]])
    kz, modes = kz[order], jnp.asarray(modes[:, order])
    half, k0d = 2 * n, 2 * np.pi / HBN_WAVELENGTH * 0.5
    zero = jnp.zeros((half, half))
    propagation = (
        jnp.diag(jnp.exp(1j * kz[:half] * k0d)),
        zero,
        zero,
        jnp.diag(jnp.exp(-1j * kz[half:] * k0d)),
    )
    w_in, _ = _harmonic_modes(1.0, kx, ky, direction)
    w_out, _ = _harmonic_modes(2.0, kx, ky, direction)
    return _star(_star(_interface(w_in, modes), propagation), _interface(modes, w_out))


class TestEdgeSmoothing:
    def test_zero_width_is_the_hard_edge(self):
        hard = rcwa.lamellar_layer(LN_TILTED, EYE, 0.4, 0.3)
        smooth = rcwa.lamellar_layer(LN_TILTED, EYE, 0.4, 0.3, edge_width=0.0)
        a = rcwa.solve_grating([hard], 0.8, 1.0, 0.3, 1.0, GLASS, 8)
        b = rcwa.solve_grating([smooth], 0.8, 1.0, 0.3, 1.0, GLASS, 8)
        np.testing.assert_array_equal(np.asarray(a.R), np.asarray(b.R))

    def test_smoothing_is_a_box_convolution(self):
        """Width w turns each step into a linear ramp w wide, centred on the edge."""
        from hyperbolic_optics.rcwa import _toeplitz

        width, fill = 0.08, 0.4
        boundaries = jnp.array([0.0, fill, 1.0])
        values = jnp.array([3.0, 1.0])
        toeplitz = np.asarray(_toeplitz(values, boundaries, 6, width))
        coefficients = np.concatenate([toeplitz[6:, 6][::-1][:-1], toeplitz[6, 6:]])[::-1]

        x = (np.arange(200_000) + 0.5) / 200_000
        # distance to the nearer edge, signed positive inside the ridge
        inside = np.minimum(x, fill - x)
        inside = np.where(x > (1 + fill) / 2, x - 1, inside)
        ramp = np.clip(0.5 + inside / width, 0.0, 1.0)
        profile = 1.0 + 2.0 * ramp
        for index, m in enumerate(range(-6, 7)):
            expected = np.mean(profile * np.exp(-2j * np.pi * m * x))
            assert complex(coefficients[index]) == pytest.approx(expected, abs=2e-7)

    def test_smoothed_lossless_grating_conserves_energy(self):
        layer = rcwa.lamellar_layer(SILICON, EYE, 0.4, 0.35, edge_width=0.05)
        result = rcwa.solve_grating([layer], 0.9, 1.0, 0.3, 1.0, GLASS, 15)
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-11)


class TestHyperbolicAndNonSymmetric:
    def test_mirror_symmetries_fix_the_off_diagonal_signs(self):
        """x → −x flips ε_xy and ε_xz and reverses the orders; y → −y flips ε_xy, ε_yz."""
        eps = _hbn_tilted()
        base = _hbn_grating(eps, 15)
        mirror_x = jnp.diag(jnp.array([-1.0, 1.0, 1.0]))
        flipped = _hbn_grating(mirror_x @ eps @ mirror_x, 15, azimuth=np.pi - 0.4)
        np.testing.assert_allclose(flipped.R[::-1], base.R, atol=1e-9)
        np.testing.assert_allclose(flipped.T[::-1], base.T, atol=1e-9)
        mirror_y = jnp.diag(jnp.array([1.0, -1.0, 1.0]))
        flipped = _hbn_grating(mirror_y @ eps @ mirror_y, 15, azimuth=-0.4)
        np.testing.assert_allclose(flipped.R, base.R, atol=1e-12)
        np.testing.assert_allclose(flipped.T, base.T, atol=1e-12)

    @pytest.mark.parametrize("lossless", [False, True], ids=["lossy", "lossless"])
    def test_hyperbolic_layer_matches_eigenmode_solution(self, lossless):
        """41 orders of a full-tensor hyperbolic ridge: propagating high-k modes, kz ~ 170."""
        eps = _hbn_tilted()
        eps = jnp.real(eps).astype(complex) if lossless else eps
        result = _hbn_grating(eps, 20)
        reference = _eigenmode_scattering(eps, 20)
        size = 41
        incident = [20, size + 20]
        for got, block in ((result.r, reference[2]), (result.t, reference[0])):
            want = np.asarray(block)[:, incident]
            want = np.stack([want[:size], want[size:]], axis=-2)  # [order, out, in]
            np.testing.assert_allclose(np.asarray(got), want, atol=1e-8)

    def test_lossless_hyperbolic_grating_conserves_energy(self):
        eps = jnp.real(_hbn_tilted()).astype(complex)
        np.testing.assert_allclose(_total(_hbn_grating(eps, 20)), 1.0, atol=1e-9)
        # 81 orders: high-k modes with kz ≈ 360 set a ~1e-6 floor (see gratings.md)
        np.testing.assert_allclose(_total(_hbn_grating(eps, 40)), 1.0, atol=1e-5)

    def test_lossy_hyperbolic_grating_is_passive(self):
        result = _hbn_grating(_hbn_tilted(), 20)
        total = _total(result)
        assert (total < 1.0).all() and (total > 0.1).all()
        assert (np.asarray(result.R) >= -1e-12).all() and (np.asarray(result.T) >= -1e-12).all()

    def test_lossless_gyrotropic_grating_conserves_energy(self):
        """A Hermitian, non-symmetric tensor (ε_xy = −ε_yx, imaginary): lossless, non-reciprocal."""
        gyro = jnp.array([[4.0, 0.8j, 0.3], [-0.8j, 4.0, 0.2j], [0.3, -0.2j, 3.0]])
        layers = [rcwa.lamellar_layer(gyro, EYE, 0.45, 0.4)]
        result = rcwa.solve_grating(layers, 0.9, 1.0, 0.4, 1.0, GLASS, 20, azimuth_rad=0.6)
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-10)

    @pytest.mark.parametrize("eps_xx", [1e-2 + 1e-2j, 1e-4 + 1e-2j, 0.01j])
    def test_near_zero_eps_xx_stays_finite_and_passive(self, eps_xx):
        """Li's rule needs 1/ε_xx; with some loss it is safe right down to Re ε_xx = 0."""
        ridge = jnp.diag(jnp.array([eps_xx, 2.0, 2.0]))
        result = _hbn_grating(ridge, 20)
        total = _total(result)
        assert np.isfinite(np.asarray(result.T)).all()
        assert (total <= 1.0 + 1e-12).all() and (total > 0.9).all()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _grating_objective(p):
    ridge = bcm.rotate_tensor(
        jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex)),
        bcm.rotation_matrix(0.0, p["tilt"], 0.0),
    )
    # Two patterned layers, so their relative offset matters (a lone layer's
    # efficiencies are shift-invariant; see test_lateral_shift_only_changes_phases).
    layers = [
        rcwa.lamellar_layer(ridge, EYE, p["fill"], p["thickness"], offset=p["offset"]),
        rcwa.lamellar_layer(2.1 * EYE, EYE, 0.5, 0.1),
    ]
    result = rcwa.solve_grating(
        layers, p["period"], p["wavelength"], p["theta"], 1.0, GLASS, 8, azimuth_rad=p["azimuth"]
    )
    # first transmitted order p → s, zeroth reflected p → p, first transmitted s → s
    return result.T[9, 1, 0] + 0.5 * result.R[8, 0, 0] - 0.3 * result.T[9, 1, 1]


GRATING_PARAMS = {
    "fill": 0.42,
    "offset": 0.1,
    "thickness": 0.35,
    "period": 0.9,
    "wavelength": 1.0,
    "theta": 0.3,
    "azimuth": 0.2,
    "tilt": 0.6,
}


@pytest.mark.parametrize("name", sorted(GRATING_PARAMS))
def test_gradient_against_finite_differences(name):
    params = {k: jnp.asarray(v) for k, v in GRATING_PARAMS.items()}
    grad = float(jax.grad(lambda v: _grating_objective({**params, name: v}))(params[name]))

    def at(value):
        return float(_grating_objective({**params, name: jnp.asarray(value)}))

    base = float(params[name])
    errors = [
        abs((at(base + h) - at(base - h)) / (2 * h) - grad) / abs(grad) for h in (1e-4, 1e-5, 1e-6)
    ]
    assert abs(grad) > 1e-3, "pick a point where the derivative is appreciably nonzero"
    assert min(errors) < 1e-6, f"d/d{name}: relative errors {errors}"


def test_twisted_liquid_crystal_polarization_grating():
    """A tilted, twisted LC grating (full tensor, xz and yz terms): lossless, and its −1
    transmitted order switches off at λ = n_sub·Λ, where it turns evanescent."""
    n_o, n_e, n_sub, period, thickness, slices = 1.525, 1.775, 1.7, 0.4, 1.0, 8
    x = (jnp.arange(32) + 0.5) / 32
    tilt, twist = jnp.radians(25.0), jnp.radians(217.0)
    layers = []
    for k in range(slices):
        phi = jnp.pi * x + twist * (k + 0.5) * thickness / slices
        u = jnp.stack([jnp.cos(tilt) * jnp.cos(phi), jnp.cos(tilt) * jnp.sin(phi),
                       jnp.sin(tilt) * jnp.ones_like(phi)], axis=-1)  # fmt: skip
        eps = n_o**2 * jnp.eye(3) + (n_e**2 - n_o**2) * u[:, :, None] * u[:, None, :]
        layers.append(rcwa.sampled_layer(eps.astype(complex), thickness / slices))
    circular = jnp.array([1, 1j]) / np.sqrt(2)
    minus_one = []
    for wavelength in (0.66, 0.70):
        res = rcwa.solve_grating(layers, period, wavelength, 0.0, 1.0, n_sub**2, 6)
        orders = np.asarray(res.orders)
        total = np.sum(np.asarray(res.R) + np.asarray(res.T), axis=(0, 1))
        np.testing.assert_allclose(total, 1.0, atol=1e-9)
        power = np.asarray(res.T)[orders == -1][0] @ np.abs(circular) ** 2
        minus_one.append(float(power.sum()))
    assert minus_one[0] > 1e-3 and minus_one[1] < 1e-12
