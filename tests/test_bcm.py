"""Tests for the differentiable JAX solver (``hyperbolic_optics.bcm``).

Forward results are checked against closed forms (Fresnel, Airy) and, complex
coefficient for complex coefficient, against the NumPy engine's scattering
backend, fed the very tensors it used. Gradients are checked against central
finite differences at several step sizes. Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)  # reference accuracy; bcm never sets this itself

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm  # noqa: E402
from hyperbolic_optics.layers import _euler_rotation_matrix  # noqa: E402
from hyperbolic_optics.scattering import scattering_coefficients  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402
from hyperbolic_optics.waves import Wave  # noqa: E402

NO_LAYERS = (jnp.zeros((0, 3, 3)), jnp.zeros((0,)))


def _fresnel(n1, n2, theta):
    """Single-interface amplitudes (r_p, r_s, t_p, t_s) on the package's convention."""
    c1 = np.cos(theta)
    c2 = np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2 + 0j)
    r_p = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    r_s = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    t_p = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    t_s = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    return r_p, r_s, t_p, t_s


def _uniaxial(eps_o, eps_e, tilt, azimuth):
    """Uniaxial tensor with its axis at ``tilt`` out of the interface, ``azimuth`` in it."""
    axis = jnp.array(
        [jnp.cos(tilt) * jnp.cos(azimuth), jnp.cos(tilt) * jnp.sin(azimuth), jnp.sin(tilt)]
    )
    return eps_o * jnp.eye(3) + (eps_e - eps_o) * jnp.outer(axis, axis)


# A lossless anisotropic multilayer (arXiv:2608.06185 Fig. 3, LiNbO3 axis tilted
# out of plane so every off-diagonal component is present).
LN = _uniaxial(5.46, 5.06, np.radians(45.0), np.radians(30.0))
# Complex so a lossy tensor can be written into it (.at[].set would drop Im).
MULTILAYER_EPS = jnp.stack(
    [1.38**2 * jnp.eye(3), 2.40**2 * jnp.eye(3), LN, 1.46**2 * jnp.eye(3)]
).astype(complex)
MULTILAYER_D = jnp.array([0.070, 0.190, 2.8, 0.080])


# ---------------------------------------------------------------------------
# Building blocks against the NumPy engine
# ---------------------------------------------------------------------------


class TestBuildingBlocks:
    def test_rotation_matches_payload_convention(self):
        angles = (0.3, -1.1, 2.4)
        np.testing.assert_allclose(
            np.asarray(bcm.rotation_matrix(*angles)),
            np.real(_euler_rotation_matrix(*angles)),
            atol=1e-15,
        )

    def test_system_matrix_matches_berreman(self):
        """Same Δ as Wave.delta_matrix_calc for general complex ε and μ."""
        rng = np.random.default_rng(0)
        eps = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) + 4 * np.eye(3)
        mu = 0.2 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) + np.eye(3)
        kx = 0.7
        canonical = (None, None, None, None)  # [A, B, F, T]
        wave = Wave(np.full((1, 1, 1, 1), kx), eps[canonical], mu[canonical], semi_infinite=True)
        wave.delta_matrix_calc()
        np.testing.assert_allclose(
            np.asarray(bcm.system_matrix(jnp.asarray(eps), jnp.asarray(mu), kx)),
            wave.berreman_matrix[0, 0, 0, 0],
            rtol=1e-14,
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


class TestClosedForms:
    @pytest.mark.parametrize("eps_in,eps_out", [(2.25, 1.0), (1.0, 2.25), (5.5, 1.0)])
    def test_single_interface(self, eps_in, eps_out):
        """No layers: exactly Fresnel, including past the critical angle."""
        theta = np.radians(np.linspace(0.0, 85.0, 18))
        result = bcm.solve_stack(*NO_LAYERS, 1.0, theta, eps_in, eps_out)
        r_p, r_s, t_p, t_s = _fresnel(np.sqrt(eps_in), np.sqrt(eps_out), theta)
        for got, want in (
            (result.r[:, 0, 0], r_p),
            (result.r[:, 1, 1], r_s),
            (result.t[:, 0, 0], t_p),
            (result.t[:, 1, 1], t_s),
        ):
            np.testing.assert_allclose(np.asarray(got), want, atol=1e-13)
        np.testing.assert_allclose(np.asarray(result.r[:, 0, 1]), 0.0, atol=1e-15)

    def test_film_through_quarter_wave_thicknesses(self):
        """The E-at-one-face / H-at-the-other relation is singular here; this is not.

        Normal incidence on an n = 2 film with thicknesses stepping exactly onto
        every odd multiple of λ/4n up to 8 quarter-waves.
        """
        n, n_sub, wavelength = 2.0, 1.5, 1.0
        thickness = jnp.linspace(0.0, 1.0, 401)
        eps = jnp.broadcast_to(n**2 * jnp.eye(3), (401, 1, 3, 3))
        result = bcm.solve_stack(eps, thickness[:, None], wavelength, 0.0, 1.0, n_sub**2)

        r01, r12 = (1 - n) / (1 + n), (n - n_sub) / (n + n_sub)
        phase = np.exp(4j * np.pi * n * np.asarray(thickness) / wavelength)
        airy = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
        np.testing.assert_allclose(np.asarray(result.r[:, 1, 1]), airy, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.r[:, 0, 0]), -airy, atol=1e-12)


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_lossless_multilayer_conserves_energy(self):
        wavelength = jnp.linspace(0.45, 1.0, 111)
        result = bcm.solve_stack(
            MULTILAYER_EPS, MULTILAYER_D, wavelength, np.radians(30.0), 1.0, 1.52**2
        )
        total = np.asarray(result.R.sum(axis=-2) + result.T.sum(axis=-2))  # per input
        np.testing.assert_allclose(total, 1.0, atol=1e-12)
        assert np.asarray(result.R[:, 1, 0]).max() > 1e-3, "the tilted axis should convert p to s"

    def test_absorbing_hyperbolic_film_is_passive(self):
        eps = _uniaxial(-4.84 + 0.755j, 2.27 + 0.001j, np.radians(30.0), np.radians(20.0))
        theta = np.radians(np.linspace(0.0, 80.0, 41))
        result = bcm.solve_stack(eps[None], jnp.array([1.0]), 10.0, theta, 1.0, 2.4**2)
        total = np.asarray(result.R.sum(axis=-2) + result.T.sum(axis=-2))
        assert (total <= 1.0 + 1e-12).all() and (total > 0.0).all()
        assert (np.asarray(result.T) >= -1e-15).all()

    def test_converged_in_doublings(self):
        """The default count is converged: four more doublings change nothing."""
        wavelength = jnp.linspace(0.45, 1.0, 23)
        args = (MULTILAYER_EPS, MULTILAYER_D, wavelength, np.radians(30.0), 1.0, 1.52**2)
        base = bcm.solve_stack(*args)
        finer = bcm.solve_stack(*args, doublings=bcm.DEFAULT_DOUBLINGS + 4)
        np.testing.assert_allclose(np.asarray(base.r), np.asarray(finer.r), atol=1e-12)
        np.testing.assert_allclose(np.asarray(base.t), np.asarray(finer.t), atol=1e-12)

    def test_jit_and_vmap_agree_with_eager(self):
        wavelength = jnp.linspace(0.45, 1.0, 7)
        args = (MULTILAYER_EPS, MULTILAYER_D, wavelength, np.radians(30.0), 1.0, 1.52**2)
        eager = bcm.solve_stack(*args)
        jitted = jax.jit(bcm.solve_stack, static_argnames="doublings")(*args)
        mapped = jax.vmap(lambda w: bcm.solve_stack(*args[:2], w, *args[3:]).r)(wavelength)
        np.testing.assert_allclose(np.asarray(jitted.r), np.asarray(eager.r), atol=1e-14)
        np.testing.assert_allclose(np.asarray(mapped), np.asarray(eager.r), atol=1e-14)


# ---------------------------------------------------------------------------
# Against the NumPy engine, coefficient for coefficient
# ---------------------------------------------------------------------------


def _inputs_from_structure(structure):
    """The executed structure's own tensors, thicknesses and axes, as bcm inputs."""
    layers = structure.layers[1:-1]
    batch = np.broadcast_shapes(*(layer.eps_tensor.shape[:-2] for layer in layers))
    eps = np.stack([np.broadcast_to(ly.eps_tensor, batch + (3, 3)) for ly in layers], axis=-3)
    mu = np.stack([np.broadcast_to(ly.mu_tensor, batch + (3, 3)) for ly in layers], axis=-3)
    thickness_um = np.array([layer.thickness * 1e4 for layer in layers])  # cm -> µm
    eps_prism = float(structure.eps_prism)
    incidence = np.arcsin(np.real(structure.k_x) / np.sqrt(eps_prism))  # [A, 1, 1, 1]
    wavelength_um = 2 * np.pi / structure.k_0 * 1e4  # [1, 1, F, 1]
    return {
        "eps": eps,
        "mu": mu,
        "thickness_um": thickness_um,
        "wavelength_um": wavelength_um,
        "incidence_rad": incidence,
        "eps_incident": eps_prism,
        "eps_exit": float(structure.layers[-1].eps_exit),
    }


def _assert_matches_engine(payload, atol):
    structure = Structure()
    structure.execute(payload, backend="scattering")
    reference = scattering_coefficients(structure.layers, structure.k_0)
    result = bcm.solve_stack(**_inputs_from_structure(structure))
    # Package names put the incident polarization first: r_ps is p in, s out.
    names = {"pp": (0, 0), "ps": (1, 0), "sp": (0, 1), "ss": (1, 1)}
    for kind, got in (("r", result.r), ("t", result.t)):
        for pair, (out, inc) in names.items():
            want = np.asarray(reference[f"{kind}_{pair}"])  # canonical [A, B, F, T]
            np.testing.assert_allclose(
                np.broadcast_to(np.asarray(got[..., out, inc]), want.shape),
                want,
                atol=atol,
                err_msg=f"{kind}_{pair}",
            )


def _payload(scenario, layers, eps_prism, eps_exit):
    return {
        "ScenarioData": scenario,
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": eps_prism},
            *layers,
            {"type": "Semi Infinite Isotropic Layer", "permittivity": eps_exit},
        ],
    }


class TestAgainstNumpyEngine:
    def test_dispersive_hyperbolic_calcite(self):
        """Calcite through its hyperbolic bands, rotated in and out of plane."""
        payload = _payload(
            {"type": "Incident", "incidentAngle": [5.0, 25.0, 45.0, 65.0]},
            [
                {
                    "type": "Crystal Layer",
                    "material": "Calcite",
                    "thickness": 0.6,
                    "rotationY": 70,
                    "rotationZ": 40,
                }
            ],
            eps_prism=5.5,
            eps_exit=1.0,
        )
        _assert_matches_engine(payload, atol=1e-11)

    def test_otto_gap_over_a_biaxial_crystal(self):
        """Evanescent gap above α-MoO3, swept in azimuth (tensors carry the B axis)."""
        payload = _payload(
            {
                "type": "Azimuthal",
                "incidentAngle": 50.0,
                "frequency": 950.0,
                "azimuthal_angle": [0.0, 30.0, 60.0, 90.0],
            },
            [
                {"type": "Isotropic Middle-Stack Layer", "thickness": 1.5, "permittivity": 1.0},
                {"type": "Crystal Layer", "material": "MoO3", "thickness": 0.4},
            ],
            eps_prism=12.5,
            eps_exit=2.0,
        )
        _assert_matches_engine(payload, atol=1e-11)

    def test_magnetic_layer(self):
        """Gyrotropic permeability: μ enters Δ as fully as ε does."""
        material = {
            "eps_xx": 3.0,
            "eps_yy": 3.5,
            "eps_zz": 2.5,
            "eps_xz": 0.2,
            "mu_xx": 1.2,
            "mu_yy": 1.2,
            "mu_zz": 1.0,
            "mu_xy": {"real": 0.0, "imag": 0.3},
            "mu_yx": {"real": 0.0, "imag": -0.3},
        }
        payload = _payload(
            {"type": "Incident", "incidentAngle": [10.0, 40.0], "frequency": [900.0, 1100.0]},
            [{"type": "Crystal Layer", "material": material, "thickness": 2.0, "rotationZ": 25}],
            eps_prism=2.0,
            eps_exit=1.5,
        )
        _assert_matches_engine(payload, atol=1e-11)

    def test_thick_evanescent_gap(self):
        """150 µm of evanescent gap takes the transfer product to NaN; not this."""
        payload = _payload(
            {"type": "Simple", "incidentAngle": 45.0, "azimuthal_angle": 0.0, "frequency": 1460.0},
            [
                {"type": "Isotropic Middle-Stack Layer", "thickness": 150.0, "permittivity": 1.0},
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.0},
            ],
            eps_prism=50.0,
            eps_exit=1.0,
        )
        _assert_matches_engine(payload, atol=1e-11)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _objective(params):
    """A scalar mixing co- and cross-polarized channels of the tilted multilayer."""
    ln = bcm.rotate_tensor(
        jnp.diag(jnp.array([5.46, 5.46, params["eps_e"]])),
        bcm.rotation_matrix(0.0, params["tilt"], params["azimuth"]),
    )
    eps = MULTILAYER_EPS.at[2].set(ln)
    thickness = MULTILAYER_D.at[2].set(params["thickness"])
    result = bcm.solve_stack(eps, thickness, params["wavelength"], params["theta"], 1.0, 1.52**2)
    return result.R[1, 0] + 0.5 * result.T[0, 0] - 0.3 * result.T[1, 0]


PARAMS = {
    "eps_e": 5.06 + 0.02j,
    "tilt": 0.7,
    "azimuth": 0.4,
    "thickness": 2.8,
    "wavelength": 0.633,
    "theta": 0.5,
}


class TestGradients:
    @pytest.mark.parametrize("name", ["tilt", "azimuth", "thickness", "wavelength", "theta"])
    def test_real_parameters(self, name):
        params = {k: jnp.asarray(v) for k, v in PARAMS.items()}
        grad = float(jax.grad(lambda value: _objective({**params, name: value}))(params[name]))

        def at(value):
            return float(_objective({**params, name: jnp.asarray(value)}))

        base = float(params[name])
        errors = [
            abs((at(base + h) - at(base - h)) / (2 * h) - grad) / abs(grad)
            for h in (1e-4, 1e-5, 1e-6)
        ]
        assert abs(grad) > 1e-3, "pick a point where the derivative is appreciably nonzero"
        # Central differences converge as h² until rounding takes over, so the
        # best of the three steps is the honest comparison.
        assert min(errors) < 1e-7, f"d/d{name}: relative errors {errors} at steps 1e-4..1e-6"

    def test_complex_tensor_component(self):
        """Derivative in the real and imaginary parts of ε_e (loss is differentiable)."""
        params = {k: jnp.asarray(v) for k, v in PARAMS.items()}

        def of(real, imag):
            return _objective({**params, "eps_e": real + 1j * imag})

        re, im = jnp.real(params["eps_e"]), jnp.imag(params["eps_e"])
        grad = jax.grad(of, argnums=(0, 1))(re, im)
        assert abs(float(grad[1])) > 1e-3, "the loss derivative must not be trivially zero"
        h = 1e-5
        fd = (
            (float(of(re + h, im)) - float(of(re - h, im))) / (2 * h),
            (float(of(re, im + h)) - float(of(re, im - h))) / (2 * h),
        )
        np.testing.assert_allclose(np.array(grad, dtype=float), fd, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64")
def test_warns_without_x64():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.warns(UserWarning, match="complex64"):
            bcm.solve_stack(*NO_LAYERS, 1.0, 0.3, 2.25, 1.0)
    finally:
        jax.config.update("jax_enable_x64", True)
