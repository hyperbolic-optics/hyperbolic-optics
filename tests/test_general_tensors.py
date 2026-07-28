"""
Tests for fully general (non-symmetric) ε/μ tensors and the mode-direction sort.

The built-in material library is entirely lossy and reciprocal, so none of the
other test modules exercise lossless media, anisotropic permeability, or
non-symmetric tensors. Those three cases share a failure mode: a layer whose
four Berreman modes are not all of the same kind (some propagating, some
evanescent), which is what :meth:`Wave.wave_sorting` has to partition correctly.
"""

import numpy as np
import pytest

from hyperbolic_optics.materials import ArbitraryMaterial
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.waves import Wave


def build_payload(material, incident_angle=45.0, azimuthal_angle=20.0, rotation_y=30):
    """Semi-infinite arbitrary-material stack under a prism."""
    return {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": incident_angle,
            "azimuthal_angle": azimuthal_angle,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 11.56},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": material,
                "rotationX": 0,
                "rotationY": rotation_y,
                "rotationZ": 0,
            },
        ],
    }


def total_reflectance(material, **kwargs):
    """Return (R_p, R_s), each summing the co- and cross-polarized channels."""
    structure = Structure()
    structure.execute(build_payload(material, **kwargs))
    power = lambda r: float(np.abs(r).item() ** 2)  # noqa: E731
    return (
        power(structure.r_pp) + power(structure.r_ps),
        power(structure.r_ss) + power(structure.r_sp),
    )


def diagonal_wave(eps_diag, mu_diag, kx):
    """A Wave for a diagonal medium, built directly on the canonical layout."""
    eps = np.zeros((1, 1, 1, 1, 3, 3), dtype=complex)
    mu = np.zeros((1, 1, 1, 1, 3, 3), dtype=complex)
    for i in range(3):
        eps[..., i, i] = eps_diag[i]
        mu[..., i, i] = mu_diag[i]
    return Wave(np.full((1, 1, 1, 1), kx, dtype=complex), eps, mu, semi_infinite=True)


class TestGeneralTensorComponents:
    """All nine components of ε and μ are independently settable."""

    def test_upper_triangle_only_stays_symmetric(self):
        """Naming only the upper triangle keeps the historical symmetric meaning."""
        mat = ArbitraryMaterial(
            {"eps_xx": 2.0, "eps_yy": 3.0, "eps_zz": 4.0, "eps_xy": {"real": 0.5, "imag": 0.0}}
        )
        eps = mat.fetch_permittivity_tensor()

        assert np.allclose(eps, eps.T)
        assert eps[1, 0] == complex(0.5, 0.0)

    def test_lower_triangle_independent_when_given(self):
        """Every off-diagonal component survives verbatim into the tensor."""
        mat = ArbitraryMaterial(
            {
                "eps_xx": 2.0,
                "eps_yy": 3.0,
                "eps_zz": 4.0,
                "eps_xy": 0.5,
                "eps_yx": -0.5,
                "eps_xz": 1.0,
                "eps_zx": -1.0,
                "eps_yz": 0.2,
                "eps_zy": 0.7,
            }
        )
        eps = mat.fetch_permittivity_tensor()

        assert not np.allclose(eps, eps.T)
        expected = np.array(
            [[2.0, 0.5, 1.0], [-0.5, 3.0, 0.2], [-1.0, 0.7, 4.0]], dtype=np.complex128
        )
        assert np.allclose(eps, expected)

    def test_gyrotropic_permeability_is_hermitian(self):
        """A lossless gyrotropic μ (μ_xy = -μ_yx = iκ) is expressible and Hermitian."""
        kappa = 0.6
        mat = ArbitraryMaterial(
            {
                "mu_xx": 2.0,
                "mu_yy": 2.0,
                "mu_zz": 1.0,
                "mu_xy": {"real": 0.0, "imag": kappa},
                "mu_yx": {"real": 0.0, "imag": -kappa},
            }
        )
        mu = mat.fetch_magnetic_tensor()

        assert mu[0, 1] == complex(0.0, kappa)
        assert mu[1, 0] == complex(0.0, -kappa)
        assert np.allclose(mu, mu.conj().T)

    def test_mu_r_shorthand_still_works(self):
        """The mu_r scalar shorthand keeps setting the diagonal."""
        mat = ArbitraryMaterial({"eps_xx": 2.0, "mu_r": 3.0})
        mu = mat.fetch_magnetic_tensor()

        assert np.allclose(np.diag(mu), 3.0)
        assert np.allclose(mu - np.diag(np.diag(mu)), 0.0)


class TestAsymmetryReachesSolver:
    """The lower triangle must survive rotation and reach the reflection result."""

    def test_antisymmetry_survives_rotation(self):
        """R·A·Rᵀ preserves antisymmetry, so a gyrotropic tensor stays gyrotropic."""
        from hyperbolic_optics.layers import _euler_rotation_matrix

        rotation = _euler_rotation_matrix(np.float64(0.3), np.float64(0.7), np.float64(1.1))
        antisymmetric = np.array(
            [[0.0, 0.6j, 0.0], [-0.6j, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=complex
        )
        rotated = rotation @ antisymmetric @ np.swapaxes(rotation, -2, -1)

        assert np.allclose(rotated, -np.swapaxes(rotated, -2, -1))

    def test_lower_triangle_changes_the_result(self):
        """Guards against a regression that re-symmetrizes the tensor."""
        base = {"eps_xx": {"real": 5.0, "imag": 0.3}, "eps_yy": 3.0, "eps_zz": -2.0}
        symmetric = base | {"eps_xy": {"real": 0.0, "imag": 0.8}}
        gyroelectric = symmetric | {"eps_yx": {"real": 0.0, "imag": -0.8}}

        assert not np.allclose(total_reflectance(symmetric), total_reflectance(gyroelectric))


class TestModeDirectionSorting:
    """Forward/backward partitioning across the regimes the built-ins never reach."""

    def test_partition_is_a_permutation_with_mixed_modes(self):
        """A lossless uniaxial between its two light lines mixes mode kinds.

        ε_o = 5, ε_e = 2 puts the ordinary wave propagating and the
        extraordinary evanescent at kx = 1.8, the case a spliced sort ordering
        turns into a non-permutation.
        """
        wave = diagonal_wave([5.0, 5.0, 2.0], [1.0, 1.0, 1.0], kx=1.8)
        wave.delta_matrix_calc()
        eigenvalues = np.linalg.eigvals(wave.berreman_matrix)

        # Precondition: the regime really is mixed.
        evanescent = np.abs(np.imag(eigenvalues)) > Wave.DIRECTION_TOL
        assert evanescent.any() and not evanescent.all()

        transmitted_kz, reflected_kz, _, _ = wave.wave_sorting()
        selected = np.concatenate([transmitted_kz, reflected_kz], axis=-1).ravel()

        assert len(selected) == 4
        for value in eigenvalues.ravel():
            assert np.isclose(selected, value).any(), f"mode {value} dropped from the partition"

    def test_hyperbolic_forward_mode_follows_energy_not_phase(self):
        """In a type-II hyperbolic medium S_z and Re(kz) point opposite ways."""
        wave = diagonal_wave([-3.0, -3.0, 4.0], [1.0, 1.0, 1.0], kx=3.0)
        wave.delta_matrix_calc()
        transmitted_kz, _, transmitted_fields, _ = wave.wave_sorting()

        propagating = np.abs(np.imag(transmitted_kz)) <= Wave.DIRECTION_TOL
        assert propagating.any(), "expected a propagating mode in the transmitted set"

        Ex, Ey = transmitted_fields[..., 0, :], transmitted_fields[..., 1, :]
        Hx, Hy = transmitted_fields[..., 2, :], transmitted_fields[..., 3, :]
        Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

        # Energy flows into the medium ...
        assert (Sz[propagating] > 0).all()
        # ... while the phase front travels the other way.
        assert (np.real(transmitted_kz)[propagating] < 0).all()

    @pytest.mark.parametrize(
        "label,material",
        [
            ("lossless uniaxial", {"eps_xx": 5.0, "eps_yy": 5.0, "eps_zz": 2.0}),
            ("lossless type-I hyperbolic", {"eps_xx": 4.0, "eps_yy": 4.0, "eps_zz": -3.0}),
            ("lossless type-II hyperbolic", {"eps_xx": -3.0, "eps_yy": -3.0, "eps_zz": 4.0}),
            (
                "lossless anisotropic permeability",
                {
                    "eps_xx": 5.0,
                    "eps_yy": 5.0,
                    "eps_zz": 5.0,
                    "mu_xx": 2.0,
                    "mu_yy": 2.0,
                    "mu_zz": -1.0,
                },
            ),
            (
                "lossless gyrotropic permeability",
                {
                    "eps_xx": 4.0,
                    "eps_yy": 4.0,
                    "eps_zz": 4.0,
                    "mu_xx": 2.0,
                    "mu_yy": 2.0,
                    "mu_zz": 1.0,
                    "mu_xy": {"real": 0.0, "imag": 0.6},
                    "mu_yx": {"real": 0.0, "imag": -0.6},
                },
            ),
        ],
    )
    def test_passive_lossless_media_do_not_amplify(self, label, material):
        """A passive lossless half-space cannot reflect more than it receives.

        Swept over incident angle rather than probed at one geometry: which
        modes are propagating and which are evanescent depends on kx, so a
        single angle can sit entirely inside one regime and pass vacuously.
        """
        checked_mixed_regime = False

        for angle in np.linspace(1.0, 89.0, 45):
            structure = Structure()
            structure.execute(build_payload(material, incident_angle=float(angle)))

            power = lambda r: float(np.abs(r).item() ** 2)  # noqa: E731
            R_p = power(structure.r_pp) + power(structure.r_ps)
            R_s = power(structure.r_ss) + power(structure.r_sp)

            assert np.isfinite(R_p) and np.isfinite(R_s), f"{label}: NaN at {angle:.1f}°"
            assert R_p <= 1.0 + 1e-9, f"{label}: R_p = {R_p} at {angle:.1f}°"
            assert R_s <= 1.0 + 1e-9, f"{label}: R_s = {R_s} at {angle:.1f}°"

            _, kz = structure.layers[-1].profile.tangential_modes()
            evanescent = np.abs(np.imag(kz)) > Wave.DIRECTION_TOL
            checked_mixed_regime |= bool(evanescent.any() and not evanescent.all())

        assert checked_mixed_regime, (
            f"{label}: no swept angle produced both propagating and evanescent "
            "modes, so this case never exercised the partition"
        )

    @pytest.mark.parametrize(
        "eps_diag",
        [(5.0, 5.0, 2.0), (-3.0, -3.0, 4.0), (4.0, 4.0, -3.0)],
        ids=["uniaxial", "type-II hyperbolic", "type-I hyperbolic"],
    )
    def test_zero_damping_limit_is_continuous(self, eps_diag):
        """R(γ = 0) must equal lim γ→0 R(γ), not jump to a different branch."""

        def reflectance(damping):
            material = {
                key: {"real": value, "imag": damping}
                for key, value in zip(("eps_xx", "eps_yy", "eps_zz"), eps_diag, strict=True)
            }
            return total_reflectance(material)

        lossless = np.asarray(reflectance(0.0))
        near_lossless = np.asarray(reflectance(1e-9))

        assert np.allclose(lossless, near_lossless, atol=1e-6)


class TestModePartitionInvariants:
    """The specific checks the audit asked for, on the partition itself."""

    def test_forward_backward_split_is_two_and_two(self):
        """Four modes must resolve to two forward and two backward, everywhere.

        A 3/1 split would mean the direction test had failed on one mode, which
        the argsort-and-halve then hides by taking the first two regardless.
        """
        eps_options = [(5.0, 5.0, 2.0), (-3.0, -3.0, 4.0), (4.0, 4.0, -3.0), (5.0, 3.0, -2.0)]
        mu_options = [(1.0, 1.0, 1.0), (2.0, 2.0, -1.0)]
        checked = 0

        for eps_diag in eps_options:
            for mu_diag in mu_options:
                for kx in np.linspace(0.1, 6.0, 25):
                    wave = diagonal_wave(eps_diag, mu_diag, float(kx))
                    wave.delta_matrix_calc()
                    eigenvalues = np.linalg.eigvals(wave.berreman_matrix)

                    transmitted, reflected, _, _ = wave.wave_sorting()
                    assert transmitted.shape[-1] == 2
                    assert reflected.shape[-1] == 2

                    selected = np.concatenate([transmitted, reflected], axis=-1).ravel()
                    for value in eigenvalues.ravel():
                        assert np.isclose(selected, value).any(), (
                            f"eps={eps_diag} mu={mu_diag} kx={kx}: mode {value} lost"
                        )
                    checked += 1

        assert checked == len(eps_options) * len(mu_options) * 25

    def test_slot_zero_polarization_varies_continuously_with_rotation(self):
        """A small rotation must not make the slot-0 mode jump between p and s.

        The p/s ordering used to be chosen per point between two criteria sorted
        in opposite directions, so crossing the threshold flipped which mode sat
        in slot 0 -- discontinuously, for an arbitrarily small parameter change.
        """
        fractions = []
        for rotation_z in np.linspace(0.0, 1.0, 21):
            payload = {
                "ScenarioData": {
                    "type": "Simple",
                    "incidentAngle": 45.0,
                    "azimuthal_angle": 0.0,
                    "frequency": 1460.0,
                },
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": 11.56},
                    {
                        "type": "Semi Infinite Anisotropic Layer",
                        "material": {"eps_xx": 2.0, "eps_yy": 3.0, "eps_zz": 5.0},
                        "rotationX": 0,
                        "rotationY": 40,
                        "rotationZ": float(rotation_z),
                    },
                ],
            }
            structure = Structure()
            structure.execute(payload)
            profile = structure.layers[-1].profile
            Ex = np.abs(np.asarray(profile.transmitted_Ex).ravel()[0]) ** 2
            Ey = np.abs(np.asarray(profile.transmitted_Ey).ravel()[0]) ** 2
            fractions.append(Ey / (Ex + Ey))

        steps = np.abs(np.diff(fractions))
        assert steps.max() < 0.05, (
            f"slot-0 polarization character jumps by {steps.max():.3f} over a "
            "0.05 degree step; the ordering is flipping"
        )
