"""
Physics anchors that can actually fail.

The suite's headline conservation check (``FieldProfile.summary``'s
``flux_bookkeeping_residual``) is a telescoping identity -- the per-layer
absorptances are defined as successive flux differences, so
``R + T + sum(A) - 1`` cancels algebraically and returns machine epsilon
whatever the fields did. The checks here are chosen so that a wrong result
makes them fail:

* passivity -- a passive stack cannot return more power than it receives, nor
  absorb a negative amount;
* cross-validation -- an isotropic medium routed through the anisotropic
  solver must reproduce the independent closed-form isotropic path;
* consistency -- quantities computed two different ways must agree.
"""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.waves import Wave

ISOTROPIC_EXIT_EPS = 2.25
PRISM_EPS = 5.5


class TestDegenerateSicExit:
    """SiC modal amplitudes depend on the degenerate basis; physical fields do not."""

    @pytest.mark.parametrize("backend", ["transfer", "scattering"])
    @pytest.mark.parametrize("mix_basis", [False, True])
    def test_aln_film_fields_match_analytic_solution(self, backend, mix_basis, monkeypatch):
        if mix_basis:
            original = Wave.wave_sorting
            angle = 0.37
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            def mixed(wave):
                forward, backward, down, up = original(wave)
                if wave.semi_infinite:
                    np.testing.assert_allclose(forward[..., 0], forward[..., 1], atol=1e-12)
                    down, up = down @ rotation, up @ rotation
                return forward, backward, down, up

            monkeypatch.setattr(Wave, "wave_sorting", mixed)

        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {
                    "type": "Incident",
                    "incidentAngle": [0.0, 10.0, 30.0, 55.0, 75.0, 85.0],
                    "frequency": [650.0, 750.0, 850.0, 950.0, 1000.0, 1040.0],
                },
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": 11.56},
                    {"type": "Crystal Layer", "material": "AlN", "thickness": 1.0, "rotationY": 0},
                    {"type": "Semi Infinite Anisotropic Layer", "material": "SiC", "rotationY": 0},
                ],
            },
            backend=backend,
        )
        profile = FieldProfile(structure)
        kx = structure.k_x
        q0 = np.sqrt(11.56 - kx**2 + 0j)
        optical_depth = structure.k_0 * structure.layers[1].thickness

        for polarization in ("s", "p"):
            admittance = [q0 if polarization == "s" else 11.56 / q0]
            propagation, shifts = [], []
            for layer in structure.layers[1:]:
                eps = layer.eps_tensor
                # The payload engine adds 1e-8 rad about y even at rotationY=0.
                # Include εxz in the analytic TM dispersion and phase exactly.
                if polarization == "s":
                    q = np.sqrt(eps[..., 1, 1] - kx**2 + 0j)
                    shift = 0.0
                    y = q
                else:
                    xx, zz, xz = eps[..., 0, 0], eps[..., 2, 2], eps[..., 0, 2]
                    q = np.sqrt((xx * zz - xz**2) * (zz - kx**2) / zz**2 + 0j)
                    q = np.where(q.imag < 0, -q, q)
                    shift = -kx * xz / zz
                    y = q * zz / (zz - kx**2)
                propagation.append(q)
                shifts.append(shift)
                admittance.append(y)
            y0, y1, y2 = admittance
            r01, r12 = (y0 - y1) / (y0 + y1), (y1 - y2) / (y1 + y2)
            phase = np.exp(1j * propagation[0] * optical_depth)
            transmitted = (
                (1 + r01)
                * (1 + r12)
                * phase
                * np.exp(1j * shifts[0] * optical_depth)
                / (1 + r01 * r12 * phase**2)
            )
            zero = np.zeros_like(transmitted)
            if polarization == "s":
                expected = np.stack([zero, transmitted, -y2 * transmitted, zero], axis=-1)
            else:
                transmitted = transmitted * q0 / np.sqrt(11.56)
                expected = np.stack([transmitted, zero, zero, y2 * transmitted], axis=-1)
            actual = profile._solve(polarization)[4][-1]
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

        if mix_basis:
            # Modal cross terms can be nonzero despite zero physical cross-polarization.
            assert np.max(abs(profile.transmission_coefficients()["t_ps"])) > 0.01


def stack(exit_layer, incident_angle=30.0, gap_thickness=0.5):
    return {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": incident_angle,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": PRISM_EPS},
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": gap_thickness,
                "permittivity": 1.0,
            },
            exit_layer,
        ],
    }


ISOTROPIC_EXIT = {
    "type": "Semi Infinite Isotropic Layer",
    "permittivity": ISOTROPIC_EXIT_EPS,
}

ANISOTROPIC_EXIT = {
    "type": "Semi Infinite Anisotropic Layer",
    "rotationX": 0,
    "rotationY": 0,
    "rotationZ": 0,
    "material": {
        "eps_xx": ISOTROPIC_EXIT_EPS,
        "eps_yy": ISOTROPIC_EXIT_EPS,
        "eps_zz": ISOTROPIC_EXIT_EPS,
    },
}


def solve(exit_layer, incident_angle=30.0):
    structure = Structure()
    structure.execute(stack(exit_layer, incident_angle=incident_angle))
    return structure


class TestAnisotropicSolverMatchesIsotropicClosedForm:
    """An isotropic tensor through the 4x4 eigen-solver must match the analytic path.

    ``IsotropicSemiInfiniteLayer`` builds its exit matrix in closed form
    (:class:`~hyperbolic_optics.layers.AmbientExitMedium`), independently of the
    Berreman eigen-decomposition. Feeding the same medium in as a diagonal
    permittivity tensor therefore cross-validates the whole eigen path against
    a reference the eigen path had no part in producing.
    """

    @pytest.mark.parametrize("angle", [10.0, 30.0, 55.0, 75.0])
    def test_reflection_coefficients_match(self, angle):
        iso = solve(ISOTROPIC_EXIT, angle)
        aniso = solve(ANISOTROPIC_EXIT, angle)

        for name in ("r_pp", "r_ss", "r_ps", "r_sp"):
            np.testing.assert_allclose(
                np.asarray(getattr(aniso, name)),
                np.asarray(getattr(iso, name)),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{name} disagrees between the isotropic and anisotropic paths",
            )

    @pytest.mark.parametrize("angle", [10.0, 30.0, 55.0, 75.0])
    def test_transmission_amplitudes_match(self, angle):
        """Guards the exit-mode normalization.

        The eigensolver normalizes over the tangential 4-vector, the closed-form
        path over E. Without rescaling, these disagree by an angle-dependent
        factor that silently propagates into every reported ``t``.
        """
        iso = FieldProfile(solve(ISOTROPIC_EXIT, angle)).transmission_coefficients()
        aniso = FieldProfile(solve(ANISOTROPIC_EXIT, angle)).transmission_coefficients()

        for name in ("t_pp", "t_ss", "t_ps", "t_sp"):
            np.testing.assert_allclose(
                np.asarray(aniso[name]),
                np.asarray(iso[name]),
                rtol=1e-9,
                # The cross terms are identically zero here, so the comparison
                # is against the solver's noise floor, not against a magnitude.
                atol=1e-9,
                err_msg=f"{name} disagrees between the isotropic and anisotropic paths",
            )

    def test_power_quantities_match(self):
        iso = FieldProfile(solve(ISOTROPIC_EXIT))
        aniso = FieldProfile(solve(ANISOTROPIC_EXIT))

        for polarization in ("p", "s"):
            np.testing.assert_allclose(
                np.asarray(aniso.transmittance(polarization)),
                np.asarray(iso.transmittance(polarization)),
                rtol=1e-9,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.asarray(aniso.reflectance(polarization)),
                np.asarray(iso.reflectance(polarization)),
                rtol=1e-9,
                atol=1e-12,
            )


class TestPassivity:
    """A passive stack absorbs a non-negative amount in every layer."""

    @staticmethod
    def _quartz_stack(frequency):
        return {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 40.0,
                "azimuthal_angle": 0.0,
                "frequency": frequency,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {
                    "type": "Crystal Layer",
                    "material": "Quartz",
                    "thickness": 1.0,
                    "rotationY": 70,
                },
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Quartz",
                    "rotationY": 90,
                },
            ],
        }

    @pytest.mark.parametrize("frequency", [450.0, 500.0, 550.0])
    def test_layer_absorptance_is_non_negative(self, frequency):
        """Inside every material's fitted band, no layer may supply power."""
        structure = Structure()
        structure.execute(self._quartz_stack(frequency))
        profile = FieldProfile(structure)

        for polarization in ("p", "s"):
            for entry in profile.layer_absorption(polarization):
                absorptance = np.asarray(entry["absorptance"])
                assert np.all(absorptance >= -1e-9), (
                    f"layer {entry['index']} has negative absorptance at "
                    f"{frequency} cm^-1 for {polarization}: {absorptance.min()}"
                )
            assert np.all(np.asarray(profile.transmittance(polarization)) >= -1e-9)
            assert np.all(np.asarray(profile.reflectance(polarization)) <= 1.0 + 1e-9)

    def test_extrapolated_material_warns(self):
        """Stacking materials with disjoint fitted ranges must not be silent.

        Quartz is fitted over 410-600 cm^-1; pairing it with Sapphire resolves
        the grid to Sapphire's range and evaluates Quartz far outside its own,
        where the factorized form returns Im(eps) < 0 -- gain from a passive
        crystal, which surfaces as negative layer absorptance.
        """
        payload = {
            "ScenarioData": {"type": "Incident"},
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {
                    "type": "Crystal Layer",
                    "material": "Quartz",
                    "thickness": 1.0,
                    "rotationY": 70,
                },
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Sapphire",
                    "rotationY": 90,
                },
            ],
        }
        with pytest.warns(UserWarning, match="outside the"):
            Structure().execute(payload)


class TestReflectanceConsistency:
    """``reflectance`` must equal the co- plus cross-polarized power it is built from."""

    @pytest.mark.parametrize("material", ["Calcite", "Quartz", "Sapphire"])
    def test_reflectance_matches_coefficient_sum(self, material):
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 30.0,
                "frequency": 1460.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 12.5},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": material,
                    "rotationX": 0,
                    "rotationY": 45,
                    "rotationZ": 0,
                },
            ],
        }
        structure = Structure()
        structure.execute(payload)
        profile = FieldProfile(structure)

        for polarization, co, cross in (("p", "r_pp", "r_ps"), ("s", "r_ss", "r_sp")):
            expected = (
                np.abs(np.asarray(getattr(structure, co))) ** 2
                + np.abs(np.asarray(getattr(structure, cross))) ** 2
            )
            np.testing.assert_allclose(
                np.asarray(profile.reflectance(polarization)),
                expected,
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{material}: reflectance('{polarization}') != |{co}|^2 + |{cross}|^2",
            )
