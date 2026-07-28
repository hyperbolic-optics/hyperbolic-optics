"""
Physics anchors that can actually fail.

The suite's headline conservation check (``FieldProfile.summary``'s
``conservation_residual``) is a telescoping identity -- the per-layer
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

ISOTROPIC_EXIT_EPS = 2.25
PRISM_EPS = 5.5


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
