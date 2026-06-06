"""Tests for the Jones (amplitude-level) polarization calculus.

Anchors: the Jones matrix must be consistent with the existing Mueller matrix
(via the shared Jones->Mueller bridge), reproduce the reflectance |r|^2 the field
machinery already validates, and behave correctly for textbook optical elements
(crossed polarizers extinguish; a QWP turns 45 deg linear into circular).
"""

import numpy as np
import pytest

from hyperbolic_optics.jones import Jones
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure

_CALCITE = {"type": "Semi Infinite Anisotropic Layer", "material": "Calcite", "rotationY": 90}


def _simple(extra, eps_prism=25.0, angle=30.0):
    structure = Structure()
    structure.execute(
        {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": angle,
                "azimuthal_angle": 0.0,
                "frequency": 1460.0,
            },
            "Layers": [{"type": "Ambient Incident Layer", "permittivity": eps_prism}, *extra],
        }
    )
    return structure


def _converting():
    # azimuthally rotated calcite film -> genuine p<->s conversion
    return _simple(
        [
            {
                "type": "Crystal Layer",
                "material": "Calcite",
                "thickness": 1.0,
                "rotationY": 90,
                "rotationZ": 30,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
        ]
    )


class TestJonesMatrix:
    def test_matrix_shape_and_entries(self):
        structure = _converting()
        j = Jones(structure).calculate_jones_matrix()
        assert j.shape == (2, 2)
        # rows = output (p, s), cols = input (p, s)
        assert j[0, 0] == complex(structure.r_pp)
        assert j[0, 1] == complex(structure.r_ps)
        assert j[1, 0] == complex(structure.r_sp)
        assert j[1, 1] == complex(structure.r_ss)

    def test_incident_p_intensity_is_reflectance(self):
        # |J @ [1,0]|^2 = |r_pp|^2 + |r_sp|^2 = R_p (the quantity FieldProfile checks)
        structure = _converting()
        jones = Jones(structure)
        jones.set_incident_polarization("linear", angle=0)
        jones.add_optical_component("sample")
        expected = abs(complex(structure.r_pp)) ** 2 + abs(complex(structure.r_sp)) ** 2
        assert float(jones.get_intensity()) == pytest.approx(expected, abs=1e-12)


class TestJonesMuellerBridge:
    def test_bridge_matches_mueller(self):
        structure = _converting()
        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()
        bridged = Jones(structure).to_mueller()
        assert np.allclose(bridged, mueller.mueller_matrix, atol=1e-12)

    def test_bridge_matches_across_incident_scenario(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Incident"},
                "Layers": [{"type": "Ambient Incident Layer", "permittivity": 12.5}, _CALCITE],
            }
        )
        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()
        bridged = Jones(structure).to_mueller()
        assert bridged.shape == mueller.mueller_matrix.shape
        assert np.allclose(bridged, mueller.mueller_matrix, atol=1e-10)


class TestOpticalComponents:
    def test_crossed_polarizers_extinguish(self):
        # polarizer(0) then polarizer(90) -> zero transmission
        jones = Jones(_simple([_CALCITE]))
        jones.set_incident_polarization("linear", angle=0)
        jones.add_optical_component("linear_polarizer", 0)
        jones.add_optical_component("linear_polarizer", 90)
        assert float(jones.get_intensity()) == pytest.approx(0.0, abs=1e-12)

    def test_qwp_turns_linear45_into_circular(self):
        # 45 deg linear through a QWP at 0 deg -> circular (|S3| = S0, S1=S2=0)
        jones = Jones(_simple([_CALCITE]))
        jones.set_incident_polarization("linear", angle=45)
        jones.add_optical_component("quarter_wave_plate", 0)
        stokes = jones.get_stokes_parameters()
        assert abs(stokes["S3"]) == pytest.approx(stokes["S0"], abs=1e-9)
        assert stokes["S1"] == pytest.approx(0.0, abs=1e-9)
        assert stokes["S2"] == pytest.approx(0.0, abs=1e-9)

    def test_unknown_component_raises(self):
        with pytest.raises(ValueError):
            Jones(_simple([_CALCITE])).add_optical_component("bogus")

    def test_unknown_polarization_raises(self):
        with pytest.raises(ValueError):
            Jones(_simple([_CALCITE])).set_incident_polarization("diagonal")


class TestIncidentStates:
    def test_circular_handedness(self):
        jones = Jones(_simple([_CALCITE]))
        jones.set_incident_polarization("circular", handedness="right")
        # right-circular -> S3 = +S0 with this convention (matches Mueller)
        e_p, e_s = jones.incident_jones
        s3 = -2.0 * np.imag(e_p * np.conj(e_s))
        assert s3 == pytest.approx(1.0, abs=1e-12)


class TestEigenpolarizations:
    """Eigenpolarizations = states with no conversion; EP diagnostics."""

    def test_eigenpolarizations_undergo_no_conversion(self):
        structure = _converting()
        jones = Jones(structure)
        result = jones.eigenpolarizations()
        j = jones.calculate_jones_matrix()
        eigenvalues = result["eigenvalues"]
        vectors = result["eigenpolarizations"]
        for i in range(2):
            v = vectors[:, i]
            # J @ v == lambda * v  (output state == input state, no conversion)
            assert np.allclose(j @ v, eigenvalues[i] * v, atol=1e-10)

    def test_discriminant_matches_eigenvalue_split(self):
        # lambda+/- = tr/2 +/- sqrt(D)  ->  (lambda+ - lambda-)^2 / 4 == D
        result = Jones(_converting()).eigenpolarizations()
        lam0, lam1 = result["eigenvalues"]
        assert (lam0 - lam1) ** 2 / 4.0 == pytest.approx(complex(result["discriminant"]), abs=1e-10)

    def test_overlap_in_unit_interval(self):
        # non-orthogonality of the two eigenpolarizations is in [0, 1]
        overlap = float(Jones(_converting()).eigenpolarizations()["eigenvector_overlap"])
        assert 0.0 - 1e-12 <= overlap <= 1.0 + 1e-12

    def test_batched_shapes(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Incident"},
                "Layers": [{"type": "Ambient Incident Layer", "permittivity": 12.5}, _CALCITE],
            }
        )
        result = Jones(structure).eigenpolarizations()
        assert result["eigenvalues"].shape == (410, 360, 2)
        assert result["eigenpolarizations"].shape == (410, 360, 2, 2)
        assert result["discriminant"].shape == (410, 360)


class TestBroadcastingCompose:
    """A kx-free ideal component composes with a swept sample by broadcasting."""

    def test_ideal_component_broadcasts_over_azimuthal_sweep(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Azimuthal", "incidentAngle": 40},
                "Layers": [{"type": "Ambient Incident Layer", "permittivity": 12.5}, _CALCITE],
            }
        )
        jones = Jones(structure)
        jones.set_incident_polarization("linear", angle=0)
        jones.add_optical_component("sample")  # [F, B, 2, 2] batched
        jones.add_optical_component("linear_polarizer", 90)  # [2, 2] broadcasts
        intensity = jones.get_intensity()
        assert intensity.shape == structure.r_pp.shape  # (410, 360)
        assert np.all(np.isfinite(intensity))
