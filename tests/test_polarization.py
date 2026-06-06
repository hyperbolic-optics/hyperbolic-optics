"""Tests for the cross-polarization tools.

Covers the three additions:

- ``FieldProfile.polarization_resolved`` (experimental co/cross-pol R/T),
- ``FieldProfile.stokes_from_field_profile`` (polarization state vs depth),
- ``Mueller.calculate_transmission_mueller_matrix`` (transmission Mueller, valid
  as a power description only for a symmetric prism == substrate system).
"""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure

_SIMPLE = {"type": "Simple", "incidentAngle": 20.0, "azimuthal_angle": 0.0, "frequency": 1460.0}


def _execute(layers, scenario=None):
    structure = Structure()
    structure.execute({"ScenarioData": scenario or _SIMPLE, "Layers": layers})
    return structure


# A finite calcite film rotated azimuthally -> genuine s<->p conversion.
def _converting_stack(eps_prism=25.0, eps_exit=1.0):
    return [
        {"type": "Ambient Incident Layer", "permittivity": eps_prism},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
        {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 1.0,
            "rotationY": 90,
            "rotationZ": 30,
        },
        {"type": "Semi Infinite Isotropic Layer", "permittivity": eps_exit},
    ]


_ISO_STACK = [
    {"type": "Ambient Incident Layer", "permittivity": 2.0},
    {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
    {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
]


class TestPolarizationResolved:
    """Experimental co/cross-pol reflectance and transmittance."""

    def test_reflection_split_matches_coefficients(self):
        structure = _execute(_converting_stack())
        fp = FieldProfile(structure)
        res = fp.polarization_resolved("p")
        # co-pol reflectance == |r_pp|^2, cross-pol == |r_ps|^2 (p incidence)
        assert res["R_co"] == pytest.approx(abs(complex(structure.r_pp)) ** 2, abs=1e-9)
        assert res["R_cross"] == pytest.approx(abs(complex(structure.r_ps)) ** 2, abs=1e-9)

    def test_cross_polarization_present(self):
        # an azimuthally rotated anisotropic film must convert some p into s
        fp = FieldProfile(_execute(_converting_stack()))
        res = fp.polarization_resolved("p")
        assert float(res["R_cross"]) > 1e-6
        assert 0.0 <= float(res["conversion_reflection"]) <= 1.0

    def test_energy_split_conserves_isotropic_exit(self):
        # with an isotropic exit the s/p split is exact, so co+cross of R and T
        # plus interior absorption must sum to 1.
        structure = _execute(_converting_stack(eps_exit=1.0))
        fp = FieldProfile(structure)
        res = fp.polarization_resolved("p")
        absorbed = float(fp.summary("p")["total_absorption"])
        total = (
            float(res["R_co"])
            + float(res["R_cross"])
            + float(res["T_co"])
            + float(res["T_cross"])
            + absorbed
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_co_cross_sum_to_totals(self):
        structure = _execute(_converting_stack(eps_exit=1.0))
        fp = FieldProfile(structure)
        res = fp.polarization_resolved("p")
        assert float(res["R_co"]) + float(res["R_cross"]) == pytest.approx(
            float(fp.reflectance("p")), abs=1e-9
        )
        assert float(res["T_co"]) + float(res["T_cross"]) == pytest.approx(
            float(fp.transmittance("p")), abs=1e-9
        )

    def test_rejects_mixed_polarization(self):
        fp = FieldProfile(_execute(_converting_stack()))
        with pytest.raises(ValueError):
            fp.polarization_resolved((1.0, 1.0))


class TestStokesFromFieldProfile:
    """Polarization state of the transverse field versus depth."""

    def test_keys_and_shapes(self):
        fp = FieldProfile(_execute(_converting_stack()))
        prof = fp.stokes_from_field_profile("p", n_points=60)
        for key in ("z", "S0", "S1", "S2", "S3", "azimuth", "ellipticity"):
            assert key in prof
        assert prof["S0"].shape == prof["z"].shape

    def test_s0_is_intensity(self):
        fp = FieldProfile(_execute(_converting_stack()))
        prof = fp.stokes_from_field_profile("p", n_points=60)
        # S0 == |Ex|^2 + |Ey|^2 from the underlying field profile
        field = fp.field_profile("p", n_points=60)
        s0 = np.abs(field["Ex"]) ** 2 + np.abs(field["Ey"]) ** 2
        assert np.allclose(prof["S0"], s0)

    def test_isotropic_p_stays_linear(self):
        # p-incidence through purely isotropic layers: Ey == 0 everywhere, so the
        # state stays linear-p (S2 = S3 = 0, S1 = S0).
        fp = FieldProfile(_execute(_ISO_STACK))
        prof = fp.stokes_from_field_profile("p", n_points=60)
        assert np.allclose(prof["S2"], 0.0, atol=1e-9)
        assert np.allclose(prof["S3"], 0.0, atol=1e-9)
        assert np.allclose(prof["S1"], prof["S0"], atol=1e-9)

    def test_rejects_thickness_sweep(self):
        layers = _converting_stack()
        layers[2]["thickness"] = [0.5, 1.0, 1.5]
        fp = FieldProfile(_execute(layers))
        with pytest.raises(ValueError):
            fp.stokes_from_field_profile("p")


class TestTransmissionMueller:
    """Transmission Mueller matrix and its symmetric-system validity."""

    def test_shape_and_real(self):
        structure = _execute(_converting_stack())
        mueller = Mueller(structure)
        m = mueller.calculate_transmission_mueller_matrix()
        assert m.shape == (4, 4)
        assert np.isrealobj(m)

    def test_symmetric_system_matches_flux_transmittance(self):
        # prism eps == substrate eps == 2 (symmetric): the |t|^2 intensity that the
        # Mueller matrix encodes equals the Poynting power transmittance, so
        # M_t[0, 0] (unpolarized -> total) == mean of the flux T over p and s.
        layers = [
            {"type": "Ambient Incident Layer", "permittivity": 2.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
            {
                "type": "Crystal Layer",
                "material": "Calcite",
                "thickness": 1.0,
                "rotationY": 90,
                "rotationZ": 30,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
        ]
        structure = _execute(layers)
        m = Mueller(structure).calculate_transmission_mueller_matrix()
        fp = FieldProfile(structure)
        flux_unpolarized = 0.5 * (float(fp.transmittance("p")) + float(fp.transmittance("s")))
        assert float(m[0, 0]) == pytest.approx(flux_unpolarized, abs=1e-9)

    def test_reflection_mueller_unchanged_by_refactor(self):
        # the shared _mueller_from_jones must reproduce the reflection Mueller.
        structure = _execute(_converting_stack())
        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()
        direct = mueller._mueller_from_jones(
            structure.r_pp, structure.r_ps, structure.r_sp, structure.r_ss
        )
        assert np.allclose(mueller.mueller_matrix, direct)
