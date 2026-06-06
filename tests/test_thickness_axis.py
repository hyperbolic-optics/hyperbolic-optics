"""Tests for thickness as a canonical scenario axis (the 4th batch axis, T).

A list-valued layer ``thickness`` sweeps the canonical ``T`` axis in a single
``execute`` (one eigendecomposition, the thickness broadcast only through the
propagation phase). The load-bearing check is that this vectorized axis agrees
elementwise with the already-tested ``ThicknessSweep`` re-run helper.
"""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.sweep import ThicknessSweep

THICKNESSES = [0.4, 0.9, 1.6, 2.3]


def _payload(scenario, thickness):
    """prism / air gap / calcite film (thickness param) / vacuum."""
    return {
        "ScenarioData": scenario,
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 25.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
            {
                "type": "Crystal Layer",
                "material": "Calcite",
                "thickness": thickness,
                "rotationY": 90,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
        ],
    }


_SIMPLE = {"type": "Simple", "incidentAngle": 30.0, "azimuthal_angle": 0.0, "frequency": 1460.0}
_INCIDENT = {"type": "Incident"}
_DISPERSION = {"type": "Dispersion", "frequency": 1460.0}


class TestThicknessAxisShapes:
    """The T axis appends trailing; T=1 leaves shapes untouched."""

    def test_simple_with_list_is_1d(self):
        structure = Structure()
        structure.execute(_payload(_SIMPLE, THICKNESSES))
        assert structure.r_pp.shape == (len(THICKNESSES),)
        # canonical transfer matrix is [A, B, F, T, 4, 4]
        assert structure.transfer_matrix.shape == (1, 1, 1, len(THICKNESSES), 4, 4)

    def test_simple_scalar_unchanged(self):
        structure = Structure()
        structure.execute(_payload(_SIMPLE, 1.0))
        assert np.ndim(structure.r_pp) == 0  # scalar, T squeezed away

    def test_incident_appends_t_axis(self):
        structure = Structure()
        structure.execute(_payload(_INCIDENT, THICKNESSES))
        assert structure.r_pp.shape == (410, 360, len(THICKNESSES))


class TestMatchesThicknessSweepHelper:
    """The vectorized T axis equals the re-run ThicknessSweep helper elementwise."""

    @pytest.mark.parametrize("scenario", [_SIMPLE, _INCIDENT, _DISPERSION])
    def test_reflection_matches(self, scenario):
        structure = Structure()
        structure.execute(_payload(scenario, THICKNESSES))
        helper = ThicknessSweep(_payload(scenario, 1.0), layer_index=2, thicknesses=THICKNESSES)
        # helper stacks thickness as axis 0; canonical run has it trailing.
        canonical = np.moveaxis(structure.r_pp, -1, 0)
        assert np.allclose(canonical, helper.r_pp, atol=1e-10)

    def test_power_quantities_match(self):
        structure = Structure()
        structure.execute(_payload(_SIMPLE, THICKNESSES))
        fp = FieldProfile(structure)
        helper = ThicknessSweep(_payload(_SIMPLE, 1.0), layer_index=2, thicknesses=THICKNESSES)
        assert np.allclose(fp.transmittance("p"), helper.transmittance("p"), atol=1e-10)
        assert np.allclose(fp.reflectance("p"), helper.reflectance("p"), atol=1e-10)


class TestThicknessAxisPhysics:
    """Energy conservation across the whole T axis."""

    def test_conservation(self):
        structure = Structure()
        structure.execute(_payload(_SIMPLE, THICKNESSES))
        fp = FieldProfile(structure)
        r = fp.reflectance("p")
        t = fp.transmittance("p")
        a = fp.summary("p")["total_absorption"]
        assert r.shape == (len(THICKNESSES),)
        assert np.allclose(r + t + a, 1.0, atol=1e-9)


class TestGuards:
    """Validation: one swept layer; field_profile undefined under a sweep."""

    def test_two_swept_layers_raise(self):
        payload = _payload(_SIMPLE, THICKNESSES)
        payload["Layers"][1]["thickness"] = [0.3, 0.6, 0.9, 1.2]  # also a list
        with pytest.raises(ValueError, match="one layer"):
            Structure().execute(payload)

    def test_field_profile_rejects_sweep(self):
        structure = Structure()
        structure.execute(_payload(_SIMPLE, THICKNESSES))
        with pytest.raises(ValueError, match="sweeping a layer thickness"):
            FieldProfile(structure).field_profile("p")

    def test_field_profile_ok_for_single_thickness(self):
        # a 1-element list (T=1) still profiles fine
        structure = Structure()
        structure.execute(_payload(_SIMPLE, [1.0]))
        prof = FieldProfile(structure).field_profile("p", n_points=20)
        assert prof["Ex"].ndim >= 1
