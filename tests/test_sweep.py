"""Tests for the thickness-sweep helper."""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.sweep import ThicknessSweep


def _film_payload(scenario, thickness=1.0):
    """prism / air gap / finite calcite film / vacuum, for the given scenario."""
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


class TestThicknessSweepShapes:
    """The thickness axis is prepended to the usual presentation shape."""

    def test_simple_scalar_becomes_1d(self):
        thicknesses = [0.5, 1.0, 1.5, 2.0]
        sweep = ThicknessSweep(_film_payload(_SIMPLE), layer_index=2, thicknesses=thicknesses)
        assert len(sweep) == 4
        assert sweep.r_pp.shape == (4,)
        assert sweep.transmittance("p").shape == (4,)
        assert np.allclose(sweep.thicknesses, thicknesses)

    def test_incident_prepends_thickness_axis(self):
        sweep = ThicknessSweep(_film_payload(_INCIDENT), layer_index=2, thicknesses=[0.5, 1.0, 1.5])
        # Incident r_pp is (410, 360); sweep prepends the thickness axis.
        for coeff in (sweep.r_pp, sweep.r_ss, sweep.r_ps, sweep.r_sp):
            assert coeff.shape == (3, 410, 360)
        assert sweep.reflectance("p").shape == (3, 410, 360)
        coeffs = sweep.transmission_coefficients()
        assert coeffs["t_pp"].shape == (3, 410, 360)


class TestThicknessSweepMatchesSingleRuns:
    """Each slice equals an independent single-thickness run."""

    def test_slice_matches_direct(self):
        thicknesses = [0.5, 1.3]
        sweep = ThicknessSweep(_film_payload(_SIMPLE), layer_index=2, thicknesses=thicknesses)
        for i, thickness in enumerate(thicknesses):
            structure = Structure()
            structure.execute(_film_payload(_SIMPLE, thickness=thickness))
            assert np.allclose(sweep.r_pp[i], structure.r_pp, atol=1e-12)
            assert np.allclose(
                sweep.transmittance("p")[i],
                float(FieldProfile(structure).transmittance("p")),
                atol=1e-12,
            )

    def test_does_not_mutate_payload(self):
        payload = _film_payload(_SIMPLE, thickness=1.0)
        ThicknessSweep(payload, layer_index=2, thicknesses=[0.5, 2.0])
        assert payload["Layers"][2]["thickness"] == 1.0  # original untouched


class TestThicknessSweepPhysics:
    """Energy conservation holds at every thickness; the swept layer absorbs."""

    def test_conservation_per_thickness(self):
        sweep = ThicknessSweep(_film_payload(_SIMPLE), layer_index=2, thicknesses=[0.3, 0.8, 1.5])
        r = sweep.reflectance("p")
        t = sweep.transmittance("p")
        a = sweep.total_absorption("p")
        assert np.allclose(r + t + a, 1.0, atol=1e-9)

    def test_swept_layer_absorption_varies(self):
        sweep = ThicknessSweep(_film_payload(_SIMPLE), layer_index=2, thicknesses=[0.2, 1.0, 3.0])
        per_layer = sweep.layer_absorption("p")
        film = next(entry for entry in per_layer if entry["index"] == 2)
        assert film["absorptance"].shape == (3,)
        # a thicker absorbing film absorbs differently across the sweep
        assert not np.allclose(film["absorptance"][0], film["absorptance"][-1])


class TestThicknessSweepValidation:
    """Input validation."""

    def test_rejects_non_finite_layer(self):
        with pytest.raises(ValueError):
            ThicknessSweep(_film_payload(_SIMPLE), layer_index=0, thicknesses=[1.0])  # prism

    def test_rejects_out_of_range(self):
        with pytest.raises(IndexError):
            ThicknessSweep(_film_payload(_SIMPLE), layer_index=9, thicknesses=[1.0])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            ThicknessSweep(_film_payload(_SIMPLE), layer_index=2, thicknesses=[])
