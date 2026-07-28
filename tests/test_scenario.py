"""
Tests for scenario creation and configuration.
"""

import numpy as np
import pytest

from hyperbolic_optics.scenario import ScenarioSetup


class TestScenarioInitialization:
    """Test scenario initialization."""

    def test_simple_scenario(self):
        """Test simple scenario creation."""
        data = {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 30.0,
            "frequency": 1460.0,
        }
        scenario = ScenarioSetup(data)

        assert scenario.type == "Simple"
        assert scenario.incident_angle is not None
        assert scenario.azimuthal_angle is not None
        assert scenario.frequency == 1460.0

    def test_incident_scenario(self):
        """Test incident scenario creation."""
        data = {"type": "Incident"}
        scenario = ScenarioSetup(data)

        assert scenario.type == "Incident"
        assert scenario.incident_angle is not None
        assert len(scenario.incident_angle) == 360

    def test_azimuthal_scenario(self):
        """Test azimuthal scenario creation."""
        data = {"type": "Azimuthal", "incidentAngle": 40.0}
        scenario = ScenarioSetup(data)

        assert scenario.type == "Azimuthal"
        assert scenario.azimuthal_angle is not None
        assert len(scenario.azimuthal_angle) == 360

    def test_dispersion_scenario(self):
        """Test dispersion scenario creation."""
        data = {"type": "Dispersion", "frequency": 1460.0}
        scenario = ScenarioSetup(data)

        assert scenario.type == "Dispersion"
        assert scenario.incident_angle is not None
        assert scenario.azimuthal_angle is not None
        assert len(scenario.incident_angle) == 180
        assert len(scenario.azimuthal_angle) == 480


class TestScenarioRanges:
    """Test that scenario ranges are physically reasonable."""

    def test_incident_angle_range(self):
        """Test incident angle range."""
        data = {"type": "Incident"}
        scenario = ScenarioSetup(data)

        # Should span from -π/2 to π/2
        assert np.min(scenario.incident_angle) > -np.pi / 2
        assert np.max(scenario.incident_angle) < np.pi / 2

    def test_azimuthal_angle_range(self):
        """Test azimuthal angle range."""
        data = {"type": "Azimuthal", "incidentAngle": 40.0}
        scenario = ScenarioSetup(data)

        # Should span from 0 to 2π
        assert np.min(scenario.azimuthal_angle) >= 0
        assert np.max(scenario.azimuthal_angle) <= 2 * np.pi

    def test_dispersion_incident_range(self):
        """Test dispersion scenario incident angle range."""
        data = {"type": "Dispersion", "frequency": 1460.0}
        scenario = ScenarioSetup(data)

        # Should span from 0 to π/2
        assert np.min(scenario.incident_angle) >= 0
        assert np.max(scenario.incident_angle) <= np.pi / 2


class TestScenarioTypes:
    """Test data types of scenario attributes."""

    def test_simple_scenario_types(self):
        """Test that simple scenario has scalar values."""
        data = {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 30.0,
            "frequency": 1460.0,
        }
        scenario = ScenarioSetup(data)

        # Should be scalars
        assert np.isscalar(scenario.incident_angle) or scenario.incident_angle.shape == ()
        assert np.isscalar(scenario.azimuthal_angle) or scenario.azimuthal_angle.shape == ()
        assert isinstance(scenario.frequency, float)

    def test_incident_scenario_types(self):
        """Test that incident scenario has array values."""
        data = {"type": "Incident"}
        scenario = ScenarioSetup(data)

        # incident_angle should be array
        assert isinstance(scenario.incident_angle, np.ndarray)
        assert scenario.incident_angle.ndim == 1


class TestAngularAxisControl:
    """Swept angular axes are user-controllable, and a scalar is not discarded."""

    def test_resolution_override(self):
        assert ScenarioSetup({"type": "Incident", "polar_points": 41}).incident_angle.shape == (41,)
        azimuthal = ScenarioSetup(
            {"type": "Azimuthal", "incidentAngle": 30.0, "azimuthal_points": 12}
        ).azimuthal_angle
        assert azimuthal.shape == (12,)

    def test_defaults_are_unchanged(self):
        """The historical grids are what a payload without overrides still gets."""
        assert ScenarioSetup({"type": "Incident"}).incident_angle.shape == (360,)
        dispersion = ScenarioSetup({"type": "Dispersion", "frequency": 1460.0})
        assert dispersion.incident_angle.shape == (180,)
        assert dispersion.azimuthal_angle.shape == (480,)

    def test_explicit_range_in_degrees(self):
        scenario = ScenarioSetup(
            {"type": "Incident", "incidentAngle": {"min": 20.0, "max": 60.0, "points": 5}}
        )
        assert np.allclose(np.degrees(scenario.incident_angle), [20.0, 30.0, 40.0, 50.0, 60.0])

    def test_explicit_list_in_degrees(self):
        scenario = ScenarioSetup({"type": "Incident", "incidentAngle": [10.0, 20.0, 30.0]})
        assert np.allclose(np.degrees(scenario.incident_angle), [10.0, 20.0, 30.0])

    def test_scalar_on_a_swept_axis_warns(self):
        """It cannot be honoured, and silently sweeping instead is the worst option."""
        with pytest.warns(UserWarning, match="cannot be honoured"):
            scenario = ScenarioSetup({"type": "Incident", "incidentAngle": 45.0})
        assert scenario.incident_angle.shape == (360,)

    def test_degenerate_resolution_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            ScenarioSetup({"type": "Incident", "polar_points": 1})
