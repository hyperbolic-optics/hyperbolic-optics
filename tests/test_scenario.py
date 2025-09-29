"""
Tests for scenario creation and configuration.
"""

import numpy as np

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
