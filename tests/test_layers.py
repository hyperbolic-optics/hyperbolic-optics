"""
Tests for layer creation and configuration.
"""

import numpy as np
import pytest

from hyperbolic_optics.layers import (
    AirGapLayer,
    CrystalLayer,
    LayerFactory,
    PrismLayer,
    SemiInfiniteCrystalLayer,
)
from hyperbolic_optics.scenario import ScenarioSetup


@pytest.fixture
def simple_scenario():
    """Create a simple scenario for testing."""
    data = {
        "type": "Simple",
        "incidentAngle": 45.0,
        "azimuthal_angle": 0.0,
        "frequency": 1460.0,
    }
    return ScenarioSetup(data)


@pytest.fixture
def incident_scenario():
    """Create an incident scenario for testing."""
    data = {"type": "Incident"}
    return ScenarioSetup(data)


class TestLayerFactory:
    """Test layer factory functionality."""

    def test_layer_factory_prism(self, simple_scenario):
        """Test creating a prism layer."""
        factory = LayerFactory()
        layer_data = {"type": "Ambient Incident Layer", "permittivity": 50.0}

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = factory.create_layer(layer_data, simple_scenario, kx, k0)
        assert isinstance(layer, PrismLayer)

    def test_layer_factory_airgap(self, simple_scenario):
        """Test creating an air gap layer."""
        factory = LayerFactory()
        layer_data = {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.5,
            "permittivity": 1.0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = factory.create_layer(layer_data, simple_scenario, kx, k0)
        assert isinstance(layer, AirGapLayer)

    def test_layer_factory_crystal(self, simple_scenario):
        """Test creating a crystal layer."""
        factory = LayerFactory()
        layer_data = {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 1.0,
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = factory.create_layer(layer_data, simple_scenario, kx, k0)
        assert isinstance(layer, CrystalLayer)

    def test_layer_factory_semi_infinite(self, simple_scenario):
        """Test creating a semi-infinite layer."""
        factory = LayerFactory()
        layer_data = {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = factory.create_layer(layer_data, simple_scenario, kx, k0)
        assert isinstance(layer, SemiInfiniteCrystalLayer)


class TestPrismLayer:
    """Test prism layer functionality."""

    def test_prism_initialization(self, simple_scenario):
        """Test prism layer initializes correctly."""
        layer_data = {"type": "Ambient Incident Layer", "permittivity": 50.0}

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = PrismLayer(layer_data, simple_scenario, kx, k0)
        assert layer.eps_prism == 50.0
        assert layer.matrix is not None

    def test_prism_matrix_shape_simple(self, simple_scenario):
        """Test prism matrix shape for simple scenario."""
        layer_data = {"type": "Ambient Incident Layer", "permittivity": 50.0}

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = PrismLayer(layer_data, simple_scenario, kx, k0)
        # For simple scenario, should be [4, 4]
        assert layer.matrix.shape == (4, 4)

    def test_prism_matrix_shape_incident(self, incident_scenario):
        """Test prism matrix shape for incident scenario."""
        layer_data = {"type": "Ambient Incident Layer", "permittivity": 50.0}

        kx = np.sqrt(50.0) * np.sin(incident_scenario.incident_angle)
        k0 = 1460.0 * 2.0 * np.pi  # Use a fixed frequency

        layer = PrismLayer(layer_data, incident_scenario, kx, k0)
        # For incident scenario, should be [360, 1, 4, 4]
        assert layer.matrix.shape == (360, 1, 4, 4)


class TestAirGapLayer:
    """Test air gap layer functionality."""

    def test_airgap_initialization(self, simple_scenario):
        """Test air gap layer initializes correctly."""
        layer_data = {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.5,
            "permittivity": 1.0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = AirGapLayer(layer_data, simple_scenario, kx, k0)
        assert layer.thickness == 0.5 * 1e-4  # Converted to cm
        assert layer.permittivity == complex(1.0, 0)

    def test_airgap_complex_permittivity(self, simple_scenario):
        """Test air gap with complex permittivity."""
        layer_data = {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.5,
            "permittivity": {"real": 2.5, "imag": 0.1},
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = AirGapLayer(layer_data, simple_scenario, kx, k0)
        assert layer.permittivity == complex(2.5, 0.1)

    def test_airgap_tensors(self, simple_scenario):
        """Test that air gap has proper isotropic tensors."""
        layer_data = {
            "type": "Isotropic Middle-Stack Layer",
            "thickness": 0.5,
            "permittivity": 2.0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = AirGapLayer(layer_data, simple_scenario, kx, k0)

        # Check that tensors are diagonal
        assert layer.eps_tensor.shape == (3, 3)
        assert layer.mu_tensor.shape == (3, 3)


class TestCrystalLayer:
    """Test crystal layer functionality."""

    def test_crystal_initialization(self, simple_scenario):
        """Test crystal layer initializes correctly."""
        layer_data = {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 1.0,
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = CrystalLayer(layer_data, simple_scenario, kx, k0)
        assert layer.material.name == "Calcite-Upper"
        assert layer.thickness == 1.0 * 1e-4  # Converted to cm

    def test_crystal_rotations(self, simple_scenario):
        """Test crystal rotation angles."""
        layer_data = {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 1.0,
            "rotationX": 30,
            "rotationY": 60,
            "rotationZ": 90,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = CrystalLayer(layer_data, simple_scenario, kx, k0)

        # Rotations should be converted to radians
        assert np.isclose(layer.rotationX, np.radians(30))
        assert np.isclose(layer.rotationY, np.radians(60), atol=1e-7)

    def test_crystal_tensors(self, simple_scenario):
        """Test that crystal has both eps and mu tensors."""
        layer_data = {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 1.0,
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = CrystalLayer(layer_data, simple_scenario, kx, k0)

        assert layer.eps_tensor is not None
        assert layer.mu_tensor is not None
        assert layer.eps_tensor.shape[-2:] == (3, 3)
        assert layer.mu_tensor.shape[-2:] == (3, 3)


class TestSemiInfiniteCrystalLayer:
    """Test semi-infinite crystal layer functionality."""

    def test_semi_infinite_initialization(self, simple_scenario):
        """Test semi-infinite layer initializes correctly."""
        layer_data = {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = SemiInfiniteCrystalLayer(layer_data, simple_scenario, kx, k0)
        assert layer.material.name == "Calcite-Upper"
        assert layer.thickness is None

    def test_semi_infinite_matrix(self, simple_scenario):
        """Test semi-infinite layer matrix."""
        layer_data = {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationX": 0,
            "rotationY": 90,
            "rotationZ": 0,
        }

        kx = np.sqrt(50.0) * np.sin(simple_scenario.incident_angle)
        k0 = simple_scenario.frequency * 2.0 * np.pi

        layer = SemiInfiniteCrystalLayer(layer_data, simple_scenario, kx, k0)
        assert layer.matrix is not None
