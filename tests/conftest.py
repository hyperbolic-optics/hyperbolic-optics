"""
Pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def simple_payload():
    """Fixture for simple scenario payload."""
    return {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.1,
                "permittivity": 1.0,
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0,
            },
        ],
    }


@pytest.fixture
def incident_payload():
    """Fixture for incident scenario payload."""
    return {
        "ScenarioData": {
            "type": "Incident",
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0,
            },
        ],
    }


@pytest.fixture
def azimuthal_payload():
    """Fixture for azimuthal scenario payload."""
    return {
        "ScenarioData": {
            "type": "Azimuthal",
            "incidentAngle": 40,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 12.5},
            {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 90,
                "rotationZ": 0,
            },
        ],
    }


@pytest.fixture
def dispersion_payload():
    """Fixture for dispersion scenario payload."""
    return {
        "ScenarioData": {"type": "Dispersion", "frequency": 1460.0},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 25.0},
            {
                "type": "Isotropic Middle-Stack Layer",
                "thickness": 0.5,
                "permittivity": 1.0,
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationX": 0,
                "rotationY": 70,
                "rotationZ": 0,
            },
        ],
    }


@pytest.fixture
def arbitrary_material_payload():
    """Fixture for arbitrary material payload."""
    return {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 22.5},
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": {
                    "eps_xx": {"real": 2.27, "imag": 0.001},
                    "eps_yy": {"real": -4.84, "imag": 0.755},
                    "eps_zz": {"real": -4.84, "imag": 0.755},
                    "eps_xy": {"real": 0.0, "imag": 0.0},
                    "eps_xz": {"real": 0.0, "imag": 0.0},
                    "eps_yz": {"real": 0.0, "imag": 0.0},
                },
                "rotationX": 0,
                "rotationY": 0,
                "rotationZ": 0.0,
            },
        ],
    }
