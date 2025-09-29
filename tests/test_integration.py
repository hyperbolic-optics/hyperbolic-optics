"""
Integration tests for complete workflows.
"""

import numpy as np
import pytest

from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure


@pytest.mark.integration
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    def test_simple_workflow(self, simple_payload):
        """Test complete simple scenario workflow."""
        # Create structure
        structure = Structure()
        structure.execute(simple_payload)

        # Calculate Mueller matrix
        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=45)
        mueller.add_optical_component("anisotropic_sample")

        # Get all parameters
        params = mueller.get_all_parameters()

        # Verify we got all expected outputs
        assert "S0" in params
        assert "DOP" in params
        assert params["S0"] is not None

    def test_incident_workflow(self, incident_payload):
        """Test complete incident scenario workflow."""
        structure = Structure()
        structure.execute(incident_payload)

        # Verify structure
        assert structure.r_pp.shape == (410, 360)

        # Mueller analysis
        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        stokes = mueller.calculate_stokes_parameters()
        assert stokes.shape == (410, 360, 4)

    def test_azimuthal_workflow(self, azimuthal_payload):
        """Test complete azimuthal scenario workflow."""
        structure = Structure()
        structure.execute(azimuthal_payload)

        assert structure.r_pp.shape == (410, 360)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("circular", handedness="right")
        mueller.add_optical_component("anisotropic_sample")

        params = mueller.get_all_parameters()
        assert params["S0"].shape == (410, 360)

    def test_dispersion_workflow(self, dispersion_payload):
        """Test complete dispersion scenario workflow."""
        structure = Structure()
        structure.execute(dispersion_payload)

        assert structure.r_pp.shape == (180, 480)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=45)
        mueller.add_optical_component("anisotropic_sample")

        reflectivity = mueller.get_reflectivity()
        assert reflectivity.shape == (180, 480)

    def test_arbitrary_material_workflow(self, arbitrary_material_payload):
        """Test workflow with arbitrary material."""
        structure = Structure()
        structure.execute(arbitrary_material_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        params = mueller.get_all_parameters()
        assert all(key in params for key in ["S0", "S1", "S2", "S3", "DOP"])


@pytest.mark.integration
class TestMultipleComponents:
    """Test workflows with multiple optical components."""

    def test_polarizer_sample_sequence(self, simple_payload):
        """Test sequence with polarizer before sample."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("linear_polarizer", 45)
        mueller.add_optical_component("anisotropic_sample")

        reflectivity = mueller.get_reflectivity()
        assert reflectivity is not None
        assert 0 <= reflectivity <= 1

    def test_sample_polarizer_sequence(self, simple_payload):
        """Test sequence with polarizer after sample."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")
        mueller.add_optical_component("linear_polarizer", 90)

        reflectivity = mueller.get_reflectivity()
        assert reflectivity is not None

    def test_wave_plate_sample_sequence(self, simple_payload):
        """Test sequence with wave plate before sample."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("quarter_wave_plate", 45)
        mueller.add_optical_component("anisotropic_sample")

        params = mueller.get_all_parameters()
        assert params["DOP"] is not None


@pytest.mark.integration
class TestDifferentMaterials:
    """Test workflows with different materials."""

    def test_quartz_workflow(self):
        """Test workflow with Quartz."""
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 0.0,
                "frequency": 500.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Quartz",
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
            ],
        }

        structure = Structure()
        structure.execute(payload)

        R_pp = abs(structure.r_pp) ** 2
        R_ss = abs(structure.r_ss) ** 2

        assert 0 <= R_pp <= 1
        assert 0 <= R_ss <= 1

    def test_sapphire_workflow(self):
        """Test workflow with Sapphire."""
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 0.0,
                "frequency": 500.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Sapphire",
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
            ],
        }

        structure = Structure()
        structure.execute(payload)

        assert structure.r_pp is not None
        assert structure.r_ss is not None

    def test_gallium_oxide_workflow(self):
        """Test workflow with Gallium Oxide."""
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 0.0,
                "frequency": 600.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "GalliumOxide",
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
            ],
        }

        structure = Structure()
        structure.execute(payload)

        assert structure.r_pp is not None


@pytest.mark.integration
class TestMultilayerStructures:
    """Test multilayer structure workflows."""

    def test_three_layer_structure(self):
        """Test structure with crystal layer between air gaps."""
        payload = {
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
                    "type": "Crystal Layer",
                    "material": "Calcite",
                    "thickness": 0.5,
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
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

        structure = Structure()
        structure.execute(payload)

        assert len(structure.layers) == 5
        assert structure.r_pp is not None

    def test_complex_permittivity_airgap(self):
        """Test air gap with complex permittivity."""
        payload = {
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
                    "permittivity": {"real": 2.5, "imag": 0.1},
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

        structure = Structure()
        structure.execute(payload)

        assert structure.r_pp is not None


@pytest.mark.integration
class TestPhysicalConsistency:
    """Test physical consistency across different scenarios."""

    def test_reciprocity(self, simple_payload):
        """Test that r_ps and r_sp have proper relationship."""
        structure = Structure()
        structure.execute(simple_payload)

        # For reciprocal media, certain relationships should hold
        # This is a basic check that both exist and are calculated
        assert structure.r_ps is not None
        assert structure.r_sp is not None

    def test_energy_conservation_across_scenarios(self):
        """Test energy conservation for different scenarios."""
        scenarios = ["Simple", "Incident", "Azimuthal"]

        for scenario_type in scenarios:
            if scenario_type == "Simple":
                payload = {
                    "ScenarioData": {
                        "type": scenario_type,
                        "incidentAngle": 45.0,
                        "azimuthal_angle": 0.0,
                        "frequency": 1460.0,
                    },
                    "Layers": [
                        {"type": "Ambient Incident Layer", "permittivity": 50.0},
                        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                        {
                            "type": "Semi Infinite Anisotropic Layer",
                            "material": "Calcite",
                            "rotationX": 0,
                            "rotationY": 90,
                            "rotationZ": 0,
                        },
                    ],
                }
            elif scenario_type == "Incident":
                payload = {
                    "ScenarioData": {"type": scenario_type},
                    "Layers": [
                        {"type": "Ambient Incident Layer", "permittivity": 50.0},
                        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                        {
                            "type": "Semi Infinite Anisotropic Layer",
                            "material": "Calcite",
                            "rotationX": 0,
                            "rotationY": 90,
                            "rotationZ": 0,
                        },
                    ],
                }
            else:  # Azimuthal
                payload = {
                    "ScenarioData": {"type": scenario_type, "incidentAngle": 45.0},
                    "Layers": [
                        {"type": "Ambient Incident Layer", "permittivity": 50.0},
                        {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                        {
                            "type": "Semi Infinite Anisotropic Layer",
                            "material": "Calcite",
                            "rotationX": 0,
                            "rotationY": 90,
                            "rotationZ": 0,
                        },
                    ],
                }

            structure = Structure()
            structure.execute(payload)

            R_total = (
                np.abs(structure.r_pp) ** 2
                + np.abs(structure.r_ss) ** 2
                + np.abs(structure.r_ps) ** 2
                + np.abs(structure.r_sp) ** 2
            )

            # Total reflectivity should not exceed 2 (1 for each polarization)
            assert np.all(R_total <= 2.1)  # Small tolerance for numerical errors

    def test_rotation_invariance_properties(self):
        """Test properties under rotation for simple case."""
        base_payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 0.0,
                "frequency": 1460.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
            ],
        }

        structure1 = Structure()
        structure1.execute(base_payload)

        # Create payload with 180 degree rotation
        rotated_payload = base_payload.copy()
        rotated_payload["Layers"][2]["rotationZ"] = 180

        structure2 = Structure()
        structure2.execute(rotated_payload)

        # Both should produce valid results
        assert structure1.r_pp is not None
        assert structure2.r_pp is not None


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance and numerical stability."""

    def test_large_incident_array(self):
        """Test that large incident arrays complete in reasonable time."""
        payload = {
            "ScenarioData": {"type": "Incident"},
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationX": 0,
                    "rotationY": 90,
                    "rotationZ": 0,
                },
            ],
        }

        structure = Structure()
        structure.execute(payload)

        # Check for NaN or Inf values
        assert not np.any(np.isnan(structure.r_pp))
        assert not np.any(np.isinf(structure.r_pp))

    def test_dispersion_numerical_stability(self):
        """Test numerical stability for dispersion calculations."""
        payload = {
            "ScenarioData": {"type": "Dispersion", "frequency": 1460.0},
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.1},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationX": 0,
                    "rotationY": 70,
                    "rotationZ": 0,
                },
            ],
        }

        structure = Structure()
        structure.execute(payload)

        # Check for numerical issues
        assert not np.any(np.isnan(structure.r_pp))
        assert not np.any(np.isinf(structure.r_pp))
        assert np.all(np.isfinite(structure.r_pp))
