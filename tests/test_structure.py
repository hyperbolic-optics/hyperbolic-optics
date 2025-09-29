"""
Tests for the Structure class and overall simulation workflow.
"""

import numpy as np

from hyperbolic_optics.structure import Structure


class TestStructureBasicFunctionality:
    """Test basic Structure functionality."""

    def test_structure_initialization(self):
        """Test that Structure initializes correctly."""
        structure = Structure()
        assert structure.scenario is None
        assert structure.layers == []
        assert structure.r_pp is None

    def test_simple_scenario_execution(self, simple_payload):
        """Test simple scenario executes without errors."""
        structure = Structure()
        structure.execute(simple_payload)

        # Check that reflection coefficients are calculated
        assert structure.r_pp is not None
        assert structure.r_ss is not None
        assert structure.r_ps is not None
        assert structure.r_sp is not None

        # Check they are complex numbers (scalars for Simple scenario)
        assert np.iscomplexobj(structure.r_pp)
        assert np.iscomplexobj(structure.r_ss)

    def test_incident_scenario_execution(self, incident_payload):
        """Test incident scenario executes and produces correct shape."""
        structure = Structure()
        structure.execute(incident_payload)

        # Should produce arrays with 360 angle points
        assert structure.r_pp.shape == (410, 360)
        assert structure.r_ss.shape == (410, 360)
        assert structure.r_ps.shape == (410, 360)
        assert structure.r_sp.shape == (410, 360)

    def test_azimuthal_scenario_execution(self, azimuthal_payload):
        """Test azimuthal scenario executes and produces correct shape."""
        structure = Structure()
        structure.execute(azimuthal_payload)

        # Should produce arrays with 360 azimuthal points
        assert structure.r_pp.shape == (410, 360)
        assert structure.r_ss.shape == (410, 360)
        assert structure.r_ps.shape == (410, 360)
        assert structure.r_sp.shape == (410, 360)

    def test_dispersion_scenario_execution(self, dispersion_payload):
        """Test dispersion scenario executes and produces correct shape."""
        structure = Structure()
        structure.execute(dispersion_payload)

        # Should produce arrays with 180x480 points
        assert structure.r_pp.shape == (180, 480)
        assert structure.r_ss.shape == (180, 480)
        assert structure.r_ps.shape == (180, 480)
        assert structure.r_sp.shape == (180, 480)


class TestReflectionCoefficients:
    """Test properties of reflection coefficients."""

    def test_reflection_coefficients_are_complex(self, simple_payload):
        """Test that reflection coefficients are complex."""
        structure = Structure()
        structure.execute(simple_payload)

        assert np.iscomplexobj(structure.r_pp)
        assert np.iscomplexobj(structure.r_ss)
        assert np.iscomplexobj(structure.r_ps)
        assert np.iscomplexobj(structure.r_sp)

    def test_reflectivity_bounds(self, simple_payload):
        """Test that reflectivity values are physically reasonable (0 to 1)."""
        structure = Structure()
        structure.execute(simple_payload)

        R_pp = abs(structure.r_pp) ** 2
        R_ss = abs(structure.r_ss) ** 2
        R_ps = abs(structure.r_ps) ** 2
        R_sp = abs(structure.r_sp) ** 2

        # All reflectivities should be between 0 and 1
        assert 0 <= R_pp <= 1
        assert 0 <= R_ss <= 1
        assert 0 <= R_ps <= 1
        assert 0 <= R_sp <= 1

        # Total reflectivity should not exceed 1
        R_total = R_pp + R_ps + R_ss + R_sp
        assert R_total <= 2  # Maximum is 2 (one for each polarization)

    def test_reflectivity_bounds_array(self, incident_payload):
        """Test reflectivity bounds for array scenarios."""
        structure = Structure()
        structure.execute(incident_payload)

        R_pp = np.abs(structure.r_pp) ** 2
        R_ss = np.abs(structure.r_ss) ** 2

        # Check bounds
        tolerance = 1e-6
        assert np.all(R_pp >= -tolerance)  # Allow small negative due to numerical error
        assert np.all(R_pp <= 1 + tolerance)  # Allow slightly above 1
        assert np.all(R_ss >= -tolerance)
        assert np.all(R_ss <= 1 + tolerance)


class TestArbitraryMaterial:
    """Test arbitrary material functionality."""

    def test_arbitrary_material_execution(self, arbitrary_material_payload):
        """Test that arbitrary materials work correctly."""
        structure = Structure()
        structure.execute(arbitrary_material_payload)

        assert structure.r_pp is not None
        assert structure.r_ss is not None
        assert np.iscomplexobj(structure.r_pp)

    def test_arbitrary_material_reflectivity(self, arbitrary_material_payload):
        """Test arbitrary material produces reasonable reflectivity."""
        structure = Structure()
        structure.execute(arbitrary_material_payload)

        R_pp = abs(structure.r_pp) ** 2
        R_ss = abs(structure.r_ss) ** 2

        assert 0 <= R_pp <= 1
        assert 0 <= R_ss <= 1


class TestStructureAttributes:
    """Test that structure attributes are set correctly."""

    def test_scenario_attributes(self, simple_payload):
        """Test that scenario attributes are correctly set."""
        structure = Structure()
        structure.execute(simple_payload)

        assert structure.incident_angle is not None
        assert structure.frequency is not None
        assert structure.k_x is not None
        assert structure.k_0 is not None

    def test_layer_creation(self, simple_payload):
        """Test that layers are created correctly."""
        structure = Structure()
        structure.execute(simple_payload)

        # Should have 3 layers
        assert len(structure.layers) == 3

    def test_prism_permittivity(self, simple_payload):
        """Test that prism permittivity is set correctly."""
        structure = Structure()
        structure.execute(simple_payload)

        assert structure.eps_prism == 50.0
