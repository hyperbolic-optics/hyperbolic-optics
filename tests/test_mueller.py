"""
Tests for Mueller matrix calculations and Stokes parameters.
"""

import numpy as np

from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure


class TestMuellerInitialization:
    """Test Mueller class initialization."""

    def test_mueller_initialization(self, simple_payload):
        """Test Mueller initializes correctly."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        assert mueller.structure == structure
        assert mueller.mueller_matrix is None
        assert mueller.stokes_parameters is None

    def test_default_incident_polarization(self, simple_payload):
        """Test default incident polarization is unpolarized."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        expected = np.array([1, 0, 0, 0], dtype=np.float64)
        assert np.allclose(mueller.incident_stokes, expected)


class TestPolarizationSettings:
    """Test polarization setting functionality."""

    def test_linear_polarization(self, simple_payload):
        """Test linear polarization at various angles."""
        structure = Structure()
        structure.execute(simple_payload)
        mueller = Mueller(structure)

        # p-polarization (0 degrees)
        mueller.set_incident_polarization("linear", angle=0)
        assert np.allclose(mueller.incident_stokes, [1, 1, 0, 0])

        # s-polarization (90 degrees)
        mueller.set_incident_polarization("linear", angle=90)
        assert np.allclose(mueller.incident_stokes, [1, -1, 0, 0])

        # 45 degrees
        mueller.set_incident_polarization("linear", angle=45)
        assert np.allclose(mueller.incident_stokes, [1, 0, 1, 0])

    def test_circular_polarization(self, simple_payload):
        """Test circular polarization."""
        structure = Structure()
        structure.execute(simple_payload)
        mueller = Mueller(structure)

        # Right circular
        mueller.set_incident_polarization("circular", handedness="right")
        assert np.allclose(mueller.incident_stokes, [1, 0, 0, 1])

        # Left circular
        mueller.set_incident_polarization("circular", handedness="left")
        assert np.allclose(mueller.incident_stokes, [1, 0, 0, -1])


class TestMuellerMatrix:
    """Test Mueller matrix calculations."""

    def test_mueller_matrix_calculation(self, simple_payload):
        """Test that Mueller matrix is calculated correctly."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        assert mueller.mueller_matrix is not None
        # For Simple scenario, matrix should be [4, 4]
        assert mueller.mueller_matrix.shape == (4, 4)

    def test_mueller_matrix_real(self, simple_payload):
        """Test that Mueller matrix is real-valued."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()

        # Mueller matrix should be real
        assert np.all(np.isreal(mueller.mueller_matrix))

    def test_mueller_matrix_incident_scenario(self, incident_payload):
        """Test Mueller matrix for incident scenario."""
        structure = Structure()
        structure.execute(incident_payload)

        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()

        # Should have shape [360, 410, 4, 4]
        assert mueller.mueller_matrix.shape == (410, 360, 4, 4)


class TestOpticalComponents:
    """Test optical component functionality."""

    def test_add_anisotropic_sample(self, simple_payload):
        """Test adding anisotropic sample component."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.add_optical_component("anisotropic_sample")

        assert len(mueller.optical_components) == 1
        assert mueller.anisotropic_sample_added is True

    def test_linear_polarizer(self, simple_payload):
        """Test linear polarizer component."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        polarizer = mueller.linear_polarizer(0)

        assert polarizer.shape == (4, 4)
        assert np.all(np.isreal(polarizer))

    def test_quarter_wave_plate(self, simple_payload):
        """Test quarter-wave plate component."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        qwp = mueller.quarter_wave_plate(45)

        assert qwp.shape == (4, 4)
        assert np.all(np.isreal(qwp))

    def test_half_wave_plate(self, simple_payload):
        """Test half-wave plate component."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        hwp = mueller.half_wave_plate(45)

        assert hwp.shape == (4, 4)
        assert np.all(np.isreal(hwp))


class TestStokesParameters:
    """Test Stokes parameter calculations."""

    def test_stokes_calculation_simple(self, simple_payload):
        """Test Stokes parameter calculation for simple scenario."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")
        stokes = mueller.calculate_stokes_parameters()

        # Should be a 1D array of length 4
        assert stokes.shape == (4,)
        assert np.all(np.isreal(stokes))

    def test_stokes_calculation_incident(self, incident_payload):
        """Test Stokes parameter calculation for incident scenario."""
        structure = Structure()
        structure.execute(incident_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")
        stokes = mueller.calculate_stokes_parameters()

        # Should be [360, 410, 4]
        assert stokes.shape == (410, 360, 4)

    def test_s0_is_reflectance(self, simple_payload):
        """Test that S0 equals total reflectance."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        reflectivity = mueller.get_reflectivity()
        stokes = mueller.get_stokes_parameters()

        assert np.allclose(reflectivity, stokes["S0"])

    def test_dop_bounds(self, simple_payload):
        """Test that degree of polarization is between 0 and 1."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        dop = mueller.get_degree_of_polarisation()

        # DOP should be clipped to [0, 1]
        assert np.all(dop >= 0)
        assert np.all(dop <= 1)

    def test_get_all_parameters(self, simple_payload):
        """Test getting all parameters at once."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        params = mueller.get_all_parameters()

        # Should contain all keys
        expected_keys = ["S0", "S1", "S2", "S3", "DOP", "Ellipticity", "Azimuth"]
        for key in expected_keys:
            assert key in params


class TestMuellerReset:
    """Test Mueller reset functionality."""

    def test_reset(self, simple_payload):
        """Test that reset clears all state."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=45)
        mueller.add_optical_component("anisotropic_sample")
        mueller.calculate_stokes_parameters()

        mueller.reset()

        assert mueller.mueller_matrix is None
        assert mueller.stokes_parameters is None
        assert len(mueller.optical_components) == 0
        assert mueller.anisotropic_sample_added is False
        assert np.allclose(mueller.incident_stokes, [1, 0, 0, 0])


class TestPhysicalConstraints:
    """Test that Mueller matrix satisfies physical constraints."""

    def test_energy_conservation(self, simple_payload):
        """Test that total reflectance doesn't exceed 1."""
        structure = Structure()
        structure.execute(simple_payload)

        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=0)
        mueller.add_optical_component("anisotropic_sample")

        reflectivity = mueller.get_reflectivity()

        # Reflectivity should be between 0 and 1
        assert np.all(reflectivity >= 0)
        assert np.all(reflectivity <= 1)
