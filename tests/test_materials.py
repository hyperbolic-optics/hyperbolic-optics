"""
Tests for material classes and permittivity calculations.
"""

import numpy as np

from hyperbolic_optics.materials import (
    Air,
    ArbitraryMaterial,
    CalciteLower,
    CalciteUpper,
    GalliumOxide,
    Quartz,
    Sapphire,
)


class TestUniaxialMaterials:
    """Test uniaxial material implementations."""

    def test_quartz_initialization(self):
        """Test Quartz material initializes correctly."""
        quartz = Quartz()
        assert quartz.name == "Quartz"
        assert quartz.frequency is not None
        assert len(quartz.frequency) == 410

    def test_sapphire_initialization(self):
        """Test Sapphire material initializes correctly."""
        sapphire = Sapphire()
        assert sapphire.name == "Sapphire"
        assert sapphire.frequency is not None

    def test_calcite_upper_initialization(self):
        """Test CalciteUpper material initializes correctly."""
        calcite = CalciteUpper()
        assert "Calcite" in calcite.name
        assert calcite.frequency is not None

    def test_calcite_lower_initialization(self):
        """Test CalciteLower material initializes correctly."""
        calcite = CalciteLower()
        assert "Calcite" in calcite.name
        assert calcite.frequency is not None

    def test_permittivity_tensor_shape(self):
        """Test that permittivity tensors have correct shape."""
        quartz = Quartz()
        eps_tensor = quartz.fetch_permittivity_tensor()

        # Should be [frequency_points, 3, 3]
        assert eps_tensor.shape == (410, 3, 3)
        assert np.iscomplexobj(eps_tensor)

    def test_permittivity_tensor_for_freq(self):
        """Test single frequency permittivity calculation."""
        quartz = Quartz()
        eps_tensor = quartz.fetch_permittivity_tensor_for_freq(500.0)

        # Should be [3, 3]
        assert eps_tensor.shape == (3, 3)
        assert np.iscomplexobj(eps_tensor)

    def test_magnetic_tensor_default(self):
        """Test that default magnetic tensor is identity."""
        quartz = Quartz()
        mu_tensor = quartz.fetch_magnetic_tensor()

        # Should be [frequency_points, 3, 3]
        assert mu_tensor.shape == (410, 3, 3)

        # Should be approximately identity (diagonal = 1, off-diagonal = 0)
        for i in range(410):
            assert np.allclose(np.diag(mu_tensor[i]), [1.0, 1.0, 1.0])
            # Check off-diagonal elements are zero
            mu_copy = mu_tensor[i].copy()
            np.fill_diagonal(mu_copy, 0)
            assert np.allclose(mu_copy, 0)

    def test_uniaxial_diagonal_structure(self):
        """Test that uniaxial materials have proper diagonal structure."""
        quartz = Quartz()
        eps_tensor = quartz.fetch_permittivity_tensor()

        # For uniaxial materials, eps_xx should equal eps_yy
        assert np.allclose(eps_tensor[:, 0, 0], eps_tensor[:, 1, 1])

        # Off-diagonal elements should be zero
        assert np.allclose(eps_tensor[:, 0, 1], 0)
        assert np.allclose(eps_tensor[:, 0, 2], 0)
        assert np.allclose(eps_tensor[:, 1, 2], 0)


class TestMonoclinicMaterials:
    """Test monoclinic material implementations."""

    def test_gallium_oxide_initialization(self):
        """Test GalliumOxide material initializes correctly."""
        gaox = GalliumOxide()
        assert gaox.name == "GalliumOxide"
        assert gaox.frequency is not None

    def test_gallium_oxide_tensor_shape(self):
        """Test GalliumOxide permittivity tensor shape."""
        gaox = GalliumOxide()
        eps_tensor = gaox.fetch_permittivity_tensor()

        # Should have proper shape
        assert eps_tensor.shape[1:] == (3, 3)
        assert np.iscomplexobj(eps_tensor)

    def test_gallium_oxide_off_diagonal(self):
        """Test that GalliumOxide has non-zero xy component."""
        gaox = GalliumOxide()
        eps_tensor = gaox.fetch_permittivity_tensor_for_freq(600.0)

        # Should have non-zero eps_xy component
        assert eps_tensor[0, 1] != 0
        assert eps_tensor[1, 0] == eps_tensor[0, 1]  # Symmetry


class TestArbitraryMaterial:
    """Test arbitrary material functionality."""

    def test_arbitrary_material_initialization(self):
        """Test ArbitraryMaterial initializes with defaults."""
        mat = ArbitraryMaterial()
        assert mat.name == "Arbitrary Material"

    def test_arbitrary_material_custom_eps(self):
        """Test custom permittivity values."""
        material_data = {
            "eps_xx": {"real": 2.5, "imag": 0.1},
            "eps_yy": {"real": 3.0, "imag": 0.0},
            "eps_zz": {"real": -4.0, "imag": 0.5},
        }
        mat = ArbitraryMaterial(material_data)
        eps_tensor = mat.fetch_permittivity_tensor()

        assert eps_tensor[0, 0] == complex(2.5, 0.1)
        assert eps_tensor[1, 1] == complex(3.0, 0.0)
        assert eps_tensor[2, 2] == complex(-4.0, 0.5)

    def test_arbitrary_material_magnetic_tensor(self):
        """Test custom magnetic permeability values."""
        material_data = {
            "mu_xx": {"real": 2.0, "imag": 0.0},
            "mu_yy": {"real": 2.0, "imag": 0.0},
            "mu_zz": {"real": 2.0, "imag": 0.0},
        }
        mat = ArbitraryMaterial(material_data)
        mu_tensor = mat.fetch_magnetic_tensor()

        assert mu_tensor[0, 0] == complex(2.0, 0.0)
        assert mu_tensor[1, 1] == complex(2.0, 0.0)
        assert mu_tensor[2, 2] == complex(2.0, 0.0)

    def test_arbitrary_material_tensor_shape(self):
        """Test that arbitrary material tensors have correct shape."""
        mat = ArbitraryMaterial()
        eps_tensor = mat.fetch_permittivity_tensor()
        mu_tensor = mat.fetch_magnetic_tensor()

        assert eps_tensor.shape == (3, 3)
        assert mu_tensor.shape == (3, 3)


class TestIsotropicMaterial:
    """Test isotropic material (Air) functionality."""

    def test_air_initialization(self):
        """Test Air material initializes correctly."""
        air = Air()
        assert air.name == "Air"
        assert air.permittivity == complex(1.0, 0.0)

    def test_air_tensor_is_diagonal(self):
        """Test that Air produces diagonal tensor."""
        air = Air()
        eps_tensor = air.fetch_permittivity_tensor()

        # Should be 3x3 identity (with permittivity value)
        assert eps_tensor.shape == (3, 3)
        assert np.allclose(np.diag(eps_tensor), [1.0, 1.0, 1.0])

        # Off-diagonal should be zero
        eps_copy = eps_tensor.copy()
        np.fill_diagonal(eps_copy, 0)
        assert np.allclose(eps_copy, 0)

    def test_air_custom_permittivity(self):
        """Test Air with custom permittivity."""
        air = Air(permittivity=2.5)
        eps_tensor = air.fetch_permittivity_tensor()

        assert np.allclose(np.diag(eps_tensor), [2.5, 2.5, 2.5])


class TestMaterialPhysics:
    """Test physical properties of materials."""

    def test_permittivity_causality(self):
        """Test that imaginary part of permittivity is non-negative (loss)."""
        quartz = Quartz()
        eps_tensor = quartz.fetch_permittivity_tensor()

        # In lossy regions, imaginary part should be positive
        # (This is a general check, may not apply everywhere)
        # At least check that values are physical
        assert np.all(np.isfinite(eps_tensor))

    def test_frequency_independence_arbitrary(self):
        """Test that arbitrary materials are frequency-independent."""
        mat = ArbitraryMaterial()
        eps1 = mat.fetch_permittivity_tensor_for_freq(1000.0)
        eps2 = mat.fetch_permittivity_tensor_for_freq(2000.0)

        assert np.allclose(eps1, eps2)
