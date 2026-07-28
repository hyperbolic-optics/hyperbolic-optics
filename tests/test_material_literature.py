"""
Shipped materials against the numbers in the papers they cite.

Nothing else in the suite compares a material to a literature value, so a
mis-transcribed phonon frequency or a band attached to the wrong crystal axis
produces a plausible-looking spectrum and no test notices. These check the
observable that matters -- where Re(eps) goes negative, i.e. where the material
is Reststrahlen/hyperbolic -- against the published band edges.
"""

import numpy as np
import pytest

from hyperbolic_optics.materials import HexagonalBoronNitride, MolybdenumTrioxide

# Alvarez-Perez et al., Adv. Mater. 32, 1908176 (2020), Table 1 (experiment).
# Axis convention is the paper's own (Fig. 1a): x=[100], y=[001], z=[010],
# with z=[010] the van der Waals stacking axis. Band assignment per Fig. 5.
MOO3_EPS_INF = {"x": 5.78, "y": 6.07, "z": 4.47}
MOO3_BANDS = {
    "x": (821.4, 963.0),  # RB2, [100]
    "y": (544.6, 850.1),  # RB1, [001]
    "z": (956.7, 1006.9),  # RB3, [010]
}

# Caldwell et al., Nat. Commun. 5, 5221 (2014), Supplementary Table.
HBN_EPS_INF = {"ordinary": 4.90, "extraordinary": 2.95}
HBN_BANDS = {
    "ordinary": (1360.0, 1614.0),  # in-plane E1u
    "extraordinary": (760.0, 825.0),  # out-of-plane A2u
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "ordinary": 0, "extraordinary": 2}

MARGIN = 5.0  # cm^-1 either side of a band edge


def real_eps(material, axis, frequency):
    tensor = np.asarray(material.fetch_permittivity_tensor_for_freq(frequency))
    index = AXIS_INDEX[axis]
    return float(np.real(tensor[..., index, index]).item())


class TestMolybdenumTrioxideBands:
    """alpha-MoO3's three Reststrahlen bands sit on the axes the paper assigns them to."""

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_reststrahlen_band_matches_literature(self, axis):
        material = MolybdenumTrioxide()
        omega_to, omega_lo = MOO3_BANDS[axis]

        assert real_eps(material, axis, omega_to + MARGIN) < 0, (
            f"eps_{axis} should be negative just above omega_TO = {omega_to}"
        )
        assert real_eps(material, axis, omega_lo - MARGIN) < 0, (
            f"eps_{axis} should be negative just below omega_LO = {omega_lo}"
        )
        assert real_eps(material, axis, omega_to - MARGIN) > 0, (
            f"eps_{axis} should be positive below the band"
        )
        assert real_eps(material, axis, omega_lo + MARGIN) > 0, (
            f"eps_{axis} should be positive above the band"
        )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_high_frequency_limit(self, axis):
        """Far above every phonon the factorized form tends to eps_inf."""
        material = MolybdenumTrioxide()
        assert real_eps(material, axis, 1.0e6) == pytest.approx(MOO3_EPS_INF[axis], rel=1e-6)

    def test_bands_are_on_distinct_axes(self):
        """Each band belongs to one axis only -- catches a band/axis swap.

        The y=[001] and z=[010] assignments were previously exchanged, which is
        invisible in an unrotated spectrum plot but rotates the whole in-plane
        hyperbolic response by 90 degrees.
        """
        material = MolybdenumTrioxide()
        for axis, (omega_to, omega_lo) in MOO3_BANDS.items():
            midband = 0.5 * (omega_to + omega_lo)
            negative = [a for a in ("x", "y", "z") if real_eps(material, a, midband) < 0]
            assert axis in negative, f"{axis} is not negative in the middle of its own band"


class TestHexagonalBoronNitrideBands:
    """hBN's two Reststrahlen bands, in-plane and out-of-plane."""

    @pytest.mark.parametrize("axis", ["ordinary", "extraordinary"])
    def test_reststrahlen_band_matches_literature(self, axis):
        material = HexagonalBoronNitride()
        omega_to, omega_lo = HBN_BANDS[axis]

        assert real_eps(material, axis, omega_to + MARGIN) < 0
        assert real_eps(material, axis, omega_lo - MARGIN) < 0
        assert real_eps(material, axis, omega_to - MARGIN) > 0
        assert real_eps(material, axis, omega_lo + MARGIN) > 0

    @pytest.mark.parametrize("axis", ["ordinary", "extraordinary"])
    def test_high_frequency_limit(self, axis):
        material = HexagonalBoronNitride()
        assert real_eps(material, axis, 1.0e6) == pytest.approx(HBN_EPS_INF[axis], rel=1e-6)

    def test_upper_band_is_in_plane(self):
        """hBN is type-II hyperbolic in the upper band: in-plane negative, out-of-plane positive."""
        material = HexagonalBoronNitride()
        midband = 0.5 * sum(HBN_BANDS["ordinary"])
        assert real_eps(material, "ordinary", midband) < 0
        assert real_eps(material, "extraordinary", midband) > 0

    def test_lower_band_is_out_of_plane(self):
        """And type-I in the lower band, the other way round."""
        material = HexagonalBoronNitride()
        midband = 0.5 * sum(HBN_BANDS["extraordinary"])
        assert real_eps(material, "extraordinary", midband) < 0
        assert real_eps(material, "ordinary", midband) > 0
