"""Tests for numerical transmission, layer-resolved absorption, and field profiles.

The load-bearing checks are *independent* of the field machinery:

- ``reflectance`` reconstructed from the propagated fields must equal ``|r|²``
  computed by the (analytic) reflection path in ``structure.py``;
- for a single lossless dielectric interface, ``R``/``T`` must equal the analytic
  Fresnel result;
- per-layer absorption must be non-negative (passivity) and, for a bulk
  semi-infinite absorber, the flux must decay monotonically into the crystal.

Energy conservation ``R + T + ΣA = 1`` holds by construction (the flux differences
telescope), so it is asserted as a cheap structural guard rather than a physics proof.
"""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure


def _simple(eps_prism, extra_layers, freq=1460.0, angle=30.0):
    """Build + execute a Simple-scenario structure (prism + given extra layers)."""
    structure = Structure()
    structure.execute(
        {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": angle,
                "azimuthal_angle": 0.0,
                "frequency": freq,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": eps_prism},
                *extra_layers,
            ],
        }
    )
    return structure


def _fresnel_R(eps1, eps2, theta_rad):
    """Analytic single-interface Fresnel power reflectance (R_s, R_p)."""
    n1, n2 = np.sqrt(complex(eps1)), np.sqrt(complex(eps2))
    cos1 = np.cos(theta_rad)
    sin2 = (n1 / n2) * np.sin(theta_rad)
    cos2 = np.sqrt(1 - sin2**2)
    r_s = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    r_p = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    return abs(r_s) ** 2, abs(r_p) ** 2


_CALCITE = {"type": "Semi Infinite Anisotropic Layer", "material": "Calcite", "rotationY": 90}


class TestReflectanceMatchesCoefficients:
    """The field-reconstructed R must equal |r|^2 from the analytic path."""

    @pytest.mark.parametrize(
        "eps_prism,extra",
        [
            (50.0, [_CALCITE]),  # bulk anisotropic
            (
                2.0,
                [
                    {
                        "type": "Isotropic Middle-Stack Layer",
                        "thickness": 0.5,
                        "permittivity": 1.0,
                    },
                    {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
                ],
            ),
            (
                50.0,
                [
                    {
                        "type": "Crystal Layer",
                        "material": "Calcite",
                        "thickness": 1.0,
                        "rotationY": 90,
                    },
                    _CALCITE,
                ],
            ),
        ],
    )
    def test_reflectance_p_and_s(self, eps_prism, extra):
        structure = _simple(eps_prism, extra)
        fp = FieldProfile(structure)
        r_p = abs(structure.r_pp) ** 2 + abs(structure.r_sp) ** 2
        r_s = abs(structure.r_ss) ** 2 + abs(structure.r_ps) ** 2
        assert np.allclose(fp.reflectance("p"), r_p, atol=1e-9)
        assert np.allclose(fp.reflectance("s"), r_s, atol=1e-9)


class TestFresnel:
    """Single lossless dielectric interface reproduces analytic Fresnel R and T."""

    def test_single_interface(self):
        eps1, eps2, angle_deg = 4.0, 1.0, 30.0
        structure = _simple(
            eps1, [{"type": "Semi Infinite Isotropic Layer", "permittivity": eps2}], angle=angle_deg
        )
        fp = FieldProfile(structure)
        rs, rp = _fresnel_R(eps1, eps2, np.radians(angle_deg))
        assert fp.reflectance("p") == pytest.approx(rp, abs=1e-6)
        assert fp.reflectance("s") == pytest.approx(rs, abs=1e-6)
        # lossless: T = 1 - R
        assert fp.transmittance("p") == pytest.approx(1.0 - rp, abs=1e-6)
        assert fp.transmittance("s") == pytest.approx(1.0 - rs, abs=1e-6)


class TestEnergyConservation:
    """R + T + sum(A) == 1 (structural) across material/scenario paths."""

    @pytest.mark.parametrize("pol", ["p", "s"])
    @pytest.mark.parametrize(
        "eps_prism,extra",
        [
            (50.0, [_CALCITE]),
            (
                50.0,
                [
                    {
                        "type": "Crystal Layer",
                        "material": "Calcite",
                        "thickness": 1.0,
                        "rotationY": 90,
                    },
                    _CALCITE,
                ],
            ),
            (
                12.5,
                [
                    {
                        "type": "Isotropic Middle-Stack Layer",
                        "thickness": 0.5,
                        "permittivity": 2.0,
                        "permeability": {"real": 1.5, "imag": 0.1},
                    },
                    _CALCITE,
                ],
            ),
            (
                25.0,
                [
                    {
                        "type": "Semi Infinite Anisotropic Layer",
                        "material": "GalliumOxide",
                        "rotationY": 90,
                    }
                ],
            ),
        ],
    )
    def test_conservation(self, pol, eps_prism, extra):
        fp = FieldProfile(_simple(eps_prism, extra))
        assert fp.check_conservation(pol) < 1e-9


class TestPassivity:
    """Absorption in passive layers is non-negative; total absorption in [0, 1]."""

    def test_layer_absorption_nonnegative(self):
        structure = _simple(
            50.0,
            [
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.5, "rotationY": 90},
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.5, "rotationY": 20},
                _CALCITE,
            ],
        )
        fp = FieldProfile(structure)
        summ = fp.summary("p")
        for layer in summ["layers"]:
            assert layer["absorptance"] >= -1e-12
        assert 0.0 - 1e-9 <= float(summ["total_absorption"]) <= 1.0 + 1e-9


class TestBulkSemiInfinite:
    """The flagged case: a single semi-infinite anisotropic exit."""

    def test_total_into_bulk_is_one_minus_R(self):
        fp = FieldProfile(_simple(50.0, [_CALCITE]))
        for pol in ("p", "s"):
            R = float(fp.reflectance(pol))
            T = float(fp.transmittance(pol))
            assert T == pytest.approx(1.0 - R, abs=1e-9)
            assert not fp.layer_absorption(pol)  # no interior layers to resolve

    def test_flux_decays_into_crystal(self):
        fp = FieldProfile(_simple(50.0, [_CALCITE]))
        prof = fp.field_profile("p", n_points=100)
        sz = prof["Sz"]
        # monotonic non-increasing into a passive bulk (allow tiny numerical noise)
        assert np.all(np.diff(sz) <= 1e-9)
        assert sz[0] == pytest.approx(1.0 - float(fp.reflectance("p")), abs=1e-6)
        assert sz[-1] < sz[0]

    def test_explicit_window_controls_depth(self):
        fp = FieldProfile(_simple(50.0, [_CALCITE]))
        prof = fp.field_profile("p", n_points=50, semi_inf_thickness=2.0)  # microns
        assert prof["z"][-1] == pytest.approx(2.0, abs=1e-6)


class TestFieldContinuity:
    """Tangential E and H are continuous across interior interfaces."""

    def test_tangential_continuity(self):
        structure = _simple(
            50.0,
            [
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.0, "rotationY": 90},
                _CALCITE,
            ],
        )
        fp = FieldProfile(structure)
        n_points = 80
        prof = fp.field_profile("p", n_points=n_points)
        # seam between the (single) interior layer and the exit window:
        seam = n_points  # first index of the exit segment
        for comp in ("Ex", "Ey", "Hx", "Hy"):  # tangential components only
            below = prof[comp][seam - 1]
            above = prof[comp][seam]
            assert np.allclose(below, above, atol=1e-6, rtol=1e-4)


class TestIsotropicExitProfile:
    """Field profile through a lossless isotropic exit (distinct reconstruction path)."""

    def test_lossless_profile_conserves_flux(self):
        # prism eps=2, air gap, exit eps=2: kx=0.707 so the gap and the exit both
        # propagate (real kz) and are lossless -> the auto window deterministically
        # uses the wavelength fallback, and flux is constant through the stack.
        structure = _simple(
            2.0,
            [
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5, "permittivity": 1.0},
                {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
            ],
        )
        fp = FieldProfile(structure)
        # isotropic exit modes are reconstructed from the dynamical-matrix columns.
        prof = fp.field_profile("p", n_points=60)
        sz = prof["Sz"]
        # lossless everywhere -> flux constant and equal to T
        assert np.allclose(sz, float(fp.transmittance("p")), atol=1e-6)
        assert prof["z"][-1] > prof["z"][0]


class TestPolarization:
    """Jones-pair polarization input matches the 'p'/'s' shortcuts."""

    def test_jones_matches_pure(self):
        fp = FieldProfile(_simple(50.0, [_CALCITE]))
        assert fp.transmittance((0.0, 1.0)) == pytest.approx(
            float(fp.transmittance("p")), abs=1e-12
        )
        assert fp.transmittance((1.0, 0.0)) == pytest.approx(
            float(fp.transmittance("s")), abs=1e-12
        )


class TestShapes:
    """Power quantities follow the same presentation shapes as r_pp."""

    def test_incident_shape(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Incident"},
                "Layers": [{"type": "Ambient Incident Layer", "permittivity": 50.0}, _CALCITE],
            }
        )
        fp = FieldProfile(structure)
        assert fp.transmittance("p").shape == structure.r_pp.shape
        assert fp.reflectance("p").shape == structure.r_pp.shape
        assert fp.check_conservation("p") < 1e-9

    def test_dispersion_shape(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Dispersion", "frequency": 1460.0},
                "Layers": [{"type": "Ambient Incident Layer", "permittivity": 25.0}, _CALCITE],
            }
        )
        fp = FieldProfile(structure)
        assert fp.transmittance("p").shape == structure.r_pp.shape


class TestTransmissionCoefficients:
    """Amplitude transmission coefficients match the analytic path and power T."""

    def test_matches_structure_transmissivity(self):
        structure = _simple(
            50.0,
            [
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.0, "rotationY": 90},
                _CALCITE,
            ],
        )
        structure.calculate_transmissivity()  # analytic amplitude coefficients
        t = FieldProfile(structure).transmission_coefficients()
        assert np.allclose(t["t_pp"], structure.t_pp, atol=1e-9)
        assert np.allclose(t["t_ss"], structure.t_ss, atol=1e-9)
        assert np.allclose(t["t_ps"], structure.t_ps, atol=1e-9)
        assert np.allclose(t["t_sp"], structure.t_sp, atol=1e-9)

    def test_isotropic_amplitude_to_power(self):
        # Single lossless interface: power T_p relates to |t_pp|^2 by the
        # Poynting/impedance factor Re(N2 cos2)/Re(N1 cos1). Verifies the amplitude
        # coefficient and the power transmittance are mutually consistent.
        eps1, eps2, ang = 4.0, 9.0, 25.0
        structure = _simple(
            eps1, [{"type": "Semi Infinite Isotropic Layer", "permittivity": eps2}], angle=ang
        )
        fp = FieldProfile(structure)
        n1, n2 = np.sqrt(eps1), np.sqrt(eps2)
        cos1 = np.cos(np.radians(ang))
        cos2 = np.sqrt(1 - ((n1 / n2) * np.sin(np.radians(ang))) ** 2)
        t_pp = complex(fp.transmission_coefficients()["t_pp"])
        power = (n2 * cos2) / (n1 * cos1) * abs(t_pp) ** 2
        assert power == pytest.approx(float(fp.transmittance("p")), abs=1e-6)


class TestLayerResolvedHeterostructure:
    """MoO3/AlN/SiC Otto-geometry layer decomposition (the example's physics)."""

    def test_azimuthal_energy_decomposition(self):
        structure = Structure()
        structure.execute(
            {
                "ScenarioData": {"type": "Azimuthal", "incidentAngle": 28.0},
                "Layers": [
                    {"type": "Ambient Incident Layer", "permittivity": 5.76},
                    {"type": "Isotropic Middle-Stack Layer", "thickness": 8.0, "permittivity": 1.0},
                    {"type": "Crystal Layer", "material": "MoO3", "thickness": 1.2, "rotationY": 0},
                    {"type": "Crystal Layer", "material": "AlN", "thickness": 1.0, "rotationY": 0},
                    {"type": "Semi Infinite Anisotropic Layer", "material": "SiC", "rotationY": 0},
                ],
            }
        )
        fp = FieldProfile(structure)
        r = fp.reflectance("p")
        a_sic = fp.transmittance("p")  # absorbed in the lossy semi-infinite substrate
        per = {e["index"]: e["absorptance"] for e in fp.layer_absorption("p")}
        a_total = r + per[2] + per[3] + a_sic  # MoO3=2, AlN=3
        assert r.shape == (410, 360)
        assert np.allclose(a_total, 1.0, atol=1e-9)
        # interior absorptions are physical (non-negative)
        assert np.all(per[2] >= -1e-9) and np.all(per[3] >= -1e-9)


class TestGuards:
    """Constructor and input validation."""

    def test_requires_executed_structure(self):
        with pytest.raises(ValueError):
            FieldProfile(Structure())

    def test_unknown_polarization(self):
        fp = FieldProfile(_simple(50.0, [_CALCITE]))
        with pytest.raises(ValueError):
            fp.transmittance("circular")
