"""Tests for the numerically-stable scattering-matrix backend.

The correctness anchor is that the scattering backend reproduces the (golden-
locked) transfer-matrix reflection coefficients wherever the transfer method is
well-conditioned; the payoff is that it stays finite and energy-conserving in the
thick-evanescent regime where the transfer product overflows to NaN.
"""

import numpy as np
import pytest

from hyperbolic_optics.axes import present
from hyperbolic_optics.structure import Structure
from tests.golden.payloads import PAYLOADS, SLOW_PAYLOADS


def _run(payload, backend):
    structure = Structure()
    structure.execute(payload, backend=backend)
    return structure


class TestCrossCheckVsTransfer:
    """Scattering r-coefficients match the transfer backend across the battery."""

    @pytest.mark.parametrize("name", [n for n in PAYLOADS if n not in SLOW_PAYLOADS])
    def test_reflection_matches(self, name):
        transfer = _run(PAYLOADS[name], "transfer")
        scattering = _run(PAYLOADS[name], "scattering")
        # Only require agreement where the transfer matrix is well-conditioned:
        # at near-grazing / deep-evanescent points Gamma is (near-)singular and the
        # transfer coefficients are unreliable -- that is precisely where the
        # scattering method is the correct one (covered by TestStability).
        cond = present(np.linalg.cond(np.asarray(transfer.transfer_matrix)))
        reliable = np.asarray(cond) < 1e10
        for key in ("r_pp", "r_ss", "r_ps", "r_sp"):
            a = np.asarray(getattr(transfer, key))
            b = np.asarray(getattr(scattering, key))
            ok = np.isclose(a, b, atol=1e-7, rtol=1e-5) | ~reliable | ~np.isfinite(a)
            assert np.all(ok), f"{key}: {np.count_nonzero(~ok)} mismatches"

    def test_transmission_matches_for_isotropic_exit(self):
        # Transmission is in the clean s/p basis only for an isotropic exit, so
        # compare t there (crystal-exit t is in the eigenmode basis for both).
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 30.0,
                "azimuthal_angle": 0.0,
                "frequency": 1460.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 25.0},
                {"type": "Crystal Layer", "material": "Calcite", "thickness": 1.0, "rotationY": 90},
                {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
            ],
        }
        transfer = _run(payload, "transfer")
        transfer.calculate_transmissivity()
        scattering = _run(payload, "scattering")
        for key in ("t_pp", "t_ss", "t_ps", "t_sp"):
            assert complex(getattr(scattering, key)) == pytest.approx(
                complex(getattr(transfer, key)), abs=1e-7
            ), key


def _otto_gap(gap_um):
    return {
        "ScenarioData": {
            "type": "Simple",
            "incidentAngle": 45.0,
            "azimuthal_angle": 0.0,
            "frequency": 1460.0,
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {"type": "Isotropic Middle-Stack Layer", "thickness": gap_um, "permittivity": 1.0},
            {"type": "Semi Infinite Anisotropic Layer", "material": "Calcite", "rotationY": 90},
        ],
    }


class TestStability:
    """The whole point: stable where the transfer matrix overflows."""

    def test_thick_evanescent_gap(self):
        # kx = sqrt(50)*sin45 ~ 5 >> 1, so a thick air gap is deeply evanescent.
        transfer = _run(_otto_gap(250.0), "transfer")
        scattering = _run(_otto_gap(250.0), "scattering")
        # transfer product overflows -> non-finite
        assert not np.isfinite(complex(transfer.r_pp))
        # scattering stays finite and physical: frustrated TIR vanishes -> R -> 1
        r = complex(scattering.r_pp)
        assert np.isfinite(r)
        reflectance = abs(r) ** 2 + abs(complex(scattering.r_sp)) ** 2
        assert reflectance == pytest.approx(1.0, abs=1e-6)

    def test_thin_gap_agrees_with_transfer(self):
        # where the transfer method works, the two backends agree.
        transfer = _run(_otto_gap(1.0), "transfer")
        scattering = _run(_otto_gap(1.0), "scattering")
        assert complex(scattering.r_pp) == pytest.approx(complex(transfer.r_pp), abs=1e-8)


class TestEnergyConservation:
    """Lossless symmetric (prism == substrate) stack: R + T == 1."""

    def test_symmetric_lossless(self):
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 20.0,
                "azimuthal_angle": 0.0,
                "frequency": 1460.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 2.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.8, "permittivity": 1.0},
                {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
            ],
        }
        s = _run(payload, "scattering")
        # symmetric prism == substrate -> the |t|^2 impedance factor is 1
        r_p = abs(complex(s.r_pp)) ** 2 + abs(complex(s.r_sp)) ** 2
        t_p = abs(complex(s.t_pp)) ** 2 + abs(complex(s.t_sp)) ** 2
        assert r_p + t_p == pytest.approx(1.0, abs=1e-9)


class TestScenarios:
    """Scattering matches transfer across scenarios and shapes."""

    @pytest.mark.parametrize(
        "scenario",
        [
            {"type": "Incident"},
            {"type": "Azimuthal", "incidentAngle": 40},
            {"type": "Dispersion", "frequency": 1460.0},
        ],
    )
    def test_matches_across_scenarios(self, scenario):
        payload = {
            "ScenarioData": scenario,
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 12.5},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 0.5},
                {"type": "Semi Infinite Anisotropic Layer", "material": "Calcite", "rotationY": 90},
            ],
        }
        transfer = _run(payload, "transfer")
        scattering = _run(payload, "scattering")
        assert scattering.r_pp.shape == transfer.r_pp.shape
        assert np.allclose(scattering.r_pp, transfer.r_pp, atol=1e-7, rtol=1e-5)


class TestGuards:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            _run(_otto_gap(1.0), "bogus")

    def test_transfer_is_default(self):
        # execute() without a backend arg behaves as the transfer backend.
        default = Structure()
        default.execute(_otto_gap(1.0))
        explicit = _run(_otto_gap(1.0), "transfer")
        assert complex(default.r_pp) == pytest.approx(complex(explicit.r_pp), abs=0)
