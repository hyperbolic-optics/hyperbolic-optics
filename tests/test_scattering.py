"""Tests for the numerically-stable scattering-matrix backend.

The correctness anchor is that the scattering backend reproduces the (golden-
locked) transfer-matrix reflection coefficients wherever the transfer method is
well-conditioned; the payoff is that it stays finite and energy-conserving in the
thick-evanescent regime where the transfer product overflows to NaN.
"""

import numpy as np
import pytest

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure
from tests.golden.payloads import PAYLOADS, SLOW_PAYLOADS


def _run(payload, backend):
    structure = Structure()
    structure.execute(payload, backend=backend)
    return structure


class TestBackendGuards:
    """The package must say which backend is at fault, and not lie about why."""

    def test_transfer_backend_manufactures_cross_polarization(self):
        """Documents the defect the cross-check above xfails on.

        With rotationZ = 0 the crystal axes stay in the plane of incidence, so
        there is no p-s conversion and r_sp is zero by symmetry. The scattering
        cascade returns ~1e-13; the transfer product returns ~9e-3, having lost
        the precision to a growing exponential in the evanescent gap.
        """
        payload = PAYLOADS["multilayer_incident"]
        transfer = np.abs(np.asarray(_run(payload, "transfer").r_sp))
        scattering = np.abs(np.asarray(_run(payload, "scattering").r_sp))

        assert scattering.max() < 1e-7, "symmetry-forbidden r_sp should vanish"
        assert transfer.max() > 1e-3, (
            "if the transfer product no longer manufactures r_sp here, its "
            "conditioning has been fixed -- drop the xfail above"
        )

    def test_ill_conditioned_transfer_result_warns(self):
        """A NaN from the transfer product must name the backend that fixes it."""
        payload = {
            "ScenarioData": {
                "type": "Simple",
                "incidentAngle": 45.0,
                "azimuthal_angle": 0.0,
                "frequency": 1460.0,
            },
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {"type": "Isotropic Middle-Stack Layer", "thickness": 200.0, "permittivity": 1.0},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationY": 90,
                },
            ],
        }
        with pytest.warns(UserWarning, match="backend='scattering'"):
            with np.errstate(all="ignore"):
                Structure().execute(payload)

        # ... and the scattering backend handles the same stack.
        stable = _run(payload, "scattering")
        assert np.isfinite(np.asarray(stable.r_pp)).all()

    def test_field_profile_rejects_the_scattering_backend_honestly(self):
        """Not 'has not been executed' -- it has; the backend just cannot supply fields."""
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
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationY": 90,
                },
            ],
        }
        structure = _run(payload, "scattering")
        with pytest.raises(NotImplementedError, match="requires backend='transfer'"):
            FieldProfile(structure)


class TestCrossCheckVsTransfer:
    """Scattering r-coefficients match the transfer backend across the battery."""

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(
                n,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "Known transfer-backend precision loss. Its r_sp reaches "
                        "9e-3 at points where symmetry (rotationZ=0, so the "
                        "crystal axes stay in the plane of incidence) forbids any "
                        "p-s conversion; the scattering backend gives 1e-13 there. "
                        "The same payload reflects 1.000076 of the incident power. "
                        "Fixing it means reformulating the transfer product, not "
                        "adjusting this test -- see test_transfer_backend_"
                        "manufactures_cross_polarization."
                    ),
                )
                if n == "multilayer_incident"
                else [],
            )
            for n in PAYLOADS
            if n not in SLOW_PAYLOADS
        ],
    )
    def test_reflection_matches(self, name):
        transfer = _run(PAYLOADS[name], "transfer")
        scattering = _run(PAYLOADS[name], "scattering")

        # Compare wherever the transfer product returned a number. It is excused
        # only where it overflowed to NaN -- that is the regime the scattering
        # backend exists for, and TestStability covers it.
        #
        # This used to be gated on np.linalg.cond(transfer_matrix) < 1e10, which
        # excused everything: a semi-infinite exit contributes two identically
        # zero columns, so the assembled matrix is structurally rank-deficient
        # and cond is inf on every payload with one. The mask was False
        # everywhere and the assertion compared no elements at all on 10 of the
        # 11 payloads.
        for key in ("r_pp", "r_ss", "r_ps", "r_sp"):
            a = np.asarray(getattr(transfer, key))
            b = np.asarray(getattr(scattering, key))
            comparable = np.isfinite(a) & np.isfinite(b)

            assert comparable.any(), f"{name}/{key}: no finite element to compare"
            np.testing.assert_allclose(
                b[comparable],
                a[comparable],
                atol=1e-7,
                rtol=1e-5,
                err_msg=f"{name}: '{key}' differs between backends",
            )

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
