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

    def test_lost_minors_are_repaired_not_reported(self):
        """The raw transfer product invents cross-polarization; the default must not.

        With rotationZ = 0 the crystal axes stay in the plane of incidence, so
        there is no p-s conversion and r_sp is zero by symmetry. At the handful
        of points where a layer's propagation term drives the assembled matrix to
        rank one, the r_sp minor cancels to the rounding floor and the raw
        product returns ~9e-3 of it. Those points are recomputed via the cascade.
        """
        payload = PAYLOADS["multilayer_incident"]

        stabilized = Structure()
        stabilized.execute(payload)

        raw = Structure()
        raw.execute(payload)
        raw.calculate_reflectivity(stabilize=False)

        assert np.abs(np.asarray(raw.r_sp)).max() > 1e-3, (
            "the raw transfer product no longer loses these minors; if its "
            "conditioning has genuinely improved, this test can go"
        )
        assert np.abs(np.asarray(stabilized.r_sp)).max() < 1e-7, (
            "symmetry-forbidden r_sp survived the repair"
        )
        assert 0.0 < stabilized.repaired_fraction < 0.05, (
            f"expected a sparse repair, got {stabilized.repaired_fraction:.4f}"
        )

    def test_repair_leaves_well_conditioned_points_untouched(self):
        """A payload with no lost minors must be bit-identical to the raw product."""
        payload = PAYLOADS["simple_calcite"]

        stabilized = Structure()
        stabilized.execute(payload)
        assert stabilized.repaired_fraction == 0.0

        raw = Structure()
        raw.execute(payload)
        raw.calculate_reflectivity(stabilize=False)
        for key in ("r_pp", "r_ss", "r_ps", "r_sp"):
            assert np.asarray(getattr(stabilized, key)) == np.asarray(getattr(raw, key))

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

    def test_field_profile_works_under_either_backend(self):
        """Power quantities must agree wherever the transfer product is sound."""
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
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationY": 90,
                },
            ],
        }
        transfer = FieldProfile(_run(payload, "transfer"))
        cascade = FieldProfile(_run(payload, "scattering"))

        for polarization in ("p", "s"):
            for quantity in ("reflectance", "transmittance"):
                np.testing.assert_allclose(
                    np.asarray(getattr(cascade, quantity)(polarization)),
                    np.asarray(getattr(transfer, quantity)(polarization)),
                    rtol=1e-9,
                    atol=1e-12,
                    err_msg=f"{quantity}('{polarization}') differs between backends",
                )
            for one, two in zip(
                cascade.layer_absorption(polarization),
                transfer.layer_absorption(polarization),
                strict=True,
            ):
                np.testing.assert_allclose(
                    np.asarray(one["absorptance"]),
                    np.asarray(two["absorptance"]),
                    rtol=1e-9,
                    atol=1e-12,
                )

    def test_transmission_coefficients_agree_for_an_isotropic_exit(self):
        """Both are in the clean s/p basis only there; a crystal exit uses its own."""
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
        transfer = FieldProfile(_run(payload, "transfer")).transmission_coefficients()
        cascade = FieldProfile(_run(payload, "scattering")).transmission_coefficients()
        for key in ("t_pp", "t_ss", "t_ps", "t_sp"):
            assert complex(np.asarray(cascade[key])) == pytest.approx(
                complex(np.asarray(transfer[key])), abs=1e-9
            ), key

    def test_power_quantities_survive_where_the_transfer_product_does_not(self):
        """The whole point: a gap thick enough to take the transfer route to NaN."""
        payload = _otto_gap(150.0)
        with np.errstate(all="ignore"):
            broken = Structure()
            broken.execute(payload)
        assert not np.isfinite(np.asarray(broken.r_pp)).all()

        cascade = FieldProfile(_run(payload, "scattering"))
        reflectance = np.asarray(cascade.reflectance("p"))
        assert np.isfinite(reflectance).all()
        # A thick evanescent gap reflects everything.
        np.testing.assert_allclose(reflectance, 1.0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(cascade.transmittance("p")), 0.0, atol=1e-9)

    def test_depth_profile_still_says_it_needs_the_transfer_backend(self):
        """Power works under the cascade; resolving with depth does not yet."""
        structure = _run(_otto_gap(1.0), "scattering")
        with pytest.raises(NotImplementedError, match="field_profile requires"):
            FieldProfile(structure).field_profile("p")


class TestCrossCheckVsTransfer:
    """Scattering r-coefficients match the transfer backend across the battery."""

    @pytest.mark.parametrize("name", [n for n in PAYLOADS if n not in SLOW_PAYLOADS])
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
