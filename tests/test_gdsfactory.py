"""Tests for the gdsfactory bridge (``hyperbolic_optics.integrations.gdsfactory``).

The contract: what the PCell draws is what extraction reads back, through a GDS
file; anything that is not a clean line grating is refused; and simulating a
layout equals simulating its geometry directly. Skipped without gdsfactory.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
gf = pytest.importorskip("gdsfactory")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import dispersion, rcwa  # noqa: E402
from hyperbolic_optics.integrations import gdsfactory as hog  # noqa: E402

LAYER = (1, 0)


@pytest.fixture(autouse=True, scope="module")
def pdk():
    gf.gpdk.PDK.activate()


def _rectangles(component, rectangles, layer=LAYER):
    for x0, x1, y0, y1 in rectangles:
        component.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)
    return component


class TestRoundTrip:
    def test_pcell_snaps_to_the_grid_and_extraction_recovers_it(self):
        component = hog.line_grating(period_um=2.1234567, fill=0.33333, n_periods=9)
        grating = hog.extract_line_grating(component, LAYER)
        assert component.info["period_um"] == pytest.approx(2.123, abs=1e-12)
        assert grating.period_um == pytest.approx(component.info["period_um"], abs=1e-9)
        assert grating.fill == pytest.approx(component.info["fill"], abs=1e-9)
        assert grating.n_periods == 9
        assert grating.offset == pytest.approx(0.0, abs=1e-9)

    def test_through_a_gds_file(self, tmp_path):
        component = hog.line_grating(period_um=3.2, fill=0.45, n_periods=12, length_um=40.0)
        path = tmp_path / "grating.gds"
        component.write_gds(path)
        from_file = hog.extract_line_grating(gf.import_gds(path), LAYER)
        assert from_file == hog.extract_line_grating(component, LAYER)
        assert from_file.length_um == pytest.approx(40.0)

    def test_offset_is_the_first_ridge_position(self):
        c = _rectangles(gf.Component(), [(0.5 + 2 * i, 1.3 + 2 * i, 0, 5) for i in range(4)])
        grating = hog.extract_line_grating(c, LAYER)
        assert grating.period_um == pytest.approx(2.0)
        assert grating.fill == pytest.approx(0.4)
        assert grating.offset == pytest.approx(0.25)


class TestRefusesWhatIsNotALineGrating:
    @pytest.mark.parametrize(
        "rectangles,message",
        [
            ([(0, 1, 0, 5)], "at least two ridges"),
            ([(0, 1, 0, 5), (2, 3.5, 0, 5), (4, 5, 0, 5)], "widths differ"),
            ([(0, 1, 0, 5), (2, 3, 0, 5), (4.5, 5.5, 0, 5)], "not equally spaced"),
            ([(0, 1, 0, 5), (2, 3, 0, 6)], "same y range"),
            ([(0, 2, 0, 5), (2, 4, 0, 5), (4, 6, 0, 5)], "touch or overlap"),
        ],
        ids=["one_ridge", "unequal_widths", "uneven_pitch", "uneven_length", "no_gaps"],
    )
    def test_rejections(self, rectangles, message):
        with pytest.raises(ValueError, match=message):
            hog.extract_line_grating(_rectangles(gf.Component(), rectangles), LAYER)

    def test_non_rectangular_polygon(self):
        c = _rectangles(gf.Component(), [(0, 1, 0, 5), (2, 3, 0, 5)])
        c.add_polygon([(4, 0), (5, 0), (5.5, 5), (4, 5)], layer=LAYER)
        with pytest.raises(ValueError, match="axis-aligned rectangle"):
            hog.extract_line_grating(c, LAYER)

    def test_empty_layer(self):
        with pytest.raises(ValueError, match="at least two ridges"):
            hog.extract_line_grating(hog.line_grating(), (2, 0))


class TestMaterialsAndStack:
    def test_material_specifications(self):
        wavelength = jnp.array([6.8, 7.0])
        named = hog.material_tensor("hBN", wavelength)
        np.testing.assert_allclose(named, dispersion.at_wavelength("hBN", wavelength))
        iso = hog.material_tensor(2.25, wavelength)
        assert iso.shape == (2, 3, 3)
        np.testing.assert_allclose(iso[0], 2.25 * np.eye(3))
        rotated = hog.material_tensor("hBN", 7.0, rotation_deg=(0.0, 90.0, 0.0))
        base = np.asarray(dispersion.at_wavelength("hBN", 7.0))
        assert complex(rotated[0, 0]) == pytest.approx(complex(base[2, 2]), abs=1e-12)
        drude = hog.material_tensor(lambda w: 1.0 - (w / 0.2) ** 2, 7.0)
        assert complex(drude[1, 1]) == pytest.approx(1.0 - 35.0**2)

    def test_simulating_a_layout_equals_simulating_its_geometry(self):
        component = hog.line_grating(period_um=2.0, fill=0.3, n_periods=6)
        stack = [
            hog.StackLayer(0.1, "hBN", background=1.0),
            hog.StackLayer(0.8, 1.85),
        ]
        wavelength = 1e4 / 1480.0
        grating, result = hog.simulate_component(
            component, LAYER, stack, wavelength, 0.2, 1.0, 2.0, 8
        )
        eps = dispersion.at_wavelength("hBN", wavelength)
        direct = rcwa.solve_grating(
            [
                rcwa.lamellar_layer(eps, jnp.eye(3), grating.fill, 0.1),
                rcwa.uniform_layer(1.85 * jnp.eye(3), 0.8),
            ],
            grating.period_um, wavelength, 0.2, 1.0, 2.0, 8,
        )  # fmt: skip
        np.testing.assert_allclose(result.R, direct.R, atol=1e-13)
        np.testing.assert_allclose(result.T, direct.T, atol=1e-13)

    def test_stack_is_differentiable_in_geometry_and_wavelength(self):
        stack = [hog.StackLayer(0.1, "hBN", background=1.0), hog.StackLayer(0.8, 1.85)]

        def absorption(fill, wavenumber):
            grating = hog.LineGrating(period_um=2.0, fill=fill)
            layers = hog.grating_layers(grating, stack, 1e4 / wavenumber)
            res = rcwa.solve_grating(layers, 2.0, 1e4 / wavenumber, 0.0, 1.0, 2.0, 8)
            return 1.0 - jnp.sum(res.R[..., 0]) - jnp.sum(res.T[..., 0])

        grad = jax.grad(absorption, argnums=(0, 1))(0.3, 1460.0)
        h_fill, h_w = 1e-6, 1e-3
        fd_fill = (absorption(0.3 + h_fill, 1460.0) - absorption(0.3 - h_fill, 1460.0)) / (
            2 * h_fill
        )
        fd_w = (absorption(0.3, 1460.0 + h_w) - absorption(0.3, 1460.0 - h_w)) / (2 * h_w)
        assert float(grad[0]) == pytest.approx(float(fd_fill), rel=1e-5)
        assert float(grad[1]) == pytest.approx(float(fd_w), rel=1e-5)  # includes hBN dispersion
