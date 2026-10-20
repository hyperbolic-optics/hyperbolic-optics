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
from hyperbolic_optics.integrations.gdsfactory import stack as hstack  # noqa: E402

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
        grating = hog.extract_line_grating(component, LAYER)
        result = rcwa.solve_grating(
            hog.grating_layers(grating, stack, wavelength), grating.period_um, wavelength,
            0.2, 1.0, 2.0, 8,
        )  # fmt: skip
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


# ---------------------------------------------------------------------------
# PDK layer stacks and the gplugins-shaped entry points
# ---------------------------------------------------------------------------

from gdsfactory.technology import LayerLevel, LogicalLayer  # noqa: E402
from gdsfactory.technology import LayerStack as PdkLayerStack  # noqa: E402

RIB = (2, 0)


def _wafer(zmin, thickness, material, mesh_order):
    return LayerLevel(
        layer=LogicalLayer(layer=gf.get_layer("WAFER")),
        zmin=zmin,
        thickness=thickness,
        material=material,
        mesh_order=mesh_order,
    )


def _tfln_stack():
    """Thin-film LiNbO3 on insulator with a 0.3 µm partial etch on layer RIB."""
    return PdkLayerStack(
        layers={
            "substrate": _wafer(-3.0, 1.0, "si", 100),
            "box": _wafer(-2.0, 2.0, "sio2", 9),
            "ln_slab": _wafer(0.0, 0.3, "ln", 2),
            "ln_rib": LayerLevel(
                layer=LogicalLayer(layer=RIB), zmin=0.3, thickness=0.3, material="ln", mesh_order=2
            ),
        }
    )


LN_X_CUT = {"ln": hog.Oriented("LiNbO3", hog.x_cut(30.0))}


class TestPdkStacks:
    def test_cell_from_pdk_preserves_derived_mask_holes(self):
        component = gf.Component()
        for width, layer in ((0.2, (2, 0)), (0.1, (3, 0))):
            half = width / 2
            component.add_polygon(
                [(-half, -half), (half, -half), (half, half), (-half, half)], layer=layer
            )
        pdk = _tfln_stack()
        pdk.layers["ln_rib"].layer = LogicalLayer(layer=(2, 0)) - LogicalLayer(layer=(3, 0))
        cell = hog.cell_from_pdk(component, pdk, ((0.4, 0), (0, 0.4)), LN_X_CUT)
        area = 0.0
        for polygon in cell.layers[0].polygons:
            x, y = polygon.T
            area += abs(x @ np.roll(y, 1) - y @ np.roll(x, 1)) / 2
        assert area == pytest.approx(0.2**2 - 0.1**2)

    def test_cell_from_pdk_wraps_shapes_across_the_cell_edge(self):
        # an odd number of grid steps per period: the footprint cannot be exactly centred
        component = gf.Component()
        fin = [(-0.05, -0.05), (0.25, -0.05), (0.25, 0.05), (-0.05, 0.05)]
        component.add_polygon(fin, layer=RIB)
        cell = hog.cell_from_pdk(component, _tfln_stack(), ((0.325, 0), (0, 0.325)), LN_X_CUT)
        area = 0.0
        for polygon in cell.layers[0].polygons:
            x, y = polygon.T
            area += abs(x @ np.roll(y, 1) - y @ np.roll(x, 1)) / 2
        assert area == pytest.approx(0.3 * 0.1)

    def test_metasurface_of_pdk_cells_reads_through_the_stack(self, tmp_path):
        fin = gf.Component()
        fin.add_polygon([(-0.12, -0.05), (0.12, -0.05), (0.12, 0.05), (-0.12, 0.05)], layer=RIB)
        pdk, lattice = _tfln_stack(), ((0.4, 0.0), (0.0, 0.4))
        sites, turns = [[0, 0], [0.4, 0]], [0, np.pi / 2]
        layout = hog.metasurface_component(sites, [fin, fin], (), lattice, turns)
        layout.write_gds(tmp_path / "pdk.gds")
        read_sites, cells, read_lattice = hog.read_metasurface(
            gf.import_gds(tmp_path / "pdk.gds"), LN_X_CUT, layer_stack=pdk
        )
        cells = [cells[k] for k in np.argsort(read_sites[:, 0])]
        assert read_lattice == lattice
        direct = hog.cell_from_pdk(fin, pdk, lattice, LN_X_CUT)
        # the fin turns on a fixed x-cut wafer: its crystal does not
        quarter = hog.turned(direct, np.pi / 2, rotate_materials=False)
        for back, expected in ((cells[0], direct), (cells[1], quarter)):
            a = hog.cell_sax_model(back, lattice, (2, 2))(wl=1.55)
            b = hog.cell_sax_model(expected, lattice, (2, 2))(wl=1.55)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=1e-10)

    def test_cell_from_pdk_keeps_each_levels_mask_and_z_intervals(self):
        component = gf.Component()
        fin = [(-0.12, -0.05), (0.12, -0.05), (0.12, 0.05), (-0.12, 0.05)]
        cap = [(-0.04, -0.1), (0.04, -0.1), (0.04, 0.1), (-0.04, 0.1)]
        component.add_polygon(fin, layer=RIB)
        component.add_polygon(cap, layer=(3, 0))
        pdk = _tfln_stack()
        pdk.layers["cap"] = LayerLevel(
            layer=LogicalLayer(layer=(3, 0)), zmin=0.7, thickness=0.2, material="sin", mesh_order=1
        )
        actual = hog.cell_from_pdk(component, pdk, ((0.4, 0), (0, 0.4)), LN_X_CUT)
        expected = hog.VerticalStack(
            (
                hog.StackLayer(0.2, "Si3N4", background=1.0, polygons=(np.asarray(cap),)),
                hog.StackLayer(0.1, 1.0),  # zmin leaves a real air spacer
                hog.StackLayer(0.3, LN_X_CUT["ln"], background=1.0, polygons=(np.asarray(fin),)),
                hog.StackLayer(0.3, LN_X_CUT["ln"]),
                hog.StackLayer(2.0, "SiO2"),
            ),
            exit="Si",
        )
        lattice = ((0.4, 0.0), (0.0, 0.4))
        a = hog.cell_sax_model(actual, lattice, (1, 1))(wl=1.55)
        b = hog.cell_sax_model(expected, lattice, (1, 1))(wl=1.55)
        assert len(actual.layers) == 5
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-11)

    def test_generic_pdk_on_and_off_a_waveguide(self):
        stack = gf.get_active_pdk().layer_stack
        component = gf.components.rectangle(size=(2, 10), layer="WG")
        on = hog.vertical_stack(component, stack, (1.0, 5.0))
        off = hog.vertical_stack(component, stack, (5.0, 5.0))
        assert [(layer.name, round(layer.thickness_um, 6)) for layer in on.layers] == [
            ("clad", 2.78), ("core", 0.22), ("box", 3.0)
        ]  # fmt: skip
        assert [layer.name for layer in off.layers] == ["clad", "box"]
        assert on.layers[1].material == "Si" and on.exit == "Si" and on.incident == 1.0

    def test_mesh_order_tie_between_materials_is_refused(self):
        stack = PdkLayerStack(layers={
            "a": _wafer(0.0, 1.0, "si", 5), "b": _wafer(0.5, 1.0, "sio2", 5),
        })  # fmt: skip
        with pytest.raises(ValueError, match="tie on mesh_order"):
            hstack.vertical_profile(hog.line_grating(), stack, (0.1, 1.0))

    def test_a_gap_in_the_stack_is_refused(self):
        stack = PdkLayerStack(layers={
            "a": _wafer(0.0, 1.0, "si", 5), "b": _wafer(2.0, 1.0, "sio2", 5),
        })  # fmt: skip
        with pytest.raises(ValueError, match="No layer-stack level covers"):
            hstack.vertical_profile(hog.line_grating(), stack, (0.1, 1.0))

    def test_unmapped_material_names_what_to_do(self):
        with pytest.raises(KeyError, match="Oriented"):
            hog.vertical_stack(hog.line_grating(layer=RIB), _tfln_stack(), (0.1, 1.0))

    def test_exit_must_be_isotropic_and_lossless(self):
        stack = hog.VerticalStack(layers=(), incident=1.0, exit=hog.Oriented("LiNbO3", hog.x_cut()))
        with pytest.raises(ValueError, match="isotropic and lossless"):
            hog.solve_stack(stack, [1.55], [0.0])

    def test_crystal_cuts(self):
        base = np.asarray(hog.material_tensor("LiNbO3", 1.55))
        along_x = np.asarray(hog.material_tensor(hog.Oriented("LiNbO3", hog.x_cut(0.0)), 1.55))
        along_y = np.asarray(hog.material_tensor(hog.Oriented("LiNbO3", hog.x_cut(90.0)), 1.55))
        assert along_x[0, 0] == pytest.approx(base[2, 2]) and along_x[2, 2] == pytest.approx(
            base[0, 0]
        )
        assert along_y[1, 1] == pytest.approx(base[2, 2])
        np.testing.assert_allclose(
            hog.material_tensor(hog.Oriented("LiNbO3", hog.z_cut()), 1.55), base
        )

    def test_warns_outside_a_materials_range(self):
        with pytest.warns(UserWarning, match="outside"):
            hog.material_tensor("LiNbO3", 6.0)


class TestPlanarResponse:
    @staticmethod
    def _stack():
        ln = hog.Oriented("LiNbO3", hog.x_cut(30.0))
        return hog.VerticalStack(
            layers=(hog.StackLayer(0.6, ln, name="ln"), hog.StackLayer(2.0, "SiO2", name="box")),
            incident=1.0,
            exit="Si",
        )

    def test_jax_and_numpy_engines_agree(self):
        args = (self._stack(), np.linspace(1.5, 1.6, 4), [0.0, 35.0, 70.0], [0.0, 50.0])
        jax_result = hog.solve_stack(*args, solver="jax")
        numpy_result = hog.solve_stack(*args, solver="numpy")
        # a Crystal Layer adds 1e-8 rad to rotationY, so the engines differ at that level
        np.testing.assert_allclose(jax_result.r, numpy_result.r, atol=1e-7)
        np.testing.assert_allclose(jax_result.t, numpy_result.t, atol=1e-7)
        np.testing.assert_allclose(jax_result.R.sum(-2) + jax_result.T.sum(-2), 1.0, atol=1e-12)
        assert jax_result.R[..., 1, 0].max() > 1e-3, "the rotated x-cut crystal converts p to s"

    def test_mueller_carries_the_converted_power_in_the_right_channel(self):
        result = hog.solve_stack(self._stack(), [1.55], [40.0], [20.0])
        mueller = result.mueller()[0, 0, 0]
        p_in = mueller @ np.array([1.0, 1.0, 0.0, 0.0])
        assert (p_in[0] - p_in[1]) / 2 == pytest.approx(float(result.R[0, 0, 0, 1, 0]), rel=1e-12)
        assert mueller[0, 0] == pytest.approx(0.5 * float(result.R[0, 0, 0].sum()), rel=1e-12)

    def test_ellipsometry_matches_the_jones_class(self):
        from hyperbolic_optics.jones import Jones
        from hyperbolic_optics.structure import Structure

        ours = hog.solve_stack(self._stack(), [1.55], [55.0], [20.0], solver="numpy").ellipsometry()
        tensor = np.asarray(
            hog.material_tensor(hog.Oriented("LiNbO3", (0.0, 90.0, 50.0)), 1.55)
        )  # x_cut(30) turned by the 20° azimuth
        keys = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
        payload = {
            "ScenarioData": {"type": "Simple", "incidentAngle": 55.0, "azimuthal_angle": 0.0,
                             "frequency": 1e4 / 1.55},
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 1.0},
                {"type": "Crystal Layer", "thickness": 0.6, "material": {
                    f"eps_{k}": {"real": float(v.real), "imag": float(v.imag)}
                    for k, v in zip(keys, tensor.reshape(-1))}},
                {"type": "Crystal Layer", "thickness": 2.0, "material": "SiO2"},
                {"type": "Semi Infinite Isotropic Layer",
                 "permittivity": float(hog.material_tensor("Si", 1.55)[0, 0].real)},
            ],
        }  # fmt: skip
        structure = Structure()
        structure.execute(payload)
        theirs = Jones(structure).ellipsometric_parameters()
        for key in ("Psi", "Delta", "Psi_ps", "Delta_ps", "Psi_sp", "Delta_sp"):
            assert float(ours[key].ravel()[0]) == pytest.approx(float(theirs[key]), abs=1e-5), key

    def test_simulate_stack_from_a_layout(self):
        stack, result = hog.simulate_stack(
            gf.components.rectangle(size=(2, 10), layer="WG"),
            gf.get_active_pdk().layer_stack,
            (1.0, 5.0),
            [1.55],
            [0.0, 20.0],
        )
        assert result.R.shape == (1, 2, 1, 2, 2)
        np.testing.assert_allclose(result.R.sum(-2) + result.T.sum(-2), 1.0, atol=1e-12)
        assert [layer.name for layer in stack.layers] == ["clad", "core", "box"]


class TestPdkGratings:
    def test_partial_etch_is_found_from_the_stack(self):
        component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        grating, stack = hog.grating_stack(component, RIB, _tfln_stack(), LN_X_CUT)
        names = [(layer.name, round(layer.thickness_um, 6)) for layer in stack.layers]
        assert names == [("ln_rib / ambient", 0.3), ("ln_slab", 0.3), ("box", 2.0)]
        assert stack.layers[0].background == 1.0 and stack.layers[1].background is None
        assert stack.exit == "Si"

    def test_simulate_grating_equals_the_hand_built_stack(self):
        component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        grating, _, response = hog.simulate_grating(
            component, RIB, _tfln_stack(), [1.55], [10.0], max_order=12, materials=LN_X_CUT
        )
        ln = hog.material_tensor(LN_X_CUT["ln"], 1.55)
        direct = rcwa.solve_grating(
            [
                rcwa.lamellar_layer(ln, jnp.eye(3), grating.fill, 0.3),
                rcwa.uniform_layer(ln, 0.3),
                rcwa.uniform_layer(hog.material_tensor("SiO2", 1.55), 2.0),
            ],
            grating.period_um, 1.55, np.radians(10.0), 1.0,
            float(hog.material_tensor("Si", 1.55)[0, 0].real), 12,
        )  # fmt: skip
        np.testing.assert_allclose(response.R[0, 0], direct.R, atol=1e-12)
        np.testing.assert_allclose(response.T[0, 0], direct.T, atol=1e-12)
        total = response.R.sum((-3, -2)) + response.T.sum((-3, -2))
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        assert response.mueller(0).shape == (1, 1, 4, 4)

    def test_a_second_pattern_inside_the_ridges_is_refused(self):
        # a copy: PCell outputs are cached and locked against modification
        component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB).copy()
        stack = _tfln_stack()
        extra = LayerLevel(layer=LogicalLayer(layer=(3, 0)), zmin=0.6, thickness=0.1,
                           material="sio2", mesh_order=1)  # fmt: skip
        stack.layers["cap"] = extra
        component.add_polygon([(0.0, 0.0), (0.2, 0.0), (0.2, 20.0), (0.0, 20.0)], layer=(3, 0))
        with pytest.raises(ValueError, match="varies within the ridges"):
            hog.grating_stack(component, RIB, stack, LN_X_CUT)


class TestLuxtelligencePdk:
    """Against Luxtelligence's open-source LNOI400 PDK, when it is installed.

    ``uv run --with lnoi400 pytest tests/test_gdsfactory.py -k Luxtelligence``.
    It activates its own PDK, so it restores the generic one afterwards.
    """

    @pytest.fixture(autouse=True)
    def lnoi400(self):
        module = pytest.importorskip("lnoi400")
        module.PDK.activate()
        yield module
        gf.gpdk.PDK.activate()

    def test_the_pdks_rib_waveguide_columns(self, lnoi400):
        rib = lnoi400.PDK.get_component("straight_rwg1000")
        stack = lnoi400.PDK.layer_stack
        ln = {"ln": hog.Oriented("LiNbO3", hog.x_cut())}
        x = rib.dbbox().center().x
        centre = hog.vertical_stack(rib, stack, (x, 0.0), ln)
        beside = hog.vertical_stack(rib, stack, (x, 3.0), ln)
        assert [(ly.name, round(ly.thickness_um, 6)) for ly in centre.layers] == [
            ("clad", 1.6), ("ridge", 0.2), ("slab", 0.2), ("box", 4.7)
        ]  # fmt: skip
        assert [ly.name for ly in beside.layers] == ["clad", "slab", "box"]
        assert centre.exit == "Si"

    def test_a_grating_on_ln_ridge_is_a_clad_filled_partial_etch(self, lnoi400):
        component = gf.Component()
        component.add_ref(hog.line_grating(period_um=1.2, fill=0.5, n_periods=8, layer="LN_RIDGE"))
        component.add_polygon([(-5, -5), (15, -5), (15, 25), (-5, 25)], layer="LN_SLAB")
        _, stack, response = hog.simulate_grating(
            component, "LN_RIDGE", lnoi400.PDK.layer_stack, [1.55], [30.0], max_order=10,
            materials={"ln": hog.Oriented("LiNbO3", hog.x_cut(0.0))},
        )  # fmt: skip
        patterned = [layer for layer in stack.layers if layer.background is not None]
        assert [(ly.name, round(ly.thickness_um, 6)) for ly in patterned] == [("ridge / clad", 0.2)]
        assert patterned[0].background == "SiO2"
        total = response.R.sum((-3, -2)) + response.T.sum((-3, -2))
        np.testing.assert_allclose(total, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2D-periodic patterns
# ---------------------------------------------------------------------------

from hyperbolic_optics import rcwa2d  # noqa: E402


def _vertices(array):
    return tuple(tuple(float(c) for c in row) for row in np.asarray(array))


DISC = _vertices(rcwa2d.ellipse((0.2, 0.25), (0.12, 0.08), n_vertices=24, rotation_rad=0.4))
BAR = ((0.45, 0.05), (0.55, 0.05), (0.55, 0.45), (0.45, 0.45))


class TestUnitCells:
    def test_round_trip_through_a_gds_file(self, tmp_path):
        component = hog.periodic_array((0.6, 0.5), (DISC, BAR), (5, 4), layer=LAYER)
        path = tmp_path / "array.gds"
        component.write_gds(path)
        cell = hog.extract_unit_cell(gf.import_gds(path), LAYER)
        assert cell.period_um == pytest.approx((0.6, 0.5))
        assert cell.n_cells == (5, 4)
        drawn = component.info["polygons"]
        origin = np.min([np.min(p, axis=0) for p in drawn], axis=0)
        np.testing.assert_allclose(cell.origin_um, origin, atol=1e-9)
        got = sorted(np.round(p + origin, 6).tolist() for p in cell.polygons)
        want = sorted(np.round(np.asarray(p), 6).tolist() for p in drawn)
        assert [sorted(map(tuple, p)) for p in got] == [sorted(map(tuple, p)) for p in want]

    def test_a_repeated_shape_is_one_motif(self):
        """A dimer of identical bars: the 0.25 µm spacing is not the period."""
        left = ((0.05, 0.1), (0.2, 0.1), (0.2, 0.4), (0.05, 0.4))
        right = tuple((x + 0.25, y) for x, y in left)
        component = hog.periodic_array((0.8, 0.5), (left, right), (4, 3), layer=LAYER)
        for cell in (
            hog.extract_unit_cell(component, LAYER),
            hog.extract_unit_cell(component, LAYER, period_um=(0.8, 0.5)),
        ):
            assert cell.period_um == pytest.approx((0.8, 0.5))
            assert len(cell.polygons) == 2 and cell.n_cells == (4, 3)
        with pytest.raises(ValueError, match="overlap|complete lattice"):
            hog.extract_unit_cell(component, LAYER, period_um=(0.25, 0.5))

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            ("missing", "complete lattice"),
            ("overlap", "overlap"),
            ("hole", "hole"),
        ],
    )
    def test_refuses_what_is_not_a_periodic_array(self, edit, message):
        component = gf.Component()
        for i in range(3):
            for j in range(3):
                if edit == "missing" and (i, j) == (1, 2):
                    continue
                x, y = i * 1.0, j * 1.0
                component.add_polygon(
                    [(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)], layer=LAYER
                )
                if edit == "overlap":
                    component.add_polygon(
                        [(x + 0.4, y + 0.1), (x + 0.9, y + 0.1), (x + 0.9, y + 0.3),
                         (x + 0.4, y + 0.3)],
                        layer=LAYER,
                    )  # fmt: skip
        if edit == "hole":
            ring = gf.kdb.Region(gf.kdb.Box(0, 0, 3000, 3000)) - gf.kdb.Region(
                gf.kdb.Box(1000, 1000, 2000, 2000)
            )
            component = gf.Component()
            for i in range(2):
                component.shapes(component.kcl.layer(*LAYER)).insert(ring.moved(i * 4000, 0))
        with pytest.raises(ValueError, match=message):
            hog.extract_unit_cell(component, LAYER, period_um=(1.0, 1.0))

    def test_etched_pillars_in_a_pdk_stack(self):
        component = hog.periodic_array((0.6, 0.5), (DISC,), (6, 6), layer=RIB)
        cell, stack = hog.crossed_grating_stack(component, RIB, _tfln_stack(), LN_X_CUT)
        names = [(layer.name, round(layer.thickness_um, 6)) for layer in stack.layers]
        assert names == [("ln_rib / ambient", 0.3), ("ln_slab", 0.3), ("box", 2.0)]
        assert stack.exit == "Si" and cell.n_cells == (6, 6)

    def test_simulate_equals_the_hand_built_cell(self):
        component = hog.periodic_array((0.6, 0.5), (DISC, BAR), (4, 4), layer=RIB)
        cell, _, response = hog.simulate_crossed_grating(
            component, RIB, _tfln_stack(), [1.55], [10.0], azimuth_deg=20.0, max_order=(3, 3),
            materials=LN_X_CUT,
        )  # fmt: skip
        ln = hog.material_tensor(LN_X_CUT["ln"], 1.55)
        shapes = tuple(rcwa2d.Shape(jnp.asarray(p), ln) for p in cell.polygons)
        eps_si = float(hog.material_tensor("Si", 1.55)[0, 0].real)
        direct = rcwa2d.solve_crossed_grating(
            [
                rcwa2d.PatternedLayer(0.3, jnp.eye(3, dtype=complex), shapes),
                rcwa2d.uniform(ln, 0.3),
                rcwa2d.uniform(hog.material_tensor("SiO2", 1.55), 2.0),
            ],
            cell.period_um, 1.55, np.radians(10.0), 1.0, eps_si, (3, 3),
            azimuth_rad=np.radians(20.0),
        )  # fmt: skip
        np.testing.assert_allclose(response.R[0, 0], direct.R, atol=1e-12)
        np.testing.assert_allclose(response.T[0, 0], direct.T, atol=1e-12)
        total = response.R.sum((-3, -2)) + response.T.sum((-3, -2))
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        assert response.mueller((0, 0)).shape == (1, 1, 4, 4)

    def test_stripes_drawn_as_a_2d_array_are_the_line_grating(self):
        """Rectangles spanning the y period touch across the cell edge: a line grating."""
        stripe = ((0.0, 0.0), (0.45, 0.0), (0.45, 0.3), (0.0, 0.3))
        array = hog.periodic_array((0.9, 0.3), (stripe,), (10, 20), layer=RIB)
        _, _, crossed = hog.simulate_crossed_grating(
            array, RIB, _tfln_stack(), [1.55], [10.0], max_order=(6, 1), materials=LN_X_CUT
        )
        lines = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        _, _, line = hog.simulate_grating(
            lines, RIB, _tfln_stack(), [1.55], [10.0], max_order=6, materials=LN_X_CUT
        )
        row = crossed.orders[:, 1] == 0
        np.testing.assert_allclose(crossed.R[0, 0][row], line.R[0, 0], atol=1e-12)
        np.testing.assert_allclose(crossed.T[0, 0][row], line.T[0, 0], atol=1e-12)

    def test_a_layer_patterned_differently_is_refused(self):
        component = hog.periodic_array((0.6, 0.5), (DISC,), (6, 6), layer=RIB).copy()
        component.add_polygon([(0, 0), (0.6, 0), (0.6, 0.5), (0, 0.5)], layer=(3, 0))
        stack = _tfln_stack()
        stack.layers["cap"] = LayerLevel(
            layer=LogicalLayer(layer=(3, 0)), zmin=0.6, thickness=0.1, material="sio2",
            mesh_order=1,
        )  # fmt: skip
        with pytest.raises(ValueError, match="varies"):
            hog.crossed_grating_stack(component, RIB, stack, LN_X_CUT)

    def test_differentiable_in_the_shape(self):
        stack = [hog.StackLayer(0.3, hog.Oriented("LiNbO3", hog.x_cut(30.0)), background=1.0)]

        def reflectance(scale):
            polygon = jnp.asarray(DISC) * scale
            cell = hog.UnitCell(period_um=(0.6, 0.5), polygons=(polygon,))
            layers = hog.crossed_grating_layers(cell, stack, 1.55)
            result = rcwa2d.solve_crossed_grating(layers, (0.6, 0.5), 1.55, 0.2, 1.0, 2.1, (2, 2))
            return result.R.sum()

        grad = float(jax.grad(reflectance)(1.0))
        fd = float((reflectance(1.0 + 1e-5) - reflectance(1.0 - 1e-5)) / 2e-5)
        assert grad == pytest.approx(fd, rel=1e-5)


def test_a_pdk_stack_as_a_sax_model():
    from hyperbolic_optics import bcm

    component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
    column = hog.vertical_stack(component, _tfln_stack(), (0.2, 10.0), LN_X_CUT)
    model = hog.stack_sax_model(column)
    sd = model(wl=1.55, kx=0.3, azimuth=0.2)
    tensors, thickness = [], []
    for layer in column.layers:
        tensor = hog.material_tensor(layer.material, 1.55, layer.rotation_deg)
        tensors.append(bcm.rotate_tensor(tensor, bcm.rotation_matrix(0.0, 0.0, 0.2)))
        thickness.append(layer.thickness_um)
    eps_si = float(hog.material_tensor("Si", 1.55)[0, 0].real)
    direct = bcm.stack_smatrix(jnp.stack(tensors), jnp.array(thickness), 1.55, 0.3, 1.0, eps_si)
    np.testing.assert_allclose(sd["top@p", "top@s"], direct.s[1, 0], atol=1e-13)
    np.testing.assert_allclose(sd["bottom@s", "top@p"], direct.s[0, 3], atol=1e-13)
    with pytest.raises(ValueError, match="unpatterned"):
        _, stack = hog.grating_stack(component, RIB, _tfln_stack(), LN_X_CUT)
        hog.stack_sax_model(stack)


HEX_UM = ((0.6, 0.0), (0.3, 0.6 * np.sqrt(3) / 2))
SMALL_DISC = _vertices(rcwa2d.ellipse((0.0, 0.0), (0.12, 0.12), n_vertices=24))


class TestOtherLattices:
    def test_hexagonal_round_trip_infers_the_lattice(self, tmp_path):
        component = hog.periodic_array(polygons=(SMALL_DISC,), n_cells=(6, 5), layer=LAYER,
                                       lattice_um=HEX_UM)  # fmt: skip
        path = tmp_path / "hex.gds"
        component.write_gds(path)
        cell = hog.extract_unit_cell(gf.import_gds(path), LAYER)
        assert cell.period_um is None and cell.n_cells == (6, 5)
        np.testing.assert_allclose(cell.lattice_um, component.info["lattice_um"], atol=1e-9)
        np.testing.assert_allclose(cell.lattice_um, HEX_UM, atol=1e-3)  # snapped to 1 nm
        assert isinstance(cell.lattice, rcwa2d.Lattice)

    def test_oblique_lattice_and_a_two_shape_motif(self):
        oblique = ((0.7, 0.0), (0.25, 0.55))
        bar = ((0.3, -0.05), (0.42, -0.05), (0.42, 0.1), (0.3, 0.1))
        component = hog.periodic_array(polygons=(SMALL_DISC, bar), n_cells=(4, 4), layer=LAYER,
                                       lattice_um=oblique)  # fmt: skip
        cell = hog.extract_unit_cell(component, LAYER)
        np.testing.assert_allclose(cell.lattice_um, oblique, atol=1e-9)
        assert len(cell.polygons) == 2 and cell.n_cells == (4, 4)

    def test_a_hexagonal_dimer(self):
        pair = (SMALL_DISC, tuple((x + 0.3, y) for x, y in SMALL_DISC))
        wide = ((1.2, 0.0), (0.6, 1.2 * np.sqrt(3) / 2))
        component = hog.periodic_array(polygons=pair, n_cells=(4, 4), layer=LAYER, lattice_um=wide)
        for cell in (
            hog.extract_unit_cell(component, LAYER),
            hog.extract_unit_cell(component, LAYER, lattice_um=wide),
        ):
            np.testing.assert_allclose(cell.lattice_um, wide, atol=1e-3)
            assert len(cell.polygons) == 2

    def test_simulating_a_hexagonal_layout_equals_the_hand_built_cell(self):
        component = hog.periodic_array(polygons=(SMALL_DISC,), n_cells=(6, 6), layer=RIB,
                                       lattice_um=HEX_UM)  # fmt: skip
        cell, stack, response = hog.simulate_crossed_grating(
            component, RIB, _tfln_stack(), [1.55], [10.0], azimuth_deg=20.0, max_order=(3, 3),
            materials=LN_X_CUT, truncation="circular",
        )  # fmt: skip
        assert [layer.name for layer in stack.layers][0] == "ln_rib / ambient"
        ln = hog.material_tensor(LN_X_CUT["ln"], 1.55)
        shapes = tuple(rcwa2d.Shape(jnp.asarray(p), ln) for p in cell.polygons)
        direct = rcwa2d.solve_crossed_grating(
            [
                rcwa2d.PatternedLayer(0.3, jnp.eye(3, dtype=complex), shapes),
                rcwa2d.uniform(ln, 0.3),
                rcwa2d.uniform(hog.material_tensor("SiO2", 1.55), 2.0),
            ],
            cell.lattice, 1.55, np.radians(10.0), 1.0,
            float(hog.material_tensor("Si", 1.55)[0, 0].real), (3, 3),
            azimuth_rad=np.radians(20.0), truncation="circular",
        )  # fmt: skip
        np.testing.assert_allclose(response.R[0, 0], direct.R, atol=1e-12)


# ---------------------------------------------------------------------------
# gplugins-style S-parameters
# ---------------------------------------------------------------------------

from hyperbolic_optics import bcm  # noqa: E402
from hyperbolic_optics.integrations.gdsfactory import sparameters as hsp  # noqa: E402


class TestSparameters:
    def test_cache_follows_pdk_thickness_and_callable_material(self, tmp_path):
        component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=3, layer=RIB)
        pdk = _tfln_stack()
        eps = [4.0]
        material = {"ln": lambda wl: jnp.zeros_like(wl) + eps[0]}

        def solve(overwrite=False):
            return hsp.write_sparameters_stack(
                component,
                pdk,
                (0.2, 10.0),
                [1.55],
                materials=material,
                dirpath=tmp_path,
                overwrite=overwrite,
            )["top@p,bottom@p"]

        first = solve()
        pdk.layers["ln_rib"].thickness = 0.5
        second = solve()
        assert abs(first - second).max() > 0.1
        np.testing.assert_array_equal(second, solve(overwrite=True))
        eps[0] = 6.0
        third = solve()
        assert abs(second - third).max() > 0.1
        np.testing.assert_array_equal(third, solve(overwrite=True))

    def test_cell_cache_round_trip_and_reuse(self, tmp_path, monkeypatch):
        cell = hog.VerticalStack((hog.StackLayer(0.2, "Si3N4"),), exit="SiO2")
        lattice = ((0.4, 0.0), (0.0, 0.4))
        args = (cell, lattice, [1.5, 1.55])
        expected = hog.cell_sax_model(cell, lattice, (0, 0))(wl=jnp.array([1.5, 1.55]), kx=0.3)
        data = hsp.write_sparameters_cell(*args, max_order=(0, 0), kx=0.3, dirpath=tmp_path)
        for key, value in expected.items():
            np.testing.assert_allclose(data[",".join(key)], value, atol=1e-12)
        model = hsp.model_from_npz(data)
        np.testing.assert_allclose(
            model(1.55)["top@p", "bottom@p"], expected["top@p", "bottom@p"][1], atol=1e-12
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("A cache hit must not solve again")

        monkeypatch.setattr(hsp, "cell_sax_model", lambda *a, **kw: forbidden)
        cached = hsp.write_sparameters_cell(*args, max_order=(0, 0), kx=0.3, dirpath=tmp_path)
        for key in data:
            np.testing.assert_array_equal(cached[key], data[key])

    def test_stack_sparameters(self, tmp_path):
        component = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        sp = hsp.write_sparameters_stack(
            component, _tfln_stack(), (0.2, 10.0), [1.5, 1.55], angle_deg=20.0,
            materials=LN_X_CUT, dirpath=tmp_path,
        )  # fmt: skip
        column = hog.vertical_stack(component, _tfln_stack(), (0.2, 10.0), LN_X_CUT)
        tensors = jnp.stack(
            [
                hog.material_tensor(layer.material, 1.55, layer.rotation_deg)
                for layer in column.layers
            ]
        )
        thickness = jnp.array([layer.thickness_um for layer in column.layers])
        eps_si = float(hog.material_tensor("Si", 1.55)[0, 0].real)
        direct = bcm.stack_smatrix(tensors, thickness, 1.55, np.sin(np.radians(20.0)), 1.0, eps_si)
        for i_in, port_in in enumerate(bcm.PORTS):
            for i_out, port_out in enumerate(bcm.PORTS):
                assert complex(sp[f"{port_in},{port_out}"][1]) == pytest.approx(
                    complex(direct.s[i_out, i_in]), abs=1e-12
                )

    def test_model_from_npz_interpolates_with_modes_kept(self, tmp_path):
        data = {
            "a@TE,b@TE": np.array([1.0 + 0j, 0.0 + 1j]),
            "a@TE,b@TM": np.array([0.5 + 0j, 0.5 + 0j]),
            "wavelengths": np.array([1.5, 1.6]),
        }
        path = tmp_path / "s.npz"
        np.savez(path, **data)
        model = hsp.model_from_npz(path)
        sd = model(wl=np.array([1.5, 1.55]))
        np.testing.assert_allclose(sd["a@TE", "b@TE"], [1.0, 0.5 + 0.5j])
        np.testing.assert_allclose(sd["a@TE", "b@TM"], [0.5, 0.5])

    def test_the_hash_follows_the_geometry(self):
        one = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        same = hog.line_grating(period_um=0.9, fill=0.5, n_periods=10, layer=RIB)
        other = hog.line_grating(period_um=0.9, fill=0.4, n_periods=10, layer=RIB)
        assert hsp.component_hash(one) == hsp.component_hash(same) != hsp.component_hash(other)


class TestProcessGeometry:
    @pytest.mark.parametrize("bias", [-0.01, 0.01])
    def test_periodic_stripe_offsets_have_no_seam(self, bias):
        component = _rectangles(gf.Component(), [(-0.3, 0.3, -0.05, 0.05)], layer=RIB)
        pdk = _tfln_stack()
        pdk.layers["ln_rib"].bias = bias
        cell = hog.cell_from_pdk(component, pdk, ((0.6, 0), (0, 0.6)), LN_X_CUT, sidewall_slices=1)
        patterned = [lv for lv in cell.layers if lv.polygons is not None]
        assert len(patterned) == 1
        polygons = patterned[0].polygons
        area = sum(
            abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2
            for p in polygons
        )
        # An infinite stripe changes width only; the cell boundary is not a sidewall.
        assert area == pytest.approx(0.6 * (0.1 + 2 * bias))

    def test_midpoint_sidewalls_and_bias_follow_pdk_reference_height(self):
        c = _rectangles(gf.Component(), [(-0.1, 0.1, -0.05, 0.05)], layer=RIB)
        pdk = _tfln_stack()
        level = pdk.layers["ln_rib"]
        level.sidewall_angle, level.width_to_z, level.bias = 13.0, 1.0, 0.005
        with pytest.warns(UserWarning, match="sidewall_angle/bias/z_to_bias"):
            hog.cell_from_pdk(c, pdk, ((0.6, 0), (0, 0.6)), LN_X_CUT)
        cell = hog.cell_from_pdk(c, pdk, ((0.6, 0), (0, 0.6)), LN_X_CUT, sidewall_slices=4)
        patterned = [lv for lv in cell.layers if lv.polygons is not None]
        assert len(patterned) == 4
        for i, lv in enumerate(patterned):
            z = 0.6 - (i + 0.5) * 0.075
            offset = round((0.005 + (0.6 - z) * np.tan(np.radians(13))) / c.kcl.dbu) * c.kcl.dbu
            np.testing.assert_allclose(
                np.ptp(lv.polygons[0], axis=0), [0.2 + 2 * offset, 0.1 + 2 * offset]
            )
            assert lv.thickness_um == pytest.approx(0.075)
            assert lv.material == LN_X_CUT["ln"]
        assert level.thickness == 0.3  # extraction never mutates the PDK

    def test_slices_refuse_unsupported_profiles(self):
        c = _rectangles(gf.Component(), [(-0.1, 0.1, -0.05, 0.05)], layer=RIB)
        pdk = _tfln_stack()
        pdk.layers["ln_rib"].z_to_bias = ([0.0, 1.0], [0.0, 0.01])
        with pytest.raises(ValueError, match="scalar bias only"):
            hog.cell_from_pdk(c, pdk, ((0.6, 0), (0, 0.6)), LN_X_CUT, sidewall_slices=4)
        with pytest.raises(ValueError, match="nonnegative integer"):
            hog.cell_from_pdk(c, pdk, ((0.6, 0), (0, 0.6)), LN_X_CUT, sidewall_slices=-1)

    def test_tapered_cell_spectrum_refines_and_zero_angle_is_unchanged(self):
        c = _rectangles(gf.Component(), [(-0.1, 0.1, -0.05, 0.05)], layer=RIB)
        pdk = _tfln_stack()
        lattice = ((0.6, 0), (0, 0.6))

        def solve(n):
            cell = hog.cell_from_pdk(c, pdk, lattice, LN_X_CUT, sidewall_slices=n)
            return complex(hog.cell_sax_model(cell, lattice, (1, 1))(wl=1.55)["top@p", "bottom@s"])

        np.testing.assert_allclose(solve(0), solve(4), atol=1e-12)
        pdk.layers["ln_rib"].sidewall_angle = 13.0
        pdk.layers["ln_rib"].width_to_z = 1.0
        a, b, c0 = [solve(n) for n in (2, 4, 8)]
        assert abs(c0 - b) < abs(b - a)
        assert abs(c0 - b) < 2e-4


class TestCellSparameters:
    def test_pdk_cell_cache_and_sax_agree(self, tmp_path, monkeypatch):
        from hyperbolic_optics.integrations.gdsfactory import sparameters

        component = _rectangles(gf.Component(), [(-0.1, 0.1, -0.05, 0.05)], layer=RIB)
        pdk, lattice, wl = _tfln_stack(), ((0.4, 0), (0, 0.4)), [1.5, 1.55]
        pdk.layers["ln_rib"].sidewall_angle = 13.0
        cell = hog.cell_from_pdk(component, pdk, lattice, LN_X_CUT, sidewall_slices=2)
        saved = hog.write_sparameters_cell(cell, lattice, wl, (1, 1), dirpath=tmp_path)
        reference = hog.cell_sax_model(cell, lattice, (1, 1))(wl=jnp.array(wl))
        for key, values in reference.items():
            np.testing.assert_allclose(saved[",".join(key)], values, atol=1e-12)

        def cached_only(*args, **kwargs):
            def solve(**kwargs):
                raise AssertionError("A cache hit must not invoke the solver")

            return solve

        monkeypatch.setattr(sparameters, "cell_sax_model", cached_only)
        again = hog.write_sparameters_cell(cell, lattice, wl, (1, 1), dirpath=tmp_path)
        np.testing.assert_array_equal(again["top@p,bottom@s"], saved["top@p,bottom@s"])
