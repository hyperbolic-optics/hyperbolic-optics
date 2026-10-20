"""Tests for metasurface design (``hyperbolic_optics.metasurfaces``) and its layout.

- a uniform film's library is its Airy transmission; a batch is compiled once,
  and a zero-thickness layer changes nothing (so cells of different layer
  counts share a batch);
- two half-wave layers twisted by α are a rotator by 2α: circular light keeps
  its handedness and gains −2α;
- a turned half-wave cell gives the Pancharatnam–Berry phase 2θ;
- the hyperbolic lens phase focuses at f, at the diffraction limit, and a
  TiO₂-pillar lens designed from its RCWA library focuses there too;
- the focal intensity's gradient with respect to per-site twists matches
  finite differences;
- a layout of cells differing in layer count, twist, crystal cut and shape
  reads back into the same cells, and they solve to the same Jones matrices;
- a cell's SAX model is its zeroth-order S-matrix, and for an unpatterned cell
  the planar stack's.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, rcwa2d  # noqa: E402
from hyperbolic_optics import metasurfaces as ms  # noqa: E402

WAVELENGTH, PITCH, N_GLASS = 0.633, 0.35, 1.45
EYE = jnp.eye(3, dtype=complex)
LEFT, RIGHT = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)


def _film_t(n, thickness, n_in, n_out, wavelength):
    """Normal-incidence amplitude transmission of a film (Airy)."""
    k = 2 * np.pi * n * thickness / wavelength
    t1, t2 = 2 * n_in / (n_in + n), 2 * n / (n + n_out)
    r1, r2 = (n_in - n) / (n_in + n), (n - n_out) / (n + n_out)
    return t1 * t2 * np.exp(1j * k) / (1 + r1 * r2 * np.exp(2j * k))


def _disc(radius):
    shape = rcwa2d.Shape(rcwa2d.ellipse((PITCH / 2, PITCH / 2), (radius, radius), 32), 2.4**2 * EYE)
    return [rcwa2d.PatternedLayer(0.6, EYE, (shape,))]


def _twisted_pair(alpha, wavelength=1.55, n_o=1.50, n_e=1.60):
    """Two half-wave layers (8 and 7.5 wavelengths thick, so no reflections), twisted by α."""
    eps = jnp.diag(jnp.array([n_e**2, n_o**2, n_o**2], dtype=complex))
    thickness = wavelength / (2 * (n_e - n_o))
    turns = [bcm.rotation_matrix(0.0, 0.0, k * alpha) for k in range(2)]
    return [rcwa2d.uniform(bcm.rotate_tensor(eps, r), thickness) for r in turns]


class TestLibrary:
    def test_uniform_film_is_fresnel(self):
        def cell(n):
            return [rcwa2d.uniform(n**2 * EYE, 0.3)]

        indices = np.array([1.5, 2.0, 2.4])
        t = ms.cell_library(cell, indices, (PITCH, PITCH), WAVELENGTH, N_GLASS**2, 1.0, (1, 1))
        expected = _film_t(indices, 0.3, N_GLASS, 1.0, WAVELENGTH)
        np.testing.assert_allclose(t[:, 0, 0], expected, atol=1e-10)
        np.testing.assert_allclose(t[:, 1, 1], expected, atol=1e-10)
        np.testing.assert_allclose(t[:, 0, 1], 0, atol=1e-12)

    def test_pillar_library_spans_the_circle(self):
        radii = np.linspace(0.04, 0.16, 9)
        t = ms.cell_library(_disc, radii, (PITCH, PITCH), WAVELENGTH, N_GLASS**2, 1.0, (2, 2),
                            batch_size=4)  # fmt: skip
        assert np.ptp(np.unwrap(np.angle(t[:, 0, 0]))) > 1.6 * np.pi
        power = np.abs(t[:, 0, 0]) ** 2 / N_GLASS
        assert np.all(power <= 1 + 1e-9) and np.all(power > 0.8)
        np.testing.assert_allclose(t[:, 0, 0], t[:, 1, 1], atol=1e-6)  # a disc does not convert

    def test_zero_thickness_layers_change_nothing(self):
        def padded(radius):
            spare = rcwa2d.Shape(rcwa2d.rectangle((0.1, 0.1), (0.05, 0.05)), 3.0 * EYE)
            return [rcwa2d.PatternedLayer(0.0, 2.0 * EYE, (spare,)), *_disc(radius)]

        radii = np.array([0.08, 0.12])
        args = ((PITCH, PITCH), WAVELENGTH, N_GLASS**2, 1.0, (2, 2))
        np.testing.assert_array_equal(
            ms.cell_library(padded, radii, *args), ms.cell_library(_disc, radii, *args)
        )

    def test_twisted_half_wave_pair_is_a_rotator(self):
        alphas = np.linspace(0.0, 1.5, 6)
        t = ms.cell_library(_twisted_pair, alphas, (0.5, 0.5), 1.55, 1.55**2, 1.55**2, (0, 0))
        kept = np.einsum("i,kij,j->k", np.conj(LEFT), t, LEFT)
        converted = np.einsum("i,kij,j->k", np.conj(RIGHT), t, LEFT)
        np.testing.assert_allclose(kept, kept[0] * np.exp(-2j * alphas), atol=1e-9)
        np.testing.assert_allclose(converted, 0, atol=1e-9)


class TestDesign:
    def test_geometric_phase(self):
        half_wave = np.diag([1.0, -1.0]).astype(complex)
        for theta in (0.0, 0.3, 1.1, 2.5):
            jones = np.asarray(ms.geometric_phase_jones(half_wave, theta))
            np.testing.assert_allclose(np.conj(RIGHT) @ jones @ LEFT, np.exp(2j * theta),
                                       atol=1e-12)  # fmt: skip
            assert abs(np.conj(LEFT) @ jones @ LEFT) < 1e-12

    def test_nearest_entry(self):
        library_phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        library = np.exp(1j * library_phases)[:, None, None] * np.eye(2)
        target = np.array([[0.1, 3.0], [6.2, -1.0]])
        indices, realised = ms.choose_cells(target, library)
        wrapped = np.angle(np.exp(1j * (library_phases[indices] - target)))
        assert np.all(np.abs(wrapped) <= np.pi / 16 + 1e-12)
        np.testing.assert_allclose(realised, np.exp(1j * library_phases[indices]))

    def test_ideal_lens_focuses_at_the_diffraction_limit(self):
        radius, focal = 8.0, 16.0
        sites = ms.square_sites(radius, PITCH)
        assert np.all(np.hypot(*sites.T) <= radius)
        phase = ms.lens_phase(sites[:, 0], sites[:, 1], focal, WAVELENGTH)
        coords, field = ms.sample_grid(sites, jnp.exp(1j * phase), PITCH)
        peaks = [float(jnp.abs(ms.focus(coords, coords, field, WAVELENGTH, z)).max())
                 for z in (focal - 3, focal, focal + 3)]  # fmt: skip
        assert peaks[1] > 1.2 * max(peaks[0], peaks[2])
        intensity = np.abs(np.asarray(ms.focus(coords, coords, field, WAVELENGTH, focal))) ** 2
        centre = np.argmin(np.abs(coords))
        row = intensity[:, centre]
        assert row.argmax() == centre
        fine = np.linspace(-2, 2, 4001)
        width = np.ptp(fine[np.interp(fine, coords, row) >= row.max() / 2])
        airy = 0.51 * WAVELENGTH / np.sin(np.arctan(radius / focal))
        assert width == pytest.approx(airy, rel=0.05)

    def test_designed_lens_focuses(self):
        radius, focal = 6.0, 12.0
        library = ms.cell_library(_disc, np.linspace(0.04, 0.16, 13), (PITCH, PITCH),
                                  WAVELENGTH, N_GLASS**2, 1.0, (2, 2))  # fmt: skip
        sites = ms.square_sites(radius, PITCH)
        phase = np.asarray(ms.lens_phase(sites[:, 0], sites[:, 1], focal, WAVELENGTH))
        _, realised = ms.choose_cells(phase, library)
        assert np.sqrt(np.mean(np.angle(realised * np.exp(-1j * phase)) ** 2)) < 0.3
        coords, designed = ms.sample_grid(sites, realised, PITCH)
        _, ideal = ms.sample_grid(sites, np.abs(realised) * np.exp(1j * phase), PITCH)

        def peak(field):
            return float(jnp.abs(ms.focus(coords, coords, field, WAVELENGTH, focal)).max() ** 2)

        assert peak(designed) > 0.8 * peak(ideal)  # a Strehl ratio, of the phase alone

    def test_focal_gradient_in_per_site_twists(self):
        sites = ms.square_sites(2.0, 0.5)
        wavelength, focal = 1.55, 6.0

        def on_axis(alphas):
            t = ms.cell_library(_twisted_pair, alphas, (0.5, 0.5), wavelength, 1.55**2,
                                1.55**2, (0, 0))  # fmt: skip
            kept = jnp.einsum("i,kij,j->k", jnp.conj(LEFT), t, LEFT)
            coords, field = ms.sample_grid(sites, kept, 0.5, subsamples=2)
            centre = int(np.argmin(np.abs(coords)))
            return jnp.abs(ms.focus(coords, coords, field, wavelength, focal)[centre, centre]) ** 2

        alphas = jnp.asarray(np.random.default_rng(0).uniform(0, np.pi, len(sites)))
        grad = jax.grad(on_axis)(alphas)
        step = jnp.zeros(len(sites)).at[3].set(1e-6)
        fd = (on_axis(alphas + step) - on_axis(alphas - step)) / 2e-6
        assert float(grad[3]) == pytest.approx(float(fd), rel=1e-6)


def test_local_periodicity_against_a_gradient_supercell():
    """Fins etched into an x-cut LiNbO₃ film, turned by kπ/6 on a wafer with fixed axes.

    Each fin's form birefringence adds to or cancels the crystal's, so its
    conversion depends on its angle (0.31 to 0.63 here). Six of them in one
    supercell deflect converted light into one order; each fin's library entry
    (an infinite array of itself) predicts that order's power to within the
    local-periodicity error (0.37 against 0.40 here).
    """
    hog = pytest.importorskip("hyperbolic_optics.integrations.gdsfactory")
    n, wavelength, pitch, n_glass = 6, 1.55, 0.7, 1.444
    angles = np.pi * np.arange(n) / n
    ln = hog.Oriented("LiNbO3", hog.x_cut(0.0))
    outline = rcwa2d.rectangle((0.0, 0.0), (0.6, 0.24))

    def cell(polygons):
        layer = hog.StackLayer(1.6, ln, background=1.0, polygons=polygons)
        return hog.VerticalStack((layer,), incident=n_glass**2)

    def fin(angle):
        turned = hog.turned(cell((outline,)), angle, rotate_materials=False)
        return hog.cell_layers(turned, wavelength)

    jones = ms.cell_library(
        fin, angles, rcwa2d.Lattice.square(pitch), wavelength, n_glass**2, 1.0, (3, 3)
    )
    converted = np.einsum("i,kij,j->k", RIGHT.conj(), np.asarray(jones), LEFT)
    harmonic = np.mean(converted * np.exp(-2j * np.pi * np.arange(n) / n)) * np.sinc(1 / n)
    predicted = abs(harmonic) ** 2 / n_glass  # the staircase near field's first harmonic

    fins = tuple(
        fin(a)[0].shapes[0].vertices + jnp.array([(k - (n - 1) / 2) * pitch, 0.0])
        for k, a in enumerate(angles)
    )
    result = rcwa2d.solve_crossed_grating(
        hog.cell_layers(cell(fins), wavelength), (n * pitch, pitch), wavelength, 0.0,
        n_glass**2, 1.0, (18, 3),
    )  # fmt: skip

    def power(order):
        k = int(np.flatnonzero((result.orders == order).all(axis=1))[0])
        t, T = np.asarray(result.t[k]), np.asarray(result.T[k])
        return float(np.sum(abs(t @ LEFT) ** 2 * T[:, 0] / abs(t[:, 0]) ** 2))

    assert np.ptp(abs(converted) ** 2 / n_glass) > 0.2
    assert power((1, 0)) == pytest.approx(predicted, abs=0.05)
    assert power((-1, 0)) < 0.02


class TestLayout:
    LATTICE = ((0.4, 0.0), (0.0, 0.4))
    LAYERS = ((10, 0), (11, 0), (12, 0))

    @pytest.fixture(autouse=True)
    def _pdk(self):
        gf = pytest.importorskip("gdsfactory")
        gf.gpdk.PDK.activate()

    def _cells(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        fin = (np.array([[-0.125, -0.05], [0.125, -0.05], [0.125, 0.05], [-0.125, 0.05]]),)
        wide = (np.array([[-0.1, -0.1], [0.15, -0.1], [0.15, 0.08], [-0.1, 0.08]]),)
        ln = hog.Oriented("LiNbO3", hog.x_cut(0.0))
        twisted = hog.VerticalStack(
            (
                hog.StackLayer(0.2, ln, rotation_deg=(0.0, 0.0, 30.0), name="twisted LN"),
                hog.StackLayer(0.3, 2.4**2, background=1.0, polygons=fin, name="fin"),
            ),
            incident=1.0, exit=N_GLASS**2,
        )  # fmt: skip
        holes = hog.VerticalStack(
            (
                hog.StackLayer(0.1, ln, rotation_deg=(0.0, 0.0, -45.0)),
                hog.StackLayer(0.1, ln, rotation_deg=(0.0, 0.0, 45.0)),
                hog.StackLayer(0.25, 1.0, background=ln, polygons=wide, name="holes in LN"),
            ),
            incident=1.0, exit=N_GLASS**2,
        )  # fmt: skip
        film = hog.VerticalStack((hog.StackLayer(0.15, "Si3N4"),), incident=1.0, exit=N_GLASS**2)
        return twisted, holes, film

    def _jones(self, cell):
        from hyperbolic_optics.integrations import gdsfactory as hog

        result = rcwa2d.solve_crossed_grating(
            hog.cell_layers(cell, WAVELENGTH), rcwa2d.Lattice(*self.LATTICE), WAVELENGTH, 0.0,
            1.0, N_GLASS**2, (2, 2),
        )  # fmt: skip
        return np.asarray(result.t[int(np.flatnonzero((result.orders == 0).all(axis=1))[0])])

    @pytest.mark.parametrize("rotate_materials", [True, False])
    def test_round_trip_through_a_gds_file(self, tmp_path, rotate_materials):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        twisted, holes, film = self._cells()
        cells = [twisted, holes, film, twisted, holes, twisted]
        sites = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [0.0, 0.4], [0.4, 0.4], [0.8, 0.4]])
        rotations = np.radians([0.0, 0.0, 0.0, 30.0, -60.0, 90.0])
        component = hog.metasurface_component(
            sites, cells, self.LAYERS, self.LATTICE, rotations, rotate_materials=rotate_materials
        )
        assert len(component.insts) == 6 and component.info["n_cells"] == 3
        assert (
            hog.metasurface_component(
                sites,
                cells,
                self.LAYERS,
                self.LATTICE,
                rotations,
                rotate_materials=rotate_materials,
            )
            is component
        )
        other = hog.metasurface_component(
            sites,
            cells,
            self.LAYERS,
            self.LATTICE,
            rotations,
            rotate_materials=not rotate_materials,
        )
        assert other is not component
        assert other.name != component.name

        component.write_gds(tmp_path / "metasurface.gds")
        read_sites, read_cells, lattice = hog.read_metasurface(
            gf.import_gds(tmp_path / "metasurface.gds")
        )
        assert lattice == self.LATTICE
        order = np.lexsort(np.round(read_sites, 6).T[::-1])
        expected = np.lexsort(np.round(sites, 6).T[::-1])
        np.testing.assert_allclose(read_sites[order], sites[expected], atol=1e-9)
        for k_read, k_design in zip(order, expected, strict=True):
            design = hog.turned(
                cells[k_design], rotations[k_design], rotate_materials=rotate_materials
            )
            back = read_cells[k_read]
            assert [layer.name for layer in back.layers] == [layer.name for layer in design.layers]
            np.testing.assert_allclose(self._jones(back), self._jones(design), atol=1e-9)

    def test_fixed_wafer_and_placed_crystal_frames(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        shape = (rcwa2d.rectangle((0.0, 0.0), (0.23, 0.12)),)
        layer = hog.StackLayer(
            0.2,
            hog.Oriented("LiNbO3", (17.0, 39.0, 12.0)),
            background=hog.Oriented("LiNbO3", (21.0, 8.0, 34.0)),
            rotation_deg=(11.0, 25.0, 16.0),
            polygons=shape,
        )
        cell = hog.VerticalStack((layer,), exit=N_GLASS**2)
        theta = 0.41
        rotation = np.asarray(bcm.rotation_matrix(0.0, 0.0, theta))
        for rotate_materials in (False, True):

            def turn(a):
                return hog.turned(cell, a, rotate_materials=rotate_materials)

            after = turn(theta).layers[0]
            np.testing.assert_allclose(after.polygons[0], shape[0] @ rotation[:2, :2].T)
            for name in ("material", "background"):
                before = hog.material_tensor(getattr(layer, name), WAVELENGTH, layer.rotation_deg)
                expected = rotation @ before @ rotation.T if rotate_materials else before
                actual = hog.material_tensor(getattr(after, name), WAVELENGTH, after.rotation_deg)
                np.testing.assert_allclose(actual, expected, atol=1e-12)

            @jax.jit
            def transmission(a):
                return (
                    jnp.abs(
                        hog.cell_sax_model(turn(a), self.LATTICE, (1, 1))(wl=WAVELENGTH)[
                            ("top@p", "bottom@s")
                        ]
                    )
                    ** 2
                )

            grad = jax.grad(transmission)(theta)
            fd = (transmission(theta + 1e-5) - transmission(theta - 1e-5)) / 2e-5
            np.testing.assert_allclose(grad, fd, rtol=2e-4, atol=1e-7)
        fixed, placed = (
            self._jones(hog.turned(cell, theta, rotate_materials=v)) for v in (False, True)
        )
        assert np.max(abs(fixed - placed)) > 1e-3

    def test_what_a_gds_cannot_hold_is_refused(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        twisted, _, _ = self._cells()
        with pytest.raises(ValueError, match="sites"):
            hog.metasurface_component(np.zeros((3, 2)), [twisted] * 2, self.LAYERS, self.LATTICE)
        dispersive = hog.VerticalStack((hog.StackLayer(0.1, lambda wl: 2.0 + 0 * wl),))
        with pytest.raises(TypeError, match="name"):
            hog.metasurface_component(np.zeros((1, 2)), [dispersive], self.LAYERS, self.LATTICE)
        with pytest.raises(ValueError, match="levels"):
            hog.metasurface_component(np.zeros((1, 2)), [twisted], self.LAYERS[:1], self.LATTICE)

    def test_hyperbolic_wafer_round_trip(self, tmp_path):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        wavelength = 1e4 / 1450.0
        material = hog.Oriented("hBN", (0.0, 40.0, 25.0))
        eigenvalues = np.linalg.eigvalsh(np.real(hog.material_tensor(material, wavelength)))
        assert eigenvalues.min() < 0 < eigenvalues.max()
        cell = hog.VerticalStack(
            (
                hog.StackLayer(
                    0.3,
                    material,
                    background=1.0,
                    polygons=(rcwa2d.rectangle((0.0, 0.0), (0.24, 0.12)),),
                ),
            ),
            exit=2.25,
        )
        angle = 0.43
        layout = hog.metasurface_component(
            [[0.0, 0.0]], [cell], self.LAYERS, self.LATTICE, [angle], rotate_materials=False
        )
        layout.write_gds(tmp_path / "hbn.gds")
        back = hog.read_metasurface(gf.import_gds(tmp_path / "hbn.gds"))[1][0]
        expected = hog.turned(cell, angle, rotate_materials=False)
        a = hog.cell_sax_model(back, self.LATTICE, (2, 2))(wl=wavelength)
        b = hog.cell_sax_model(expected, self.LATTICE, (2, 2))(wl=wavelength)
        assert back.layers[0].material == material
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-8)

    def test_legacy_gds_without_rotation_flag_turns_crystals(self, tmp_path):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        child = gf.Component()
        child.add_polygon([(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)], layer=(10, 0))
        child.info.update(
            dict(
                lattice_um=self.LATTICE,
                incident=1.0,
                exit=2.25,
                levels=[
                    dict(
                        thickness_um=0.2,
                        material="LiNbO3",
                        background=None,
                        rotation_deg=[11.0, 25.0, 16.0],
                        name="film",
                        layer=[10, 0],
                    )
                ],
            )
        )
        layout = gf.Component()
        layout.add_ref(child).drotate(23.0)
        layout.write_gds(tmp_path / "legacy.gds")
        back = hog.read_metasurface(gf.import_gds(tmp_path / "legacy.gds"))[1][0]
        np.testing.assert_allclose(back.layers[0].rotation_deg, [11.0, 25.0, 39.0])

    def test_empty_pattern_and_complex_cut_survive_gds(self, tmp_path):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        material = hog.Oriented(3 + 0.1j, (12, 23, 34))
        cell = hog.VerticalStack((hog.StackLayer(0.2, material, background=1.0, polygons=()),))
        layout = hog.metasurface_component([[0, 0]], [cell], self.LAYERS, self.LATTICE)
        layout.write_gds(tmp_path / "empty.gds")
        back = hog.read_metasurface(gf.import_gds(tmp_path / "empty.gds"))[1][0]
        assert back.layers[0].polygons == ()
        assert back.layers[0].material == material
        np.testing.assert_allclose(self._jones(back), self._jones(cell), atol=1e-12)

    def test_tensor_material_survives_gds(self, tmp_path):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        tensor = np.array([[3.0, 0.2j, 0.0], [-0.2j, 3.5, 0.1], [0.0, 0.1, 4.0 + 0.05j]])
        square = (np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]),)
        cell = hog.VerticalStack((hog.StackLayer(0.2, tensor, background=1.0, polygons=square),))
        layout = hog.metasurface_component([[0, 0]], [cell], self.LAYERS, self.LATTICE)
        layout.write_gds(tmp_path / "tensor.gds")
        back = hog.read_metasurface(gf.import_gds(tmp_path / "tensor.gds"))[1][0]
        np.testing.assert_array_equal(back.layers[0].material, tensor)
        np.testing.assert_allclose(self._jones(back), self._jones(cell), atol=1e-12)

    def test_shapes_overlapping_themselves_when_placed_are_refused(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        # 390 × 120 nm fits the 400 nm cell unturned; its half-diagonal (204 nm) does not
        fin = (np.asarray(rcwa2d.rectangle((0.0, 0.0), (0.39, 0.12))),)
        cell = hog.VerticalStack((hog.StackLayer(0.3, 5.76, background=1.0, polygons=fin),))
        hog.metasurface_component([[0, 0]], [cell], self.LAYERS, self.LATTICE)
        with pytest.raises(ValueError, match="periodic images"):
            hog.metasurface_component([[0, 0]], [cell], self.LAYERS, self.LATTICE, [0.3])
        halves = (fin[0] * 0.5, fin[0] * 0.5 + 0.05)
        pair = hog.VerticalStack((hog.StackLayer(0.3, 5.76, background=1.0, polygons=halves),))
        with pytest.raises(ValueError, match="overlaps itself"):
            hog.metasurface_component([[0, 0]], [pair], self.LAYERS, self.LATTICE)

    def test_ambiguous_layers_and_unsupported_references_are_refused(self):
        import gdsfactory as gf
        import kfactory as kf

        from hyperbolic_optics.integrations import gdsfactory as hog

        cell = self._cells()[0]
        with pytest.raises(ValueError, match="distinct GDS layer"):
            hog.metasurface_component([[0, 0]], [cell], [(10, 0)] * 2, self.LATTICE)
        layout = hog.metasurface_component([[0, 0]], [cell], self.LAYERS, self.LATTICE)
        nested = gf.Component()
        nested.add_ref(layout)
        with pytest.raises(ValueError, match="direct cell references"):
            hog.read_metasurface(nested)
        for magnitude, mirror in ((2, False), (1, True)):
            transformed = gf.Component()
            transformed.create_inst(
                next(iter(layout.insts)).cell, kf.kdb.DCplxTrans(magnitude, 0, mirror, 0, 0)
            )
            with pytest.raises(ValueError, match="Mirrored or magnified"):
                hog.read_metasurface(transformed)


class TestSaxModel:
    @pytest.mark.parametrize("kx", [-1.6, -0.6, 0.0, 0.6, 1.2, 1.6])
    def test_anisotropic_cell_has_planar_azimuth_and_evanescent_conventions(self, kx):
        from hyperbolic_optics.integrations import gdsfactory as hog

        cell = hog.VerticalStack(
            (hog.StackLayer(0.23, np.diag([3.0, 4.0, 5.0]), rotation_deg=(20, 35, 17)),),
            incident=2.25,
            exit=1.0,
        )
        args = dict(wl=1.55, kx=kx, azimuth=0.4)
        planar = hog.stack_sax_model(cell)(**args)
        crossed = hog.cell_sax_model(cell, TestLayout.LATTICE, (0, 0))(**args)
        for key in planar:
            np.testing.assert_allclose(crossed[key], planar[key], atol=2e-12)

    def test_wavelength_jit_and_gradient_include_half_space_dispersion(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        cell = hog.VerticalStack((hog.StackLayer(0.2, "Si3N4"),), exit="SiO2")
        model = hog.cell_sax_model(cell, TestLayout.LATTICE, (0, 0))

        def power(wl):
            return abs(model(wl=wl, kx=0.3)["top@p", "bottom@p"]) ** 2

        compiled = jax.jit(power)
        np.testing.assert_allclose(compiled(1.55), power(1.55), atol=1e-12)
        fd = (power(1.550001) - power(1.549999)) / 2e-6
        np.testing.assert_allclose(jax.grad(compiled)(1.55), fd, rtol=1e-6)

    def test_jit_does_not_silently_replace_invalid_half_space(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        cell = hog.VerticalStack((), exit=np.diag([2.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="isotropic and lossless"):
            cell.half_space_permittivity("exit", 1.55)
        evaluated = jax.jit(lambda wl: cell.half_space_permittivity("exit", wl))(1.55)
        assert jnp.isnan(evaluated)

    def test_zeroth_order_smatrix(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        twisted, _, film = TestLayout()._cells()
        sd = hog.cell_sax_model(twisted, TestLayout.LATTICE, (2, 2))(wl=WAVELENGTH, kx=0.2)
        direct = rcwa2d.crossed_grating_smatrix(
            hog.cell_layers(twisted, WAVELENGTH), rcwa2d.Lattice(*TestLayout.LATTICE), WAVELENGTH,
            np.arcsin(0.2), 1.0, N_GLASS**2, (2, 2),
        ).s  # fmt: skip
        for i_in, port_in in enumerate(bcm.PORTS):
            for i_out, port_out in enumerate(bcm.PORTS):
                assert complex(sd[port_in, port_out]) == pytest.approx(complex(direct[i_out, i_in]))

        planar = hog.stack_sax_model(film)(wl=WAVELENGTH, kx=0.2, azimuth=0.4)
        cell = hog.cell_sax_model(film, TestLayout.LATTICE, (1, 1))(wl=WAVELENGTH, kx=0.2,
                                                                     azimuth=0.4)  # fmt: skip
        for key, value in planar.items():
            assert complex(cell[key]) == pytest.approx(complex(value), abs=1e-12)

    def test_a_cell_in_a_sax_circuit_is_the_whole_structure(self):
        """cell | 4 µm glass | Si film: connected in SAX, and solved as one 2D-periodic stack.
        The cell's higher orders are evanescent in glass and decay (e⁻²⁵) across the spacer."""
        sax = pytest.importorskip("sax")
        from hyperbolic_optics.integrations import gdsfactory as hog

        twisted, _, _ = TestLayout()._cells()
        below = hog.VerticalStack(
            (hog.StackLayer(4.0, N_GLASS**2), hog.StackLayer(0.1, 12.0)),
            incident=N_GLASS**2, exit=1.0,
        )  # fmt: skip
        circuit, _ = sax.circuit(
            {
                "instances": {"cell": "cell", "below": "below"},
                "connections": {"cell,bottom": "below,top"},
                "ports": {"top": "cell,top", "bottom": "below,bottom"},
            },
            {
                "cell": hog.cell_sax_model(twisted, TestLayout.LATTICE, (2, 2)),
                "below": hog.stack_sax_model(below),
            },  # fmt: skip
            backend="filipsson_gunnar",
        )
        sd = circuit(wl=WAVELENGTH, kx=0.0)
        whole = hog.VerticalStack(twisted.layers + below.layers, incident=1.0, exit=1.0)
        s = rcwa2d.crossed_grating_smatrix(
            hog.cell_layers(whole, WAVELENGTH), rcwa2d.Lattice(*TestLayout.LATTICE), WAVELENGTH,
            0.0, 1.0, 1.0, (2, 2),
        ).s  # fmt: skip
        for i_in, port_in in enumerate(bcm.PORTS):
            for i_out, port_out in enumerate(bcm.PORTS):
                assert complex(sd[port_in, port_out]) == pytest.approx(
                    complex(s[i_out, i_in]), abs=1e-9
                )
