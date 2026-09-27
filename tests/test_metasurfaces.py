"""Tests for metasurface design (``hyperbolic_optics.metasurfaces``) and its layout.

- the cell library of a uniform film is the film's Fresnel transmission,
  ``[out, in]`` and all;
- a turned half-wave cell converts circular polarization with the
  Pancharatnam–Berry phase ``2θ``;
- a cell of two half-wave layers twisted by α is a rotator by 2α: circular
  light keeps its handedness and gains the phase −2α;
- the cell choice realises the target phase from the nearest library entry;
- the hyperbolic lens phase focuses at ``f``, at the diffraction limit;
- a TiO₂-pillar metalens designed from its RCWA library focuses at ``f``;
- the layout places one reference per site, at the site, turned as asked.

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


def _film_t(n, thickness, n_in, n_out, wavelength):
    """Normal-incidence amplitude transmission of a film (Airy)."""
    k = 2 * np.pi * n * thickness / wavelength
    t1, t2 = 2 * n_in / (n_in + n), 2 * n / (n + n_out)
    r1, r2 = (n_in - n) / (n_in + n), (n - n_out) / (n + n_out)
    return t1 * t2 * np.exp(1j * k) / (1 + r1 * r2 * np.exp(2j * k))


def _focal_plane(sites, transmission, distance, sub=3):
    """|field|² ``distance`` after a pixelated metasurface, and its coordinates."""
    n = int(np.max(np.abs(np.round(sites / PITCH))))
    size = (2 * n + 1) * sub
    pad = 3 * size
    grid = np.zeros((pad, pad), complex)
    off = (pad - size) // 2
    for (x, y), value in zip(sites, transmission):
        i, j = int(round(x / PITCH)) + n, int(round(y / PITCH)) + n
        grid[off + i * sub : off + (i + 1) * sub, off + j * sub : off + (j + 1) * sub] = value
    coords = (np.arange(pad) - (off + n * sub + (sub - 1) / 2)) * PITCH / sub
    return coords, np.abs(ms.focus(coords, coords, grid, WAVELENGTH, distance)) ** 2


class TestLibrary:
    def test_uniform_film_is_fresnel(self):
        def cell(n):
            return [rcwa2d.uniform(n**2 * jnp.eye(3, dtype=complex), 0.3)]

        indices = np.array([1.5, 2.0, 2.4])
        t = ms.cell_library(
            cell, indices, rcwa2d.Lattice.square(PITCH), WAVELENGTH, N_GLASS**2, 1.0, (1, 1)
        )
        expected = _film_t(indices, 0.3, N_GLASS, 1.0, WAVELENGTH)
        assert t.shape == (3, 2, 2)
        np.testing.assert_allclose(t[:, 0, 0], expected, atol=1e-10)
        np.testing.assert_allclose(t[:, 1, 1], expected, atol=1e-10)
        np.testing.assert_allclose(t[:, 0, 1], 0, atol=1e-12)

    def test_pillar_library_spans_the_circle(self):
        radii = np.linspace(0.04, 0.16, 9)
        t = ms.cell_library(
            ms.disc_cell(2.4**2, 0.6, PITCH), radii, rcwa2d.Lattice.square(PITCH),
            WAVELENGTH, N_GLASS**2, 1.0, (2, 2),
        )  # fmt: skip
        phase = np.unwrap(np.angle(t[:, 0, 0]))
        assert np.ptp(phase) > 1.6 * np.pi
        power = np.abs(t[:, 0, 0]) ** 2 / N_GLASS  # T = |t|²·n_out/n_in
        assert np.all(power <= 1 + 1e-9) and np.all(power > 0.8)
        # a disc is symmetric: no conversion, equal diagonal
        np.testing.assert_allclose(t[:, 0, 0], t[:, 1, 1], atol=1e-6)


class TestDesign:
    def test_geometric_phase(self):
        half_wave = np.diag([1.0, -1.0]).astype(complex)
        left, right = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)
        for theta in (0.0, 0.3, 1.1, 2.5):
            jones = ms.geometric_phase_jones(half_wave, theta)
            converted = np.conj(right) @ jones @ left
            np.testing.assert_allclose(converted, np.exp(2j * theta), atol=1e-12)
            assert abs(np.conj(left) @ jones @ left) < 1e-12

    def test_twisted_half_wave_pair_is_a_rotator(self):
        # each layer a whole number of half-waves for both indices (8 and 7.5
        # wavelengths): no reflections, so the Jones-calculus result is exact
        wavelength, n_o, n_e = 1.55, 1.50, 1.60
        thickness = wavelength / (2 * (n_e - n_o))
        eps = jnp.diag(jnp.array([n_e**2, n_o**2, n_o**2], dtype=complex))

        def twisted(alpha):
            turn = [bcm.rotation_matrix(0.0, 0.0, k * alpha) for k in range(2)]
            return [rcwa2d.uniform(bcm.rotate_tensor(eps, r), thickness) for r in turn]

        alphas = np.linspace(0.1, 1.5, 5)
        t = ms.cell_library(twisted, alphas, (0.5, 0.5), wavelength, 1.55**2, 1.55**2, (0, 0))
        left, right = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)
        kept = np.einsum("i,kij,j->k", np.conj(left), t, left)
        converted = np.einsum("i,kij,j->k", np.conj(right), t, left)
        untwisted = ms.cell_library(twisted, [0.0], (0.5, 0.5), wavelength, 1.55**2,
                                    1.55**2, (0, 0))[0]  # fmt: skip
        np.testing.assert_allclose(kept, (np.conj(left) @ untwisted @ left) * np.exp(-2j * alphas),
                                   atol=1e-9)  # fmt: skip
        np.testing.assert_allclose(converted, 0, atol=1e-9)

    def test_nearest_entry(self):
        library_phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        library = np.exp(1j * library_phases)[:, None, None] * np.eye(2)
        target = np.array([[0.1, 3.0], [6.2, -1.0]])
        indices, realised = ms.choose_cells(target, library)
        assert indices.shape == target.shape
        wrapped = np.angle(np.exp(1j * (library_phases[indices] - target)))
        assert np.all(np.abs(wrapped) <= np.pi / 16 + 1e-12)
        np.testing.assert_allclose(realised, np.exp(1j * library_phases[indices]))

    def test_ideal_lens_focuses_at_the_diffraction_limit(self):
        radius, focal = 8.0, 16.0
        sites = ms.square_sites(radius, PITCH)
        assert np.all(np.hypot(*sites.T) <= radius)
        phase = ms.lens_phase(sites[:, 0], sites[:, 1], focal, WAVELENGTH)
        peaks = []
        for distance in (focal - 3, focal, focal + 3):
            coords, intensity = _focal_plane(sites, np.exp(1j * phase), distance)
            peaks.append(intensity.max())
        assert peaks[1] > 1.5 * max(peaks[0], peaks[2])
        coords, intensity = _focal_plane(sites, np.exp(1j * phase), focal)
        centre = np.argmin(np.abs(coords))
        row = intensity[:, centre]
        assert row.argmax() == centre
        width = np.ptp(coords[row >= row.max() / 2])
        airy = 0.51 * WAVELENGTH / np.sin(np.arctan(radius / focal))
        assert abs(width - airy) < 0.15 * airy


class TestMetalens:
    def test_designed_lens_focuses(self):
        radius, focal = 6.0, 12.0
        radii = np.linspace(0.04, 0.16, 13)
        library = ms.cell_library(
            ms.disc_cell(2.4**2, 0.6, PITCH), radii, rcwa2d.Lattice.square(PITCH),
            WAVELENGTH, N_GLASS**2, 1.0, (2, 2),
        )  # fmt: skip
        sites = ms.square_sites(radius, PITCH)
        phase = ms.lens_phase(sites[:, 0], sites[:, 1], focal, WAVELENGTH)
        indices, realised = ms.choose_cells(phase, library)
        error = np.angle(realised * np.exp(-1j * phase))
        assert np.sqrt(np.mean(error**2)) < 0.3

        _, designed = _focal_plane(sites, realised, focal)
        _, ideal = _focal_plane(sites, np.abs(realised) * np.exp(1j * phase), focal)
        assert designed.max() > 0.8 * ideal.max()  # a Strehl ratio, of the phase alone


class TestLayout:
    @pytest.fixture(autouse=True)
    def _pdk(self):
        gf = pytest.importorskip("gdsfactory")
        gf.gpdk.PDK.activate()

    def test_one_reference_per_site(self, tmp_path):
        import gdsfactory as gf

        from hyperbolic_optics.integrations import gdsfactory as hog

        sites = ms.square_sites(2.0, PITCH)
        indices = np.arange(len(sites)) % 3
        rotations = np.linspace(0, np.pi, len(sites))
        polygons = [
            [np.asarray(rcwa2d.rectangle((0.0, 0.0), (0.2, 0.08)))],
            [np.asarray(rcwa2d.rectangle((0.0, 0.0), (0.24, 0.08)))],
            [np.asarray(rcwa2d.rectangle((0.0, 0.0), (0.28, 0.08)))],
            [np.asarray(rcwa2d.rectangle((0.0, 0.0), (0.30, 0.08)))],  # unused
        ]
        component = hog.metasurface_component(sites, indices, polygons, rotations_rad=rotations)
        assert len(component.insts) == len(sites)
        assert component.info["entries_used"] == (0, 1, 2)
        again = hog.metasurface_component(sites, indices, polygons, rotations_rad=rotations)
        assert again.name == component.name

        path = tmp_path / "lens.gds"
        component.write_gds(path)
        read = gf.import_gds(path)
        read.flatten()
        polygons_read = read.get_polygons_points()
        points = next(iter(polygons_read.values()))
        assert len(points) == len(sites)
        centres = np.array([np.mean(p, axis=0) for p in points])
        order = np.lexsort(np.round(centres, 3).T[::-1])
        expected = sites[np.lexsort(np.round(sites, 3).T[::-1])]
        np.testing.assert_allclose(centres[order], expected, atol=2e-3)
        # each rectangle's long side along its rotation
        lengths = np.array([0.2, 0.24, 0.28])[indices]
        for (x, y), angle, length in zip(sites[:5], rotations[:5], lengths[:5]):
            k = np.argmin(np.hypot(*(centres - (x, y)).T))
            p = np.asarray(points[k])
            spans = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)
            edge = (p - np.roll(p, 1, axis=0))[np.argmax(spans)]
            assert abs(max(spans) - length) < 3e-3
            assert abs(np.sin(np.arctan2(edge[1], edge[0]) - angle)) < 2e-2

    def test_mismatched_sites(self):
        from hyperbolic_optics.integrations import gdsfactory as hog

        with pytest.raises(ValueError, match="sites"):
            hog.metasurface_component(np.zeros((3, 2)), [0, 0], hog.disc_polygons([0.1]))
