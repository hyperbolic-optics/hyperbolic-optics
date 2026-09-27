"""Tests for the differentiable crossed-grating RCWA (``hyperbolic_optics.rcwa2d``).

The anchors, none of which share code with the 2D solver's own results:

- polygon Fourier transforms against direct quadrature;
- uniform layers against the planar solver, and stripes -- along x and along
  y -- against the 1D grating solver, for full tensors at conical incidence,
  under both factorisations;
- energy conservation (to rounding, at every truncation) and passivity, and
  the symmetries of a square pillar;
- an in-plane-anisotropic disc against FMMAX (an independent JAX RCWA code);
- gradients with respect to vertices, radius, thickness and wavelength against
  finite differences.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, rcwa, rcwa2d  # noqa: E402


def _uniaxial(eps_o, eps_e, tilt, azimuth):
    axis = jnp.array(
        [jnp.cos(tilt) * jnp.cos(azimuth), jnp.cos(tilt) * jnp.sin(azimuth), jnp.sin(tilt)]
    )
    return (eps_o * jnp.eye(3) + (eps_e - eps_o) * jnp.outer(axis, axis)).astype(complex)


EYE = jnp.eye(3, dtype=complex)
LN_TILTED = _uniaxial(5.46, 5.06, 0.7, 0.4)  # every off-diagonal component nonzero
SILICON = 3.5**2 * EYE
FACTORIZATIONS = ("normal", "laurent")


def _zeroth(result):
    return int(np.flatnonzero((np.asarray(result.orders) == 0).all(axis=1))[0])


def _total(result):
    """Summed efficiency over orders and outgoing polarizations, per input."""
    return np.asarray(result.R.sum(axis=(-3, -2)) + result.T.sum(axis=(-3, -2)))


def _pillar(eps, size=(0.3, 0.3), thickness=0.25, period=0.6):
    centre = (period / 2, period / 2)
    shape = rcwa2d.Shape(rcwa2d.rectangle(centre, size), eps)
    return rcwa2d.PatternedLayer(thickness, EYE, (shape,))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestPolygonFourier:
    def test_triangle_against_quadrature(self):
        triangle = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.8]])
        gx = jnp.array([0.0, 3.0, -7.0, 12.0])
        gy = jnp.array([0.0, 5.0, 2.0, -9.0])
        exact = np.asarray(rcwa2d.polygon_fourier(triangle, gx, gy))

        n = 2000
        xs = (np.arange(n) + 0.5) / n
        x, y = np.meshgrid(xs, xs, indexing="ij")
        v = np.asarray(triangle)
        sides = [
            (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            for a, b in zip(v, np.roll(v, -1, axis=0), strict=True)
        ]
        inside = np.all([s > 0 for s in sides], axis=0) | np.all([s < 0 for s in sides], axis=0)
        quadrature = [
            np.sum(inside * np.exp(-1j * (a * x + b * y))) / n**2
            for a, b in zip(np.asarray(gx), np.asarray(gy), strict=True)
        ]
        np.testing.assert_allclose(exact, quadrature, atol=5e-5)
        assert exact[0].real == pytest.approx(0.17)  # the area, exactly

    def test_winding_does_not_matter(self):
        square = rcwa2d.rectangle((0.3, 0.4), (0.2, 0.5))
        gx, gy = jnp.array([0.0, 4.0, -9.0]), jnp.array([0.0, 1.0, 6.0])
        np.testing.assert_allclose(
            rcwa2d.polygon_fourier(square, gx, gy),
            rcwa2d.polygon_fourier(square[::-1], gx, gy),
            atol=1e-14,
        )

    def test_rectangle_is_a_product_of_sincs(self):
        size = (0.2, 0.5)
        gx, gy = jnp.array([0.0, 4.0, -9.0, 30.0]), jnp.array([0.0, 1.0, 6.0, -2.0])
        centre = (0.3, 0.4)
        got = rcwa2d.polygon_fourier(rcwa2d.rectangle(centre, size), gx, gy)
        want = (
            size[0]
            * size[1]
            * np.sinc(np.asarray(gx) * size[0] / (2 * np.pi))
            * np.sinc(np.asarray(gy) * size[1] / (2 * np.pi))
            * np.exp(-1j * (np.asarray(gx) * centre[0] + np.asarray(gy) * centre[1]))
        )
        np.testing.assert_allclose(got, want, atol=1e-14)

    def test_ellipse_has_the_ellipse_area(self):
        vertices = rcwa2d.ellipse((0.0, 0.0), (0.2, 0.1), n_vertices=24, rotation_rad=0.3)
        area = rcwa2d.polygon_fourier(vertices, jnp.zeros(1), jnp.zeros(1))[0].real
        assert area == pytest.approx(np.pi * 0.2 * 0.1, rel=1e-12)


# ---------------------------------------------------------------------------
# Reductions to the planar and 1D solvers
# ---------------------------------------------------------------------------


class TestReductions:
    @pytest.mark.parametrize("factorization", FACTORIZATIONS)
    def test_uniform_layers_are_the_planar_solver(self, factorization):
        layers = [rcwa2d.uniform(LN_TILTED, 0.4), rcwa2d.uniform(2.1 * EYE, 0.2)]
        result = rcwa2d.solve_crossed_grating(
            layers, (0.5, 0.6), 0.8, 0.4, 1.0, 2.25, (2, 2), azimuth_rad=0.3,
            factorization=factorization,
        )  # fmt: skip
        rotation = bcm.rotation_matrix(0.0, 0.0, -0.3)
        tensors = bcm.rotate_tensor(jnp.stack([LN_TILTED, 2.1 * EYE]), rotation)
        planar = bcm.solve_stack(tensors, jnp.array([0.4, 0.2]), 0.8, 0.4, 1.0, 2.25)
        z = _zeroth(result)
        np.testing.assert_allclose(result.r[z], planar.r, atol=1e-12)
        np.testing.assert_allclose(result.t[z], planar.t, atol=1e-12)
        others = np.delete(np.asarray(result.R + result.T), z, axis=0)
        assert np.abs(others).max() < 1e-20

    @pytest.mark.parametrize(
        ("factorization", "rule_1d"), [("normal", "li"), ("laurent", "laurent")]
    )
    @pytest.mark.parametrize("ridge", [SILICON, LN_TILTED], ids=["isotropic", "tensor"])
    def test_x_stripes_are_the_1d_solver(self, factorization, rule_1d, ridge):
        """A rectangle spanning the cell in y is a line grating; conical incidence."""
        period, width, offset = 0.9, 0.36, 0.02
        shape = rcwa2d.Shape(rcwa2d.rectangle((offset + width / 2, 0.3), (width, 0.6)), ridge)
        stripes = rcwa2d.PatternedLayer(0.3, EYE, (shape,))
        crossed = rcwa2d.solve_crossed_grating(
            [stripes], (period, 0.6), 1.0, 0.35, 1.0, 2.1, (8, 1), azimuth_rad=0.4,
            factorization=factorization,
        )  # fmt: skip
        line = rcwa.solve_grating(
            [rcwa.lamellar_layer(ridge, EYE, width / period, 0.3, offset=offset / period)],
            period, 1.0, 0.35, 1.0, 2.1, 8, azimuth_rad=0.4, factorization=rule_1d,
        )  # fmt: skip
        row = np.asarray(crossed.orders)[:, 1] == 0
        np.testing.assert_allclose(np.asarray(crossed.r)[row], line.r, atol=1e-12)
        np.testing.assert_allclose(np.asarray(crossed.t)[row], line.t, atol=1e-12)
        assert np.asarray(crossed.R + crossed.T)[~row].max() < 1e-20

    @pytest.mark.parametrize(
        ("factorization", "rule_1d"), [("normal", "li"), ("laurent", "laurent")]
    )
    def test_y_stripes_are_the_1d_solver_turned(self, factorization, rule_1d):
        """Stripes along x: the 1D grating seen with the lab turned by 90°."""
        shape = rcwa2d.Shape(rcwa2d.rectangle((0.3, 0.2), (0.6, 0.36)), SILICON)
        stripes = rcwa2d.PatternedLayer(0.3, EYE, (shape,))
        crossed = rcwa2d.solve_crossed_grating(
            [stripes], (0.6, 0.9), 1.0, 0.35, 1.0, 2.1, (1, 8), azimuth_rad=0.4 + np.pi / 2,
            factorization=factorization,
        )  # fmt: skip
        line = rcwa.solve_grating(
            [rcwa.lamellar_layer(SILICON, EYE, 0.4, 0.3)], 0.9, 1.0, 0.35, 1.0, 2.1, 8,
            azimuth_rad=0.4, factorization=rule_1d,
        )  # fmt: skip
        column = np.asarray(crossed.orders)[:, 0] == 0
        np.testing.assert_allclose(
            np.asarray(crossed.R)[column].sum(-2), np.asarray(line.R).sum(-2), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(crossed.T)[column].sum(-2), np.asarray(line.T).sum(-2), atol=1e-12
        )


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


class TestPhysics:
    @pytest.mark.parametrize("factorization", FACTORIZATIONS)
    def test_lossless_conserves_energy(self, factorization):
        disc = rcwa2d.Shape(rcwa2d.ellipse((0.3, 0.25), (0.17, 0.12), rotation_rad=0.5), LN_TILTED)
        layers = [
            rcwa2d.PatternedLayer(0.25, EYE, (disc,)),
            _pillar(SILICON, size=(0.2, 0.35), thickness=0.1),
        ]
        result = rcwa2d.solve_crossed_grating(
            layers, (0.6, 0.6), 0.7, 0.5, 1.0, 2.1, (4, 4), azimuth_rad=0.3,
            factorization=factorization,
        )  # fmt: skip
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-11)

    def test_lossy_is_passive(self):
        metal = (-8.0 + 1.5j) * EYE
        hyperbolic = jnp.diag(jnp.array([-3.0 + 0.3j, 4.0 + 0.1j, 4.0 + 0.1j]))
        for eps in (metal, hyperbolic):
            result = rcwa2d.solve_crossed_grating(
                [_pillar(eps)], (0.6, 0.6), 1.0, 0.4, 1.0, 2.1, (3, 3), azimuth_rad=0.2
            )
            total = _total(result)
            assert np.all(total < 1.0) and np.all(total > 0.0)

    @pytest.mark.parametrize("factorization", FACTORIZATIONS)
    def test_square_pillar_is_polarization_blind_at_normal_incidence(self, factorization):
        result = rcwa2d.solve_crossed_grating(
            [_pillar(SILICON)], (0.6, 0.6), 1.0, 0.0, 1.0, 2.1, (3, 3),
            factorization=factorization,
        )  # fmt: skip
        z = _zeroth(result)
        assert result.R[z, 0, 0] == pytest.approx(float(result.R[z, 1, 1]), abs=1e-12)
        assert abs(float(result.R[z, 1, 0])) < 1e-24

    @pytest.mark.parametrize("factorization", FACTORIZATIONS)
    def test_square_pillar_has_fourfold_symmetry(self, factorization):
        """Incidence from azimuth φ and φ + 90° give the same zeroth-order response."""
        kwargs = {"factorization": factorization}
        a = rcwa2d.solve_crossed_grating(
            [_pillar(SILICON)], (0.6, 0.6), 1.0, 0.4, 1.0, 2.1, (3, 3), azimuth_rad=0.3, **kwargs
        )
        b = rcwa2d.solve_crossed_grating(
            [_pillar(SILICON)], (0.6, 0.6), 1.0, 0.4, 1.0, 2.1, (3, 3),
            azimuth_rad=0.3 + np.pi / 2, **kwargs,
        )  # fmt: skip
        np.testing.assert_allclose(a.R[_zeroth(a)], b.R[_zeroth(b)], atol=1e-12)
        np.testing.assert_allclose(a.T.sum(axis=(-3, -2)), b.T.sum(axis=(-3, -2)), atol=1e-12)

    def test_in_plane_anisotropy_matches_fmmax(self):
        """A disc with its optic axis in the plane at 30° from x, 30° incidence.

        FMMAX (JONES_DIRECT, 1605 terms, 1200² raster): R_pp 0.005429,
        R_ss 0.023615, R_ps 0.000109, R_sp 0.000108. FMMAX supports only
        ε_xy among the off-diagonal terms, so tilted axes are checked
        against the 1D solver and the planar one instead.
        """
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        rotation = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        eps = rotation @ jnp.diag(jnp.array([7.5, 6.25, 6.25])).astype(complex) @ rotation.T
        disc = rcwa2d.Shape(rcwa2d.ellipse((0.3, 0.3), (0.17, 0.17)), eps)
        result = rcwa2d.solve_crossed_grating(
            [rcwa2d.PatternedLayer(0.25, EYE, (disc,))], (0.6, 0.6), 1.0, np.pi / 6, 1.0, 2.1,
            (5, 5),
        )  # fmt: skip
        R = np.asarray(result.R)[_zeroth(result)]
        np.testing.assert_allclose(R[0, 0], 0.005429, atol=1.5e-4)
        np.testing.assert_allclose(R[1, 1], 0.023615, atol=1.5e-4)
        np.testing.assert_allclose(R[1, 0], 0.000109, atol=2e-5)
        np.testing.assert_allclose(R[0, 1], 0.000108, atol=2e-5)

    def test_normal_vector_converges_faster_than_laurent(self):
        """Silicon pillar; FMMAX's normal-vector formulations give R_s ≈ 0.7775."""
        errors = {}
        for factorization in FACTORIZATIONS:
            result = rcwa2d.solve_crossed_grating(
                [_pillar(12.25 * EYE)], (0.6, 0.6), 1.0, 0.3, 1.0, 2.1, (5, 5),
                factorization=factorization,
            )  # fmt: skip
            errors[factorization] = abs(float(result.R[_zeroth(result), 1, 1]) - 0.7775)
        assert errors["normal"] < 0.01 < errors["laurent"]


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestGradients:
    @staticmethod
    def _objective(params, factorization):
        width, thickness, wavelength, corner = params
        vertices = rcwa2d.rectangle((0.3, 0.3), (width, 0.25)).at[2].add(jnp.array([corner, 0.0]))
        disc = rcwa2d.Shape(rcwa2d.ellipse((0.3, 0.3), (0.1, 0.08)), SILICON)
        layers = [
            rcwa2d.PatternedLayer(thickness, EYE, (rcwa2d.Shape(vertices, LN_TILTED),)),
            rcwa2d.PatternedLayer(0.1, EYE, (disc,)),
        ]
        result = rcwa2d.solve_crossed_grating(
            layers, (0.6, 0.55), wavelength, 0.3, 1.0, 2.1, (3, 3), azimuth_rad=0.2,
            factorization=factorization,
        )  # fmt: skip
        z = _zeroth(result)
        return result.R[z, 1, 0] + 0.3 * result.T[z, 0, 0] + result.R.sum() * 0.1

    @pytest.mark.parametrize("factorization", FACTORIZATIONS)
    def test_against_finite_differences(self, factorization):
        params = jnp.array([0.28, 0.22, 0.9, 0.03])
        grad = np.asarray(jax.grad(self._objective)(params, factorization))
        step = 1e-5
        for i in range(params.size):
            e = jnp.zeros_like(params).at[i].set(step)
            fd = (
                self._objective(params + e, factorization)
                - self._objective(params - e, factorization)
            ) / (2 * step)
            assert grad[i] == pytest.approx(float(fd), rel=1e-5, abs=1e-9), i

    def test_disc_radius(self):
        def reflectance(radius):
            disc = rcwa2d.Shape(rcwa2d.ellipse((0.3, 0.3), (radius, radius)), SILICON)
            result = rcwa2d.solve_crossed_grating(
                [rcwa2d.PatternedLayer(0.25, EYE, (disc,))], (0.6, 0.6), 1.0, 0.0, 1.0, 2.1,
                (3, 3),
            )  # fmt: skip
            return result.R[_zeroth(result), 0, 0]

        grad = float(jax.grad(reflectance)(0.17))
        fd = float((reflectance(0.17 + 1e-5) - reflectance(0.17 - 1e-5)) / 2e-5)
        assert grad == pytest.approx(fd, rel=1e-5)

    def test_jit(self):
        solve = jax.jit(
            rcwa2d.solve_crossed_grating,
            static_argnames=("max_order", "factorization", "doublings", "grid"),
        )
        args = ([_pillar(LN_TILTED)], (0.6, 0.6), 1.0, 0.3, 1.0, 2.1)
        np.testing.assert_allclose(
            solve(*args, max_order=(2, 2)).R,
            rcwa2d.solve_crossed_grating(*args, max_order=(2, 2)).R,
            atol=1e-13,
        )


def test_unknown_factorization_raises():
    with pytest.raises(ValueError, match="factorization"):
        rcwa2d.solve_crossed_grating(
            [_pillar(SILICON)], (0.6, 0.6), 1.0, 0.0, 1.0, 2.1, (1, 1), factorization="li"
        )


# ---------------------------------------------------------------------------
# General lattices
# ---------------------------------------------------------------------------


def _silicon_disc(centre=(0.0, 0.0), radius=0.2):
    return rcwa2d.Shape(rcwa2d.ellipse(centre, (radius, radius), 48), SILICON)


HEX = rcwa2d.Lattice.hexagonal(0.6)


class TestLattices:
    def test_constructors(self):
        a1, a2 = np.asarray(HEX.a1), np.asarray(HEX.a2)
        assert np.linalg.norm(a1) == pytest.approx(0.6)
        assert np.linalg.norm(a2) == pytest.approx(0.6)
        assert np.degrees(np.arccos(a1 @ a2 / 0.36)) == pytest.approx(60.0)
        assert rcwa2d.Lattice.square(0.5) == rcwa2d.Lattice((0.5, 0.0), (0.0, 0.5))

    def test_uniform_layers_are_the_planar_solver_on_any_lattice(self):
        result = rcwa2d.solve_crossed_grating(
            [rcwa2d.uniform(LN_TILTED, 0.4)], HEX, 0.8, 0.4, 1.0, 2.25, (3, 3), azimuth_rad=0.3
        )
        rotation = bcm.rotation_matrix(0.0, 0.0, -0.3)
        planar = bcm.solve_stack(
            bcm.rotate_tensor(LN_TILTED[None], rotation), jnp.array([0.4]), 0.8, 0.4, 1.0, 2.25
        )
        np.testing.assert_allclose(result.r[_zeroth(result)], planar.r, atol=1e-12)

    @pytest.mark.parametrize(("factorization", "atol"), [("laurent", 1e-12), ("normal", 1e-5)])
    def test_circular_truncation_does_not_depend_on_the_basis(self, factorization, atol):
        """One square lattice, two primitive bases: the same harmonics, the same answer."""
        shape = rcwa2d.Shape(rcwa2d.ellipse((0.2, 0.15), (0.17, 0.12), 48, 0.3), LN_TILTED)
        layer = rcwa2d.PatternedLayer(0.25, EYE, (shape,))
        results = [
            rcwa2d.solve_crossed_grating(
                [layer],
                lattice,
                1.0,
                0.4,
                1.0,
                2.1,
                (5, 5),
                azimuth_rad=0.3,
                factorization=factorization,
                truncation="circular",
            )  # fmt: skip
            for lattice in (rcwa2d.Lattice.square(0.6), rcwa2d.Lattice((0.6, 0.0), (0.6, 0.6)))
        ]
        assert results[0].orders.shape == results[1].orders.shape
        np.testing.assert_allclose(
            results[0].R[_zeroth(results[0])], results[1].R[_zeroth(results[1])], atol=atol
        )

    def test_hexagonal_is_its_rectangular_supercell(self):
        """A triangular lattice is a rectangular one (a, a√3) with two discs per cell."""
        a = 0.6
        hexagonal = rcwa2d.solve_crossed_grating(
            [rcwa2d.PatternedLayer(0.25, EYE, (_silicon_disc(),))], HEX, 1.0, 0.3, 1.0, 2.1,
            (5, 5), factorization="laurent", truncation="circular",
        )  # fmt: skip
        pair = (_silicon_disc(), _silicon_disc((a / 2, a * np.sqrt(3) / 2)))
        supercell = rcwa2d.solve_crossed_grating(
            [rcwa2d.PatternedLayer(0.25, EYE, pair)],
            rcwa2d.Lattice.rectangular(a, a * np.sqrt(3)),
            1.0,
            0.3,
            1.0,
            2.1,
            (10, 10),
            factorization="laurent",
            truncation="circular",
        )  # fmt: skip  -- its shortest reciprocal vector is half the hexagonal one
        np.testing.assert_allclose(
            hexagonal.R[_zeroth(hexagonal)], supercell.R[_zeroth(supercell)], atol=1e-12
        )

    @pytest.mark.parametrize(("factorization", "atol"), [("laurent", 1e-12), ("normal", 3e-4)])
    def test_sixfold_symmetry(self, factorization, atol):
        layer = rcwa2d.PatternedLayer(0.25, EYE, (_silicon_disc(),))

        def zeroth(azimuth, incidence=0.4):
            result = rcwa2d.solve_crossed_grating(
                [layer], HEX, 1.0, incidence, 1.0, 2.1, (5, 5), azimuth_rad=azimuth,
                factorization=factorization, truncation="circular",
            )  # fmt: skip
            return np.asarray(result.R[_zeroth(result)])

        np.testing.assert_allclose(zeroth(0.3), zeroth(0.3 + np.pi / 3), atol=atol)
        normal = zeroth(0.0, incidence=0.0)
        assert normal[0, 0] == pytest.approx(normal[1, 1], abs=atol)

    def test_hexagonal_against_fmmax(self):
        """Si discs (r = 0.2 µm) on a 0.6 µm triangular lattice, 0.3 rad incidence.

        FMMAX (JONES_DIRECT, 1615 terms, 1000² raster): R_p 0.56889, R_s 0.99224.
        """
        result = rcwa2d.solve_crossed_grating(
            [rcwa2d.PatternedLayer(0.25, EYE, (_silicon_disc(),))], HEX, 1.0, 0.3, 1.0, 2.1,
            (8, 8), truncation="circular",
        )  # fmt: skip
        reflected = np.asarray(result.R[_zeroth(result)]).sum(0)
        assert reflected[0] == pytest.approx(0.56889, abs=0.015)  # still converging from above
        assert reflected[1] == pytest.approx(0.99224, abs=4e-3)
        np.testing.assert_allclose(_total(result), 1.0, atol=1e-11)

    def test_gradient_in_the_lattice_constant(self):
        def reflectance(period):
            result = rcwa2d.solve_crossed_grating(
                [rcwa2d.PatternedLayer(0.25, EYE, (_silicon_disc(),))],
                rcwa2d.Lattice.hexagonal(period),
                1.0,
                0.3,
                1.0,
                2.1,
                (3, 3),
            )
            return result.R[_zeroth(result)].sum()

        grad = float(jax.grad(reflectance)(0.6))
        fd = float((reflectance(0.6 + 1e-5) - reflectance(0.6 - 1e-5)) / 2e-5)
        assert grad == pytest.approx(fd, rel=1e-5)

    def test_circular_truncation_needs_a_concrete_lattice(self):
        def traced(period):
            return rcwa2d.solve_crossed_grating(
                [rcwa2d.uniform(SILICON, 0.1)],
                rcwa2d.Lattice.hexagonal(period),
                1.0,
                0.3,
                1.0,
                2.1,
                (2, 2),
                truncation="circular",
            ).R.sum()

        with pytest.raises(ValueError, match="concrete lattice"):
            jax.grad(traced)(0.6)


def test_tilted_axis_nanofin_against_the_paper():
    """LiNbO₃ fins with the optic axis 45° out of plane (ε has an xz term), arXiv:2608.06185
    Sec. 3A: T_xx = 0.9496 (BCM) and 0.9483 (FEM). The mirror y → −y maps the fin and the
    axis to themselves, so x → y conversion vanishes."""
    axis = jnp.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    ln = (5.46 * jnp.eye(3) + (5.06 - 5.46) * jnp.outer(axis, axis)).astype(complex)
    fin = rcwa2d.Shape(rcwa2d.rectangle((0.17, 0.17), (0.2, 0.1)), ln)
    layer = rcwa2d.PatternedLayer(0.3, jnp.eye(3, dtype=complex), (fin,))
    result = rcwa2d.solve_crossed_grating([layer], (0.34, 0.34), 0.55, 0.0, 1.0, 1.46**2, (5, 5))
    zeroth = int(np.flatnonzero((result.orders == 0).all(axis=1))[0])
    transmitted = np.asarray(result.T[zeroth])
    assert transmitted[0, 0] == pytest.approx(0.9496, abs=3e-3)
    assert transmitted[1, 0] < 1e-20
