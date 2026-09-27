"""Tests for the full-vector 2D mode solver (``hyperbolic_optics.modes2d``).

- a film filling the domain: its TE family's lowest mode is the slab's, with
  the quarter-wave lateral profile the boundaries impose (magnetic wall at the
  low edge, electric at the high one) -- exactly, for isotropic and uniaxial
  films, against the 1D solver;
- a square core is symmetric under swapping x and y: isotropic, its TE and
  TM modes are degenerate; uniaxial, turning the axis from x to y swaps them;
- a waveguide read from a PDK layout is the one drawn by hand.

Skipped when JAX is not installed (the 1D references need it).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from hyperbolic_optics import bcm, modes2d, resonances  # noqa: E402

WAVELENGTH, CLAD = 1.55, 1.444**2
K0 = 2 * np.pi / WAVELENGTH


def _film_limit(core, thickness, step):
    """(2D TE-family index, exact) for a film filling a 1 µm-wide domain."""
    exx, eyy, ezz = core
    width = 1.0
    lateral = np.pi / (2 * K0 * (width + step / 2))  # the quarter-wave between the walls

    def slab(phi):  # the film's TE mode with its in-plane wavevector turned by phi
        base = jnp.asarray(np.diag([ezz, exx, eyy]).astype(complex))  # 1D: x₁ = z, y₁ = x, z₁ = y
        rotated = bcm.rotate_tensor(base, bcm.rotation_matrix(0.0, 0.0, -phi))
        modes = resonances.slab_modes(rotated[None], jnp.array([thickness]), WAVELENGTH, CLAD, CLAD)
        return max(m.effective_index.real for m in modes)

    phi = brentq(lambda p: slab(p) * np.sin(p) - lateral, 1e-4, 0.8)
    exact = slab(phi) * np.cos(phi)
    x = np.arange(0.0, width + step / 2, step)
    y = np.arange(-1.0, thickness + 1.0 + step / 2, step)
    section = modes2d.rectangles(x, y, CLAD, [((-1.0, width + 1.0, 0.0, thickness), core)])
    modes = modes2d.solve_modes(section, WAVELENGTH, 3, exact)
    return [m.effective_index.real for m in modes if m.te_fraction > 0.5][0], exact


@pytest.mark.parametrize(
    ("core", "thickness"),
    [((12.1, 12.1, 12.1), 0.22), ((4.6, 4.2, 4.9), 0.6)],
    ids=["isotropic", "uniaxial"],
)
def test_a_film_filling_the_domain_is_the_slab(core, thickness):
    coarse, coarse_exact = _film_limit(core, thickness, 0.02)  # the walls move with the step
    fine, fine_exact = _film_limit(core, thickness, 0.01)
    assert abs(fine - fine_exact) < 1e-3
    assert abs(fine - fine_exact) < 0.5 * abs(coarse - coarse_exact)  # converging


@pytest.fixture(scope="module")
def square_grid():
    return np.arange(-1.5, 1.5 + 0.01, 0.02)


def test_an_isotropic_square_core_is_polarization_degenerate(square_grid):
    section = modes2d.rectangles(square_grid, square_grid, CLAD, [((-0.4, 0.4, -0.4, 0.4), 4.6)])
    te, tm = modes2d.solve_modes(section, WAVELENGTH, 2, 2.1)
    assert te.effective_index.real == pytest.approx(tm.effective_index.real, abs=1e-9)


def test_turning_the_axis_of_a_square_core_swaps_te_and_tm(square_grid):
    indices = {}
    for axis, core in (("x", (4.9, 4.6, 4.6)), ("y", (4.6, 4.9, 4.6))):
        section = modes2d.rectangles(
            square_grid, square_grid, CLAD, [((-0.4, 0.4, -0.4, 0.4), core)]
        )
        modes = modes2d.solve_modes(section, WAVELENGTH, 2, 2.1)
        indices[axis] = {
            ("TE" if m.te_fraction > 0.5 else "TM"): m.effective_index.real for m in modes
        }
    assert indices["x"]["TE"] == pytest.approx(indices["y"]["TM"], abs=1e-9)
    assert indices["x"]["TM"] == pytest.approx(indices["y"]["TE"], abs=1e-9)
    assert indices["x"]["TE"] > indices["x"]["TM"]  # the axis along E raises the index


def test_off_diagonal_tensors_are_refused(square_grid):
    tilted = np.array([[4.8, 0.1, 0], [0.1, 4.6, 0], [0, 0, 4.6]])
    with pytest.raises(ValueError, match="off-diagonal"):
        modes2d.rectangles(square_grid, square_grid, CLAD, [((-0.4, 0.4, -0.4, 0.4), tilted)])


def test_a_waveguide_read_from_a_layout():
    gf = pytest.importorskip("gdsfactory")
    from hyperbolic_optics.integrations import gdsfactory as hog

    gf.gpdk.PDK.activate()
    strip = gf.components.straight(length=10, width=0.5)
    section, modes = hog.waveguide_modes(
        strip, gf.get_active_pdk().layer_stack, (5.0, 0.0), WAVELENGTH, 2,
        lateral_um=(-1.5, 1.5), vertical_um=(-1.5, 1.7), step_um=0.02,
    )  # fmt: skip
    silicon = section.eps[2][len(section.x_um) // 2, np.searchsorted(section.y_um, 0.1)]
    oxide = section.eps[2][0, 5]
    hand = modes2d.rectangles(
        section.x_um, section.y_um, oxide, [((-0.25, 0.25, 0.0, 0.22), silicon)]
    )
    for a, b in zip(section.eps, hand.eps, strict=True):
        np.testing.assert_array_equal(a, b)
    te, tm = modes
    assert te.te_fraction > 0.9 and tm.te_fraction < 0.1
    assert 2.40 < te.effective_index.real < 2.50  # a 500 × 220 nm silicon strip in oxide
    assert 1.70 < tm.effective_index.real < 1.85
