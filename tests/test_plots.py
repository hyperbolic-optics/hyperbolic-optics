"""
Tests for the plotting guards.

``plots.py`` is excluded from coverage and had no tests, which is how a silent
rendering failure survived: matplotlib reads a trailing axis of size 3 or 4 as
RGB/RGBA, so a 3- or 4-point thickness sweep drew a plausible, meaningless
picture instead of raising. These lock the guards, not the pixels.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from hyperbolic_optics.mueller import Mueller  # noqa: E402
from hyperbolic_optics.plots import (  # noqa: E402
    plot_kx_frequency,
    plot_poincare_sphere,
)
from hyperbolic_optics.structure import Structure  # noqa: E402


def _swept_thickness(n_thickness):
    payload = {
        "ScenarioData": {"type": "Incident", "polar_points": 16},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 50.0},
            {
                "type": "Crystal Layer",
                "material": "Calcite",
                "thickness": list(np.linspace(0.5, 2.0, n_thickness)),
                "rotationY": 90,
            },
            {
                "type": "Semi Infinite Anisotropic Layer",
                "material": "Calcite",
                "rotationY": 90,
            },
        ],
    }
    structure = Structure()
    structure.execute(payload)
    return structure, np.abs(np.asarray(structure.r_pp)) ** 2


class TestTrailingAxisGuard:
    """A map plot draws one 2-D array, and says so for anything else."""

    @pytest.mark.parametrize("n_thickness", [3, 4])
    def test_rgb_sized_trailing_axis_is_rejected(self, n_thickness):
        """The silent case: matplotlib would render these as colour channels."""
        structure, reflectivity = _swept_thickness(n_thickness)
        with pytest.raises(ValueError, match="colour channels"):
            plot_kx_frequency(structure, reflectivity, save_name=None)

    @pytest.mark.parametrize("n_thickness", [2, 5])
    def test_other_trailing_axes_are_rejected_by_name(self, n_thickness):
        """These already failed, but from inside matplotlib and without naming the axis."""
        structure, reflectivity = _swept_thickness(n_thickness)
        with pytest.raises(ValueError, match=r"param has shape"):
            plot_kx_frequency(structure, reflectivity, save_name=None)

    def test_a_slice_still_renders(self):
        structure, reflectivity = _swept_thickness(4)
        plot_kx_frequency(structure, reflectivity[..., 0], save_name=None)


class TestPoincareColouring:
    """Colouring by a swept axis has to line up with the axis it indexes."""

    @staticmethod
    def _stokes():
        payload = {
            "ScenarioData": {"type": "Incident", "polar_points": 16},
            "Layers": [
                {"type": "Ambient Incident Layer", "permittivity": 50.0},
                {
                    "type": "Semi Infinite Anisotropic Layer",
                    "material": "Calcite",
                    "rotationY": 90,
                },
            ],
        }
        structure = Structure()
        structure.execute(payload)
        mueller = Mueller(structure)
        mueller.set_incident_polarization("linear", angle=45)
        mueller.add_optical_component("anisotropic_sample")
        return structure, mueller.get_stokes_parameters()

    def test_colour_by_frequency(self):
        """NumPy aligns trailing axes, so a length-F array against (F, angle) raised."""
        structure, stokes = self._stokes()
        plot_poincare_sphere(stokes, color_by=np.asarray(structure.frequency), save_name=None)

    def test_colour_by_nothing(self):
        _, stokes = self._stokes()
        plot_poincare_sphere(stokes, save_name=None)
