"""Reference amplitudes computed independently with NGSolve, without a runtime FEM dependency."""

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from hyperbolic_optics import rcwa2d  # noqa: E402


def test_biaxial_cell_matches_independent_fem():
    data = json.loads(
        (
            Path(__file__).parents[1]
            / "benchmarks"
            / "validation"
            / "anisotropic_cell_reference.json"
        ).read_text()
    )["fem"]
    fine = next(row for row in data["runs"] if row["maxh"] == 0.08 and row["padding"] == 1.5)
    layer = rcwa2d.PatternedLayer(
        data["height_um"],
        np.eye(3),
        (rcwa2d.Shape(rcwa2d.rectangle((0.0, 0.0), data["size_um"]), np.asarray(data["epsilon"])),),
    )
    s = np.asarray(
        rcwa2d.crossed_grating_smatrix(
            [layer], (data["pitch_um"],) * 2, data["wavelength_um"], 0.0, 1.0, 1.0, (8, 8)
        ).s
    )
    for key, result in [("r", s[:2, :2]), ("t", s[2:, :2])]:
        expected = np.array(fine[f"{key}_real"]) + 1j * np.array(fine[f"{key}_imag"])
        # Empirical Fourier/mesh convergence envelope; no LAPACK eigenvector coordinates.
        np.testing.assert_allclose(result, expected, rtol=0, atol=0.002)
    assert fine["energy_error"] < 2e-5
