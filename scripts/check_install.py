"""Run with an isolated interpreter (-I) after installing a built wheel."""

import argparse
import sys
from importlib.metadata import version

import numpy as np

from hyperbolic_optics import Structure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--integration", action="store_true")
args = parser.parse_args()
assert "jax" not in sys.modules and "gdsfactory" not in sys.modules
stack = Structure()
stack.execute(
    {
        "ScenarioData": {"type": "Incident", "incidentAngle": [0.0], "frequency": [1000.0]},
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 1.0},
            {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.25},
        ],
    }
)
np.testing.assert_allclose(abs(stack.r_pp) ** 2, 0.04)
if args.integration:
    import jax

    jax.config.update("jax_enable_x64", True)
    import gdsfactory as gf

    from hyperbolic_optics import dispersion
    from hyperbolic_optics.integrations import gdsfactory as hog

    gf.gpdk.PDK.activate()
    assert np.isfinite(dispersion.at_wavelength("LiNbO3", 1.55)).all()
    cell = hog.VerticalStack((), exit=2.25)
    s = hog.cell_sax_model(cell, ((0.4, 0.0), (0.0, 0.4)), (0, 0))(wl=1.55)
    np.testing.assert_allclose(abs(s["top@p", "top@p"]) ** 2, 0.04)
    assert callable(hog.write_sparameters_cell)
    print({p: version(p) for p in ("jax", "gdsfactory", "sax")})
print("Installed wheel smoke checks passed")
