"""SAX circuits of planar models (``hyperbolic_optics.integrations.sax``).

The models' dictionaries are checked everywhere; the circuits only where SAX is
installed. SAX pins an older JAX than this package's, so it is not a dependency;
run these with ``uv run --with sax pytest tests/test_sax.py``.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hyperbolic_optics import bcm, dispersion  # noqa: E402
from hyperbolic_optics.integrations import sax as hsax  # noqa: E402

EYE = jnp.eye(3, dtype=complex)
LN = bcm.rotate_tensor(
    jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex)), bcm.rotation_matrix(0.0, 0.7, 0.4)
)


def _whole(wl, kx):
    return bcm.stack_smatrix(jnp.stack([LN, 2.1 * EYE]), jnp.array([0.4, 0.2]), wl, kx, 1.0, 2.25)


class TestModels:
    def test_sdict_is_keyed_in_then_out(self):
        model = hsax.planar_model([(LN, 0.4), (2.1 * EYE, 0.2)], 1.0, 2.25)
        sd = model(wl=0.8, kx=0.4)
        s = np.asarray(_whole(0.8, 0.4).s)
        assert len(sd) == 16
        for i_in, port_in in enumerate(hsax.PORTS):
            for i_out, port_out in enumerate(hsax.PORTS):
                assert complex(sd[port_in, port_out]) == pytest.approx(s[i_out, i_in], abs=1e-14)

    def test_dispersive_callable_and_azimuth(self):
        def hbn(wl):
            return dispersion.at_wavelength("hBN", wl)

        model = hsax.planar_model([(hbn, 0.1)], 1.0, 11.7)
        wl = jnp.array([6.5, 6.7])
        sd = model(wl=wl, kx=0.2, azimuth=0.3)
        rotated = bcm.rotate_tensor(hbn(wl), bcm.rotation_matrix(0.0, 0.0, 0.3))
        direct = bcm.stack_smatrix(rotated[:, None], jnp.array([0.1]), wl, 0.2, 1.0, 11.7)
        np.testing.assert_allclose(sd["top@p", "bottom@s"], direct.s[:, 3, 0], atol=1e-14)

    def test_angle_helper(self):
        assert float(hsax.kx_from_angle(np.pi / 6, 2.25)) == pytest.approx(0.75)


try:
    import sax
except ImportError:  # pragma: no cover - SAX is optional
    sax = None


def _circuit(netlist, models, backend="default"):
    circuit, _ = sax.circuit(netlist, models, backend=backend)
    return circuit


@pytest.mark.skipif(sax is None, reason="SAX is not installed (uv run --with sax ...)")
class TestCircuits:
    def test_two_stacks_connected_are_the_stack_of_both(self):
        models = {
            "upper": hsax.planar_model([(LN, 0.4)], 1.0, 2.1),
            "lower": hsax.planar_model([(2.1 * EYE, 0.2)], 2.1, 2.25),
        }
        circuit = _circuit(
            {
                "instances": {"a": "upper", "b": "lower"},
                "connections": {"a,bottom": "b,top"},
                "ports": {"top": "a,top", "bottom": "b,bottom"},
            },
            models,
        )
        sd = circuit(wl=0.8, kx=0.4)
        s = np.asarray(_whole(0.8, 0.4).s)
        for i_in, port_in in enumerate(hsax.PORTS):
            for i_out, port_out in enumerate(hsax.PORTS):
                assert complex(sd[port_in, port_out]) == pytest.approx(s[i_out, i_in], abs=1e-12)

    def test_tunnelling_through_an_evanescent_gap(self):
        """glass | air | glass beyond the critical angle, as three SAX instances."""
        models = {
            "glass_to_air": hsax.planar_model([], 2.25, 1.0),
            "gap": hsax.planar_model([(1.0, 0.3)], 1.0, 1.0),
            "air_to_glass": hsax.planar_model([], 1.0, 2.25),
        }
        circuit = _circuit(
            {
                "instances": {"a": "glass_to_air", "g": "gap", "b": "air_to_glass"},
                "connections": {"a,bottom": "g,top", "g,bottom": "b,top"},
                "ports": {"top": "a,top", "bottom": "b,bottom"},
            },
            models,
        )
        sd = circuit(wl=1.0, kx=1.2)
        whole = bcm.stack_smatrix(jnp.stack([EYE]), jnp.array([0.3]), 1.0, 1.2, 2.25, 2.25)
        np.testing.assert_allclose(sd["top@s", "bottom@s"], whole.s[3, 1], atol=1e-12)
        np.testing.assert_allclose(sd["top@p", "top@p"], whole.s[0, 0], atol=1e-12)

    def test_gradients_through_the_circuit(self):
        """SAX's default KLU backend fails in the backward pass; its pure-JAX one is exact."""

        def conversion(thickness):
            models = {
                "film": hsax.planar_model([(LN, thickness)], 1.0, 2.1),
                "substrate": hsax.planar_model([], 2.1, 2.25),
            }
            circuit = _circuit(
                {
                    "instances": {"f": "film", "s": "substrate"},
                    "connections": {"f,bottom": "s,top"},
                    "ports": {"top": "f,top", "bottom": "s,bottom"},
                },
                models,
                backend="filipsson_gunnar",
            )
            return jnp.abs(circuit(wl=0.8, kx=0.4)["top@p", "top@s"]) ** 2

        grad = float(jax.grad(conversion)(0.4))
        fd = float((conversion(0.4 + 1e-6) - conversion(0.4 - 1e-6)) / 2e-6)
        assert grad == pytest.approx(fd, rel=1e-6)
