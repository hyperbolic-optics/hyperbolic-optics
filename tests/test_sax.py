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


class TestFromPayloads:
    """``structure_smatrix``: the engine's own r and t, completed with light from below."""

    LAYERS = [
        {"type": "Ambient Incident Layer", "permittivity": 5.5},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 1.5, "permittivity": 1.0},
        {
            "type": "Crystal Layer",
            "material": "Calcite",
            "thickness": 0.6,
            "rotationY": 70,
            "rotationZ": 40,
        },  # fmt: skip
        {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
    ]

    SIMPLE = {"type": "Simple", "incidentAngle": 30.0, "azimuthal_angle": 0.0, "frequency": 1470.0}

    @staticmethod
    def _executed(scenario, layers):
        from hyperbolic_optics.structure import Structure

        structure = Structure()
        structure.execute({"ScenarioData": scenario, "Layers": layers}, backend="scattering")
        return structure

    @pytest.mark.parametrize(
        "scenario",
        [
            {"type": "Incident", "incidentAngle": [5.0, 25.0, 45.0, 65.0]},
            {
                "type": "Azimuthal",
                "incidentAngle": 40.0,
                "frequency": 1470.0,
                "azimuthal_angle": [0.0, 30.0, 60.0],
            },  # fmt: skip
            {"type": "Dispersion", "frequency": 1470.0},
        ],
        ids=["angle", "azimuth", "angle-and-azimuth"],
    )
    def test_incident_side_is_the_engines_r_and_t(self, scenario):
        from hyperbolic_optics.scattering import scattering_coefficients

        structure = self._executed(scenario, self.LAYERS)
        s = np.asarray(hsax.structure_smatrix(structure).s)
        engine = scattering_coefficients(structure.layers, structure.k_0)
        kz_in = np.sqrt(5.5 - np.real(np.asarray(structure.k_x)) ** 2 + 0j)
        kz_out = np.sqrt(2.0 - np.real(np.asarray(structure.k_x)) ** 2 + 0j)
        names = {"pp": (0, 0), "ps": (1, 0), "sp": (0, 1), "ss": (1, 1)}  # (out, in)
        for pair, (out, inc) in names.items():
            r = np.broadcast_to(np.asarray(engine[f"r_{pair}"]), s.shape[:-2])
            t = np.broadcast_to(np.asarray(engine[f"t_{pair}"]), s.shape[:-2])
            np.testing.assert_allclose(s[..., out, inc], r, atol=1e-11, err_msg=f"r_{pair}")
            lit = np.broadcast_to(np.abs(kz_in) > 1e-9, s.shape[:-2])  # not at grazing cutoff
            scale = np.broadcast_to(np.sqrt(kz_out / np.where(lit, kz_in, 1.0)), s.shape[:-2])
            np.testing.assert_allclose(
                s[..., 2 + out, inc][lit], (t * scale)[lit], atol=1e-11, err_msg=f"t_{pair}"
            )
        # and the rest of the matrix is physical: passive between the ports that
        # propagate (|s|² is not a power for an evanescent one)
        propagating = np.asarray(hsax.structure_smatrix(structure).propagating)
        flat_s, flat_p = s.reshape(-1, 4, 4), propagating.reshape(-1, 4)
        for matrix, open_ports in zip(flat_s, flat_p, strict=True):
            if not open_ports.any():
                continue
            block = matrix[np.ix_(open_ports, open_ports)]
            gain = np.linalg.eigvalsh(np.eye(len(block)) - block.conj().T @ block)
            assert gain.min() > -1e-10

    def test_thickness_sweep_and_zero_layers(self):
        swept = [dict(self.LAYERS[0]), {**self.LAYERS[2], "thickness": [0.2, 0.4, 0.8]},
                 self.LAYERS[3]]  # fmt: skip
        structure = self._executed(self.SIMPLE, swept)
        s = hsax.structure_smatrix(structure).s
        assert s.shape[-3] == 3 and s.shape[-2:] == (4, 4)
        bare = self._executed(self.SIMPLE, [self.LAYERS[0], self.LAYERS[3]])
        sd = hsax.structure_sdict(bare)
        kx = np.sqrt(5.5) * np.sin(np.radians(30.0))
        c1, c2 = np.sqrt(5.5 - kx**2), np.sqrt(2.0 - kx**2)
        assert complex(np.squeeze(sd["top@s", "top@s"])) == pytest.approx((c1 - c2) / (c1 + c2))

    def test_anisotropic_exit_is_refused(self):
        layers = [self.LAYERS[0], {"type": "Semi Infinite Anisotropic Layer", "material": "Calcite",
                                   "rotationY": 70}]  # fmt: skip
        structure = self._executed(self.SIMPLE, layers)
        with pytest.raises(NotImplementedError, match="exit"):
            hsax.structure_smatrix(structure)


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
