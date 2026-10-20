#!/usr/bin/env python3
"""Optimise a LiNbO₃ film on glass into a broadband p-to-s converter by gradient ascent.

Thickness and in-plane axis angle are tuned for p→s transmission over
600–700 nm; the answer is a half-wave plate at 45°. The design is re-checked
with the NumPy engine, which shares no code with the JAX path.

Run:  uv run --extra jax python examples/optimize_polarization_converter.py
"""

import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hyperbolic_optics import bcm  # noqa: E402
from hyperbolic_optics.fields import FieldProfile  # noqa: E402
from hyperbolic_optics.structure import Structure  # noqa: E402

EPS_O, EPS_E = 5.46, 5.06  # LiNbO3, dispersion neglected
EPS_GLASS = 1.52**2
WAVELENGTH_UM = jnp.linspace(0.60, 0.70, 41)
THICKNESS_RANGE_UM = (1.0, 6.0)


def thickness_um(raw):
    """Map an unconstrained parameter onto the allowed thickness range."""
    low, high = THICKNESS_RANGE_UM
    return low + (high - low) * jax.nn.sigmoid(raw)


def film_tensor(axis_rad):
    """Extraordinary axis along x, then rotated about z by ``axis_rad``."""
    crystal = jnp.diag(jnp.array([EPS_E, EPS_O, EPS_O], dtype=complex))
    return bcm.rotate_tensor(crystal, bcm.rotation_matrix(0.0, 0.0, axis_rad))


def conversion(params):
    """p-to-s transmitted power at each wavelength."""
    result = bcm.solve_stack(
        film_tensor(params["axis"])[None],
        thickness_um(params["raw_thickness"])[None],
        WAVELENGTH_UM,
        0.0,
        1.0,
        EPS_GLASS,
    )
    return result.T[..., 1, 0]  # [out = s, in = p]


@jax.jit
def loss_and_grad(params):
    return jax.value_and_grad(lambda p: -jnp.mean(conversion(p)))(params)


def adam(params, steps=150, rate=0.05, beta1=0.9, beta2=0.999):
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    for step in range(1, steps + 1):
        loss, grad = loss_and_grad(params)
        first = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, first, grad)
        second = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g**2, second, grad)

        def update(p, m, v, step=step):
            m_hat, v_hat = m / (1 - beta1**step), v / (1 - beta2**step)
            return p - rate * m_hat / (jnp.sqrt(v_hat) + 1e-12)

        params = jax.tree.map(update, params, first, second)
        if step in (1, 10, 50, steps):
            print(
                f"  step {step:4d}: mean conversion {-float(loss):.4f}, "
                f"d = {float(thickness_um(params['raw_thickness'])):.3f} µm, "
                f"axis = {np.degrees(float(params['axis'])):.2f}°"
            )
    return params


def numpy_engine_conversion(d_um, axis_deg):
    """The same film through the payload API, for an independent check."""
    payload = {
        "ScenarioData": {
            "type": "Incident",
            "incidentAngle": [0.0],
            "frequency": (1e4 / np.asarray(WAVELENGTH_UM)).tolist(),
        },
        "Layers": [
            {"type": "Ambient Incident Layer", "permittivity": 1.0},
            {
                "type": "Crystal Layer",
                "material": {"eps_xx": EPS_E, "eps_yy": EPS_O, "eps_zz": EPS_O},
                "thickness": d_um,
                "rotationZ": axis_deg,
            },
            {"type": "Semi Infinite Isotropic Layer", "permittivity": EPS_GLASS},
        ],
    }
    structure = Structure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # arbitrary tensors carry no validity range
        structure.execute(payload)
    return np.ravel(FieldProfile(structure).polarization_resolved("p")["T_cross"])


def main():
    params = {"raw_thickness": jnp.asarray(0.0), "axis": jnp.radians(10.0)}
    print("Optimising (the first step includes JIT compilation):")
    start = time.perf_counter()
    params = adam(params)
    print(f"  {time.perf_counter() - start:.1f} s in total")

    d = float(thickness_um(params["raw_thickness"]))
    axis_deg = float(np.degrees(params["axis"]))
    birefringence = np.sqrt(EPS_O) - np.sqrt(EPS_E)
    print(f"\nOptimised: d = {d:.3f} µm, axis = {axis_deg:.2f}°")
    print(f"Half-wave plate at 650 nm: d = {0.65 / (2 * birefringence):.3f} µm, axis = 45°")

    jax_curve = np.asarray(conversion(params))
    numpy_curve = numpy_engine_conversion(d, axis_deg)
    print(f"\nMean p→s conversion over 600–700 nm: {jax_curve.mean():.4f}")
    print(
        "Re-evaluated with the NumPy engine: "
        f"max |difference| = {np.max(np.abs(jax_curve - numpy_curve)):.1e}"
    )


if __name__ == "__main__":
    main()
