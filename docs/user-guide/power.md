# Transmission, Absorption & Backends

`Structure.execute` gives you reflection coefficients. Everything else — power
transmittance, where the light is absorbed, the field through the stack — comes
from `FieldProfile`, which consumes an executed structure the same way `Mueller`
does.

## Power quantities

Every snippet on this page follows on from:

```python
payload = {
    "ScenarioData": {"type": "Incident"},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 50.0},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationY": 90,
        },
    ],
}
```

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.fields import FieldProfile

structure = Structure()
structure.execute(payload)

fp = FieldProfile(structure)

fp.reflectance("p")        # R
fp.transmittance("p")      # T, the flux crossing into the exit medium
fp.layer_absorption("p")   # one entry per finite interior layer
fp.summary("p")            # all of the above at once
```

These are computed from the propagated field, not from a closed-form
transmission formula: the tangential field `[Ex, Ey, Hx, Hy]` is continuous
across interfaces, and the normal flux `S_z = ½ Re(Ex·Hy* − Ey·Hx*)` at each
interface gives the power crossing it.

Any polarization works — pass `"p"`, `"s"`, or an explicit complex `(a_s, a_p)`
Jones pair.

!!! warning "`conservation_residual` is a bookkeeping check"
    `summary()` reports `conservation_residual = max|R + T + ΣAᵢ − 1|`. This is
    **not** a physics check. The per-layer absorptances are defined as
    successive differences of the same interface fluxes `R` and `T` are built
    from, so the sum telescopes and the residual is algebraically zero however
    wrong the fields are — it returns machine epsilon even where `R` exceeds 1.

    Use it to catch `NaN` propagation. To check the physics, test **passivity**:
    every layer absorptance and `T` non-negative, `R` at most 1.

## Choosing a backend

The default multiplies per-layer transfer matrices. That is fast, but for thick,
lossy or strongly evanescent layers the propagation terms grow exponentially and
the product loses precision — long before it overflows to `NaN` it returns
finite, plausible, wrong numbers.

```python
structure.execute(payload, backend="scattering")
```

The scattering backend cascades per-layer scattering matrices with the Redheffer
star product, so only *decaying* exponentials ever appear. It returns the same
coefficients as the transfer method where that is well-conditioned, and correct
ones where it is not — a thick evanescent Otto gap gives `NaN` under
`backend="transfer"` and correct total reflection under `backend="scattering"`.

`FieldProfile` works under either. Under the scattering backend the interface
fields are recovered from the cascade rather than by propagating the transfer
matrices, so the power quantities stay correct in that regime too.

| | `transfer` (default) | `scattering` |
|---|---|---|
| reflection / transmission coefficients | ✅ | ✅ |
| `reflectance` / `transmittance` / `layer_absorption` | ✅ | ✅ |
| `field_profile` (resolved with depth) | ✅ | ❌ |
| stable for thick / evanescent stacks | ❌ | ✅ |

You rarely need to choose manually. The default path detects when a
coefficient's `2×2` minor has cancelled down to the rounding floor and
recomputes just those points with the cascade; `Structure.repaired_fraction`
reports how many. Pass `calculate_reflectivity(stabilize=False)` for the literal
transfer product.

## Field profiles with depth

```python
profile = fp.field_profile("p")     # needs backend="transfer"
profile["z"]                        # depth axis
profile["Ex"], profile["Hy"], ...   # six field components
profile["Sz"]                       # Poynting flux vs depth
```

This is batched, but a full angle × frequency sweep times a depth axis times six
components is memory-heavy — it is meant for `Simple` or a single slice.

## Sweeping layer thickness

A layer's `thickness` may be a **list**, which sweeps it as a fourth axis in a
single `execute` — the eigendecomposition does not depend on thickness, so only
the propagation phase broadcasts over it:

```python
{"type": "Crystal Layer", "material": "Calcite",
 "thickness": [0.5, 1.0, 1.5, 2.0], "rotationY": 90}
```

Outputs gain a trailing axis: `Simple` gives `r_pp.shape == (4,)`, `Incident`
gives `(F, angle, 4)`. At most one layer may carry a list thickness.

!!! note "Slice before plotting"
    The map plots draw one 2-D array, so select a slice —
    `plot_kx_frequency(structure, R[..., 0])`. They will tell you if you forget.

For a 2-D thickness × thickness grid, combine a list thickness on one layer with
the `ThicknessSweep` helper, which re-runs the stack and stacks results along a
leading index:

```python
from hyperbolic_optics import ThicknessSweep
```
