# Scattering Matrices and SAX

A planar stack has four ports: light can arrive from either side, in either
polarization. `bcm.stack_smatrix` gives the full bidirectional scattering
matrix over those ports, and `hyperbolic_optics.integrations.sax` wraps it as
[SAX](https://github.com/flaport/sax) models, so anisotropic stacks can be
composed in SAX circuits and differentiated through them.

## The S-matrix contract

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from hyperbolic_optics import bcm

ln_tensor = bcm.rotate_tensor(jnp.diag(jnp.array([5.06, 5.46, 5.46], dtype=complex)),
                              bcm.rotation_matrix(0.0, 0.7, 0.4))   # LiNbO3, axis tilted
eps = jnp.stack([ln_tensor, 2.1 * jnp.eye(3)])        # [L, 3, 3], top down
smatrix = bcm.stack_smatrix(eps, jnp.array([0.4, 0.2]), wavelength_um=0.8,
                            kx=0.479, eps_top=1.0, eps_bottom=2.25)
smatrix.s            # [4, 4], [out, in] over bcm.PORTS
smatrix.propagating  # [4], False for a port evanescent in its half-space
```

- **Ports.** `bcm.PORTS = ("top@p", "top@s", "bottom@p", "bottom@s")`. `top`
  is the incident-side half-space and `bottom` the exit side. p and s are
  defined in each half-space's plane of incidence (x-z), as in `solve_stack`.
  The reference planes are the stack's two outer interfaces.
- **`kx`, not an angle.** The in-plane wavevector `n·sin θ` is conserved
  through the stack, while the angle changes from medium to medium. Stacks
  connected to one another must share `kx`.
- **Normalization.** Amplitudes are scaled by √kz of their half-space.
  - Between propagating ports, |s|² is the fraction of power transferred.
    A lossless stack's matrix is unitary when all four ports propagate.
  - The top-incidence block is `solve_stack`'s r, with its R and T.
- **Evanescent channels are kept.** Beyond total internal reflection a port
  is flagged `propagating = False`. It keeps the same √kz scaling, with
  complex kz, so matrices sharing that half-space still compose correctly: a
  gap below the critical angle tunnels. |s|² is not a power for such a port.
- **Reciprocity.** For symmetric ε and μ,
  `s(kx) = D · s(−kx)ᵀ · D` with `D = diag(−1, 1, −1, 1)`. Reversing a wave
  reverses its in-plane wavevector, and the p waves' sign convention (they
  share the sign of Hy) flips on reversal. Gyrotropic media break it.

`bcm.cascade(upper, lower)` connects `upper`'s bottom to `lower`'s top. That
is the stack of both, with the shared half-space of zero thickness, computed
without SAX.

## SAX models

```python
from hyperbolic_optics import dispersion
from hyperbolic_optics.integrations import sax as hsax

film = hsax.planar_model(
    [(lambda wl: dispersion.at_wavelength("hBN", wl), 0.1)],  # (material, thickness_um)
    eps_top=1.0, eps_bottom=11.7,
)
film(wl=6.7, kx=0.3, azimuth=0.2)    # SDict: {("top@p", "bottom@s"): ..., ...}
```

- **Signature.** `model(wl, kx, azimuth) -> SDict`, keyed `(in, out)` over
  the four ports; `wl` is in µm and `azimuth` turns every layer about the
  normal. Materials are numbers, 3×3 tensors or callables of the wavelength.
- **From a PDK.** `hog.stack_sax_model(stack)` takes a stack from
  `vertical_stack`.
- **Metasurface cells.** `hog.cell_sax_model(cell, lattice)` wraps
  `rcwa2d.crossed_grating_smatrix`, the zeroth-order S-matrix, on the same
  ports and normalisation (see
  [gdsfactory](gdsfactory.md#metasurfaces-with-different-cells-at-every-site)).
  Only the zeroth order is a port, so connect a cell to other elements across
  a spacer thick enough for its omitted orders to decay as `exp(−αd)`;
  near a diffraction threshold α → 0, so check convergence with d.
- **Connections.** Connect one model's `bottom` to another's `top` and pass
  `kx` as a global setting; SAX matches the `@p` and `@s` modes.
- **Gradients.** Use `sax.circuit(..., backend="filipsson_gunnar")` (pure
  JAX); the default KLU backend fails in the backward pass.
- **Not a dependency.** SAX pins JAX below 0.10, so this module doesn't import
  it: `uv run --with sax python …`.
- **Plane-wave ports only.** Connecting them to waveguide modes needs a
  mode-matching step that is not provided.

## From payloads: `Structure.execute` → SAX

```python
from hyperbolic_optics.structure import Structure

structure = Structure()
structure.execute({
    "ScenarioData": {"type": "Azimuthal", "incidentAngle": 40.0, "frequency": 1470.0,
                     "azimuthal_angle": [0.0, 30.0, 60.0]},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 5.5},
        {"type": "Isotropic Middle-Stack Layer", "thickness": 1.5, "permittivity": 1.0},
        {"type": "Crystal Layer", "material": "Calcite", "thickness": 0.6, "rotationY": 70},
        {"type": "Semi Infinite Isotropic Layer", "permittivity": 2.0},
    ],
})
sd = hsax.structure_sdict(structure)     # values [A, B, F, T] = [1, 3, 1, 1]
sd["top@p", "top@s"]                     # r_ps (p in, s out) at each azimuth
```

`Structure.execute` gives only the 8 incident-side entries, and SAX treats
missing entries as zero, so `structure_smatrix` / `structure_sdict` re-solve
the executed tensors in JAX for all 16 (they match the engine's r and t to
1e-11). The exit must be a `Semi Infinite Isotropic Layer`.

## Polarization and sweep axes

Polarization is the port mode, so conversion is an off-diagonal entry such as
`("top@p", "top@s")` and composes through a circuit. Every other variable
(wavelength, kx, azimuth, thickness) is an array axis of the SDict values;
pass `wl` and `kx` as global settings so every element sees the same grid. All
elements share the x-z p/s basis: an azimuth turns the sample, not the plane.

`examples/sax_otto_calcite.py` builds prism | air gap | x-cut calcite as a SAX
circuit, maps the surface-polariton dip, shows p→s conversion with the axis
turned, and tunes the gap to critical coupling through the circuit:

```bash
uv run --with sax python examples/sax_otto_calcite.py
```
