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

- **Signature.** A model is `model(wl, kx, azimuth) -> SDict`, keyed
  `(in, out)` as SAX expects, over the four multimode ports above.
  - `wl` is SAX's wavelength in µm.
  - `azimuth` turns every layer about the stack normal.
- **Materials.** A material is a number, a 3×3 tensor, or a callable of the
  wavelength, so dispersion comes along.
- **From a PDK.** `hog.stack_sax_model(stack)` does the same for a stack read
  from a PDK with `vertical_stack`.
- **Metasurface cells.** `rcwa2d.crossed_grating_smatrix` is a 2D-periodic
  stack's zeroth-order S-matrix on the same four ports, and
  `hog.cell_sax_model(cell, lattice)` wraps it as a SAX model (see
  [gdsfactory](gdsfactory.md#metasurfaces-with-different-cells-at-every-site)).
  Only the zeroth order is a port, so it holds for a subwavelength array.
  Cell models use the same conserved `kx`, sample `azimuth` and complex √kz
  normalization as planar models, including evanescent zeroth-order ports.
  Their wavelength-dependent materials support JIT and differentiation.
  Connect it to other elements across a spacer thick enough for its
  omitted orders to decay: an order with decay constant α has amplitude
  `exp(−αd)` across a spacer of thickness d. Check convergence with d,
  especially near a diffraction threshold where α approaches zero.
- **Connections.** In a netlist, connect one model's `bottom` to another's
  `top`, and pass `kx` as a global setting. SAX matches the `@p` and `@s`
  modes itself.
- **Gradients.** Use `sax.circuit(..., backend="filipsson_gunnar")`, which is
  pure JAX and exact. SAX's default sparse (KLU) backend evaluates these
  models but fails in the backward pass.
- **Not a dependency.** SAX pins JAX below 0.10, which narrows this package's
  JAX range to 0.9, so `hyperbolic_optics.integrations.sax` doesn't import it.
  The models are plain functions returning dictionaries. Add SAX when needed:
  `uv run --with sax python …`.
- **Plane-wave ports only.** They are not waveguide modes. Connecting them to
  a guided-mode model needs an overlap or mode-matching step that is not
  provided.

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

`Structure.execute` computes the response to light from the incident side
only. That gives the Jones matrices r and t, which are 8 of the 16 entries. A
circuit also needs the light that arrives from below. A dictionary holding
only those 8 would silently drop it, since SAX treats missing entries as zero.

`structure_smatrix` / `structure_sdict` therefore build the whole matrix:
- **From the payload's own tensors.** They use the executed structure's layer
  tensors and thicknesses, so named materials and their rotations, custom
  tensors, magnetic layers and thickness sweeps all carry over. The matrix is
  solved with the JAX solver.
- **Consistent with the engine.** The incident-side entries are the engine's
  r, and its t scaled by √(kz_out / kz_in); `tests/test_sax.py` checks both to
  1e-11 over angle, azimuth and k-space sweeps.
- **Isotropic exit only.** A `Semi Infinite Anisotropic Layer` exit is refused.

## Polarization conversion and sweep axes in SAX

- **Polarization is a mode.** SAX's multimode ports are `port@mode`. Each
  side of a stack is one port with two modes, p and s, and conversion is an
  off-diagonal entry such as `("top@p", "top@s")`. Connecting `a,bottom` to
  `b,top` connects both modes, and SAX composes the full 4 × 4 matrices, so
  conversion propagates through a circuit. The circuit tests compare all 16
  entries, cross-polarized ones included.
- **Every other variable is an array axis.** SAX combines S-matrices
  elementwise, so the values of an `SDict` can have any shape:
  - wavelength, angle (as kx), azimuth or thickness become axes;
  - arrays of the same or broadcastable shape compose;
  - a 2D map is a pair of meshgrid arrays passed as `wl` and `kx`, as in the
    Otto example.

  All instances in a circuit must see the same grid, so pass `wl` and `kx` as
  global settings. For a payload's sampled `SDict`, keep every element on the
  same scenario grid.
- **One plane of incidence.** The p/s basis is defined in the x-z plane. An
  azimuth rotates the sample, not the plane, so every element of a circuit
  shares its p and s.

## Example: Otto coupling to calcite's surface polaritons

`examples/sax_otto_calcite.py` builds `KRS-5 prism | air gap | x-cut calcite`
as a three-instance SAX circuit at calcite's ordinary Reststrahlen band.

- **The circuit is the stack.** It equals the same stack solved in one piece
  to 1e-15, in all 16 entries.
- **Axis across the plane of incidence.** The p reflectance map shows the
  surface phonon polariton, reached by tunnelling through the evanescent gap.
- **Axis at 60°.** The polariton mixes with calcite's bulk hyperbolic waves,
  and converts p to s (up to 0.16). With the axis across the plane it is 1e-30.
- **Critical coupling.** Adam on the gap and kx together, through the circuit,
  reaches R_pp = 6e-11 at 1470 cm⁻¹. The gap is 2.39 µm and kx = 1.197, near
  the isotropic estimate √(ε_o/(ε_o + 1)) ≈ 1.18.

```bash
uv run --with sax python examples/sax_otto_calcite.py
```

It takes about 20 s and writes a four-panel figure to `examples/output/`.
