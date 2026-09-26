# gdsfactory Integration

`hyperbolic_optics.integrations.gdsfactory` connects the
[grating solver](gratings.md) to [gdsfactory](https://gdsfactory.github.io/)
layouts. With it you can:

- draw a line grating as a PCell;
- read a grating back from any component or GDS file;
- describe the vertical stack, including dispersive hyperbolic materials;
- simulate or optimise, then export the result and verify it.

```bash
pip install "hyperbolic-optics[gdsfactory]"
# or, in a checkout
uv sync --extra gdsfactory
```

gdsfactory needs an active PDK before it draws anything, and this module does
not activate one for you:

```python
import gdsfactory as gf
gf.gpdk.PDK.activate()     # or your own PDK
```

## Layout → simulation

```python
import jax
jax.config.update("jax_enable_x64", True)
from hyperbolic_optics.integrations import gdsfactory as hog

component = hog.line_grating(period_um=2.65, fill=0.72, n_periods=30, length_um=60.0,
                             layer=(1, 0))
component.write_gds("absorber.gds")

grating = hog.extract_line_grating(gf.import_gds("absorber.gds"), layer=(1, 0))

def gold(wavelength_um):  # Drude gold, approximate
    w = 1e4 / wavelength_um
    return 1.0 - 7.28e4**2 / (w**2 + 1j * 215.0 * w)

stack = [
    hog.StackLayer(0.148, "hBN", background=1.0),   # hBN ribbons in air
    hog.StackLayer(1.48, 1.85),                      # CaF₂ spacer
    hog.StackLayer(0.2, gold),                       # any callable λ → ε
]
grating, result = hog.simulate_component(
    component, (1, 0), stack, wavelength_um=1e4 / 1500, incidence_rad=0.0,
    eps_incident=1.0, eps_exit=11.7, max_order=40,
)
```

### The PCell

`line_grating` snaps the period and ridge width to the layout grid. Every drawn
period is then identical, so the GDS describes a truly periodic structure, and
the snapped values are recorded in `component.info`.

### Extraction

`extract_line_grating` accepts only axis-aligned rectangles of equal width and
length, at equal pitch. It raises `ValueError` naming whatever else it finds,
because a solver fed a guessed geometry returns a confident, wrong answer.

## Materials

A `StackLayer`'s `material` and `background` can each be:

- **A built-in material name:** `"hBN"`, `"MoO3"`, `"Calcite"`, `"Sapphire"`,
  `"SiC"`, `"AlN"`, `"GaN"`, `"Quartz"` or `"GalliumOxide"`. These are the
  package's phonon-polariton models, evaluated by
  [`hyperbolic_optics.dispersion`](#dispersion-in-jax), and they match the
  NumPy classes to rounding.
- **A number** (isotropic ε) or **a 3×3 tensor**.
- **A callable** `wavelength_um → ε`, for example a Drude metal.

`rotation_deg` sets a crystal's orientation, with the payload API's
`rotationX/Y/Z` convention.

## Optimisation

`grating_layers(LineGrating(period, fill), stack, wavelength)` accepts JAX
values throughout, so an objective written with it can be differentiated with
respect to period, fill, thicknesses and wavelength. Wavelength derivatives
include the material dispersion.

`examples/hbn_absorber_gdsfactory.py` does the whole loop:

1. It optimises a hyperbolic hBN ribbon absorber on a gold mirror at 1500 cm⁻¹,
   varying period, fill, hBN thickness and spacer thickness, at 41 orders.
2. It draws the result as this PCell and writes the GDS.
3. It reads the GDS back and re-simulates at 81 and 161 orders.

| | absorption at 1500 cm⁻¹ |
|---|---|
| initial design | 0.065 |
| optimised (41 orders) | 0.939 |
| from the GDS, 81 orders | 0.938 |
| from the GDS, 161 orders | 0.938 |
| mirror and spacer only (gold's share) | 0.010 |

It takes about 25 s on a laptop CPU and writes the GDS and a four-panel
figure to `examples/output/`.

## Dispersion in JAX

`hyperbolic_optics.dispersion.permittivity("hBN", wavenumber_cm)` returns the
crystal-frame tensor `[..., 3, 3]` and is differentiable in frequency.
`at_wavelength` takes µm instead.

The models are the same oscillator fits as the NumPy classes, read from each
class's own parameters. The test suite checks every material against its class
to 1e-12, and the frequency derivatives against finite differences.

## Geometry gradients on hyperbolic ridges

Gradients with respect to fill and offset are exact, but a hyperbolic ridge
can make the truncated model's dependence on edge position ripple. See
[Gratings (RCWA)](gratings.md#gradients-with-respect-to-geometry-on-hyperbolic-ridges).

- **The absorber above is not affected.** It has a subwavelength period and
  flakes in their natural orientation, and its d/d(fill) is smooth and
  converged even with hard edges.
- **Check your own design.** Compare the gradient at two truncations before
  trusting it. Where it ripples, pass `edge_width` to `grating_layers`.

## Scope

- **Line gratings only:** parallel ridges, periodic in one direction, and one
  etch shared by every patterned layer.
- **2D-periodic layouts** (holes, pillars, arbitrary polygons) need 2D RCWA,
  which is not implemented.
- **Infinitely periodic, plane-wave results.** The number of periods drawn does
  not enter, so this describes the interior of a large array, not a finite
  device such as a grating coupler.
