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

## From a PDK: stacks and gratings (gplugins-style)

The functions below take a layout and the PDK's `LayerStack` and return
labelled results, in the shape of gdsfactory's simulation plugins:

```python
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

# x-cut thin-film LiNbO3 on insulator; ribs on layer (2, 0) etched 0.3 µm
wafer = LogicalLayer(layer=gf.get_layer("WAFER"))
tfln = LayerStack(layers={
    "substrate": LayerLevel(layer=wafer, zmin=-3.0, thickness=1.0, material="si", mesh_order=100),
    "box": LayerLevel(layer=wafer, zmin=-2.0, thickness=2.0, material="sio2", mesh_order=9),
    "ln_slab": LayerLevel(layer=wafer, zmin=0.0, thickness=0.3, material="ln", mesh_order=2),
    "ln_rib": LayerLevel(layer=LogicalLayer(layer=(2, 0)), zmin=0.3, thickness=0.3,
                         material="ln", mesh_order=2),
})
materials = {"ln": hog.Oriented("LiNbO3", hog.x_cut(0.0))}   # the cut: yours to state

ribs = hog.line_grating(period_um=1.2, fill=0.5, n_periods=12, layer=(2, 0))
grating, stack, diffraction = hog.simulate_grating(
    ribs, (2, 0), tfln, wavelengths_um=[1.55], angles_deg=[30.0], materials=materials,
)
slab, planar = hog.simulate_stack(
    ribs, tfln, point_um=(0.9, 10.0), wavelengths_um=[1.55],
    angles_deg=[0.0, 30.0, 60.0], azimuths_deg=[0.0, 45.0], materials=materials,
)
planar.mueller().shape      # (1, 3, 2, 4, 4): [wavelength, angle, azimuth, 4, 4]
```

### How a stack is read

`vertical_stack` takes the PDK's column of materials at one point of the
layout.

- **Which levels are present.** Levels on `WAFER` are present everywhere.
  Others are present where their layer has shapes, and derived (boolean)
  layers are evaluated.
- **Overlaps.** Where levels overlap in z, the lowest `mesh_order` wins, as
  gdsfactory defines it.
- **Half-spaces.** The lowest level becomes the exit half-space, and `ambient`
  (air by default) the incident one.
- **Ambiguity is refused.** A gap no level covers, or a mesh-order tie between
  different materials, raises.

`grating_stack` compares that column inside the ridges of a line grating with
the column in its gaps. Where they differ, the layer is patterned: ridges of
one material, grooves of the other. So etch depth, partial etches and what
fills the grooves all come from the PDK. The ridges and gaps are sampled at
several points in two periods, so a layer patterned some other way is refused
rather than mistaken for part of the grating.

### Materials and crystal orientation

A PDK names its materials (`"si"`, `"sio2"`, `"ln"`, …), and `materials` maps
those names to this package's models, on top of `DEFAULT_MATERIALS` (`si`,
`sio2`, `sin` and `air`). A value can be:

- **A built-in material name.** The birefringent crystals at chip wavelengths:
  `"LiNbO3"`, `"TiO2"`, `"QuartzSellmeier"`, `"CalciteSellmeier"`,
  `"SapphireSellmeier"`. The PIC materials: `"SiO2"`, `"Si3N4"`, `"Si"`. The
  infrared hyperbolic phonon models: `"hBN"`, `"MoO3"`, `"Calcite"`, … All are
  evaluated with their dispersion, and warn outside their fitted range.
- **A number** (isotropic ε) or **a 3×3 tensor**.
- **A callable** `wavelength_um → ε`, for example a Drude metal.
- **`Oriented(material, rotation_deg)`**, a crystal with its orientation.

Anisotropic PDK materials have no default, deliberately: the PDK can't know
the wafer cut. State it with:

- `x_cut(axis_azimuth_deg)`: optic axis in the wafer plane, at that angle from
  layout x;
- `z_cut()`: optic axis along the wafer normal;
- or any payload-style rotation.

### Results

`simulate_stack` returns a `PlanarResponse` over
`[wavelength, angle, azimuth]`:

- `r`, `t` (complex Jones matrices) and `R`, `T` (powers), all
  `[..., out, in]` with 0 = p and 1 = s;
- `mueller()`, the Mueller matrices;
- `ellipsometry()`: Ψ, Δ and the generalized anisotropy ratios.

The azimuth turns the sample about the stack normal. `solver="jax"`
(differentiable) is the default, and `solver="numpy"` runs the package's
eigendecomposition engine instead. The two agree to rounding.

`simulate_grating` returns a `GratingResponse` over `[wavelength, angle, order]`,
with diffraction efficiencies per order and polarization, and `mueller(order)`.

`examples/tfln_gdsfactory.py` runs both on
[Luxtelligence's open-source LNOI400 PDK](https://github.com/Luxtelligence/lxt_pdk_gf)
at 1550 nm:

```bash
uv run --with lnoi400 python examples/tfln_gdsfactory.py
```

- **Stacks come from the PDK's `LayerStack`:** under its rib waveguide
  `straight_rwg1000`, in a slab region, and under a line grating on `LN_RIDGE`,
  where the 0.2 µm partial etch with SiO₂-filled grooves is found automatically.
- **The cut is an assumption.** The PDK doesn't record it, so the example assumes
  x-cut, which is what its electro-optic cells imply.
- **What it plots:**
  - the slab's p→s conversion against crystal rotation and angle;
  - a Mueller element that is zero for any isotropic stack;
  - the grating's −1 order for the optic axis across and along the ridges.

`tests/test_gdsfactory.py` checks the extraction against the same PDK when
`lnoi400` is installed.

## Hand-written stacks

A `StackLayer`'s `material` and `background` take the same specifications, and
`rotation_deg` orients a plain name. `VerticalStack(layers, incident, exit)`
with `solve_stack` / `solve_grating` runs a stack written by hand.

## Optimisation

`grating_layers(LineGrating(period, fill), stack, wavelength)` accepts JAX
values throughout, so an objective written with it can be differentiated with
respect to period, fill, thicknesses and wavelength. Wavelength derivatives
include the material dispersion.

## 2D-periodic patterns: pillars, holes, any polygons

The same pattern of functions covers patterns that repeat in x and y:

```python
from hyperbolic_optics import rcwa2d

disc = tuple(map(tuple, rcwa2d.ellipse((0.3, 0.25), (0.12, 0.08), 48).tolist()))
array = hog.periodic_array(period_um=(0.6, 0.5), polygons=(disc,), n_cells=(20, 20),
                           layer=(2, 0))
array.write_gds("array.gds")
cell = hog.extract_unit_cell(gf.import_gds("array.gds"), (2, 0))    # or from `array`
cell, stack, response = hog.simulate_crossed_grating(
    array, (2, 0), tfln, wavelengths_um=[1.55], angles_deg=[0.0, 10.0],
    max_order=(5, 5), materials=materials,
)
response.R[0, 0, response.order(0, 0)]      # [out, in] of the zeroth order
```

- **`periodic_array`** repeats one cell's polygons on a lattice, which is
  rectangular (`period_um`) or given by its vectors (`lattice_um`), e.g.
  `((a, 0), (a/2, a·√3/2))` for a hexagonal one. The lattice vectors and every
  vertex are snapped to the layout grid, so each cell is an exact translate of
  the first.
- **`extract_unit_cell`** reads any 2D-periodic pattern back. The polygons are
  grouped into translated copies; every group must fill the same complete
  lattice, and the motif repeated on that lattice must be exactly what is
  drawn. Overlapping shapes (including across cell edges), holes and missing
  cells are refused.
  - The lattice is inferred: rectangular, hexagonal or oblique, dimers
    included. It is the shortest basis under which the drawing is one motif
    on a complete parallelogram of cells.
  - Pass `period_um` or `lattice_um` to impose one instead.
  - `cell.lattice` is the `rcwa2d.Lattice` to solve with.
- **`crossed_grating_stack`** compares the `LayerStack` column inside the
  shapes with the column around them, in two cells, as `grating_stack` does
  for ridges and gaps.
- **`crossed_grating_layers(cell, stack, wavelength)`** accepts JAX polygons,
  so a shape can be optimised and then drawn with `periodic_array`.

Stripes drawn as a 2D array (rectangles spanning the y period) give the line
grating's result to 1e-12; `tests/test_gdsfactory.py` checks it.

### Example: a hyperbolic α-MoO₃ polarization converter

`examples/moo3_metasurface_gdsfactory.py` optimises elliptical MoO₃ pillars on
a KBr spacer and gold mirror. MoO₃ is in-plane hyperbolic (ε_x < 0 < ε_y) at
880–920 cm⁻¹. The objective is the mean cross-polarized reflectance over that
band, at normal incidence.

- **Parameters:** ellipse radii and rotation, the crystal's in-plane rotation,
  and both thicknesses, at 81 harmonics.
- **Result:** mean R_ps goes from 0.08 to 0.857.
  - The optimiser turns the crystal to 44.9° and makes the ellipse nearly
    round. The conversion comes from MoO₃'s hyperbolicity, not the shape: with
    the ellipse along x it is still 0.83, and with the crystal along x it is
    2e-5.
- **Verification:** the design is written to GDS (15 × 15 cells), read back
  with its stack from a `LayerStack`, and re-simulated.
  - 0.861 at 81 harmonics;
  - 0.81–0.85 at 225 harmonics.

It takes about 5 minutes on a laptop CPU and writes the GDS and a four-panel
figure to `examples/output/`.

## Negative refraction in tilted calcite

`benchmarks/calcite_negative_refraction_gdsfactory.py` is the grating
counterpart of FDTDX's
[`hyperbolic_asymmetric_refraction.py`](https://github.com/ymahlau/fdtdx/blob/main/examples/hyperbolic_asymmetric_refraction.py),
kept as a cross-check against a full-wave code.

## S-parameters for circuits (gplugins-style)

```python
import numpy as np

strip = gf.components.straight(length=10, width=0.5)
sp = hog.write_sparameters_stack(strip, gf.get_active_pdk().layer_stack, (5.0, 0.0),
                                 np.linspace(1.5, 1.6, 3), angle_deg=20.0)
sp["top@p,bottom@s"]                  # p in from the top -> s out of the bottom
model = hog.model_from_npz(sp)        # or the saved .npz file's path
model(wl=1.55)[("top@p", "bottom@s")]
```

- **Signature.** A component, the PDK's `LayerStack` and a point in; the
  four-port plane-wave S-matrix of the stack there out, over wavelength.
- **Keys.** `"port@mode,port@mode"` (input first, as in SAX), plus
  `"wavelengths"`.
- **Caching.** Saved as `.npz` under `gf.config.PATH.sparameters`, keyed by a
  hash of the geometry and the settings; `overwrite=True` recomputes.
- **Reading.** `hog.model_from_npz` keeps the polarization modes apart;
  gplugins' reader drops the mode label.

## Metasurfaces and metalenses

A slowly varying metasurface is designed cell by cell, with local
periodicity: each cell transmits as it would in an infinite array of itself.
2D RCWA gives that transmission, and the layout places the cells.

```python
from hyperbolic_optics import metasurfaces as ms, rcwa2d

radii = np.linspace(0.04, 0.16, 25)
library = ms.cell_library(ms.disc_cell(2.4**2, 0.6, 0.35), radii,     # [P, 2, 2] Jones
                          rcwa2d.Lattice.square(0.35), 0.633, 1.45**2, 1.0, (3, 3))
sites = ms.square_sites(10.0, 0.35)                                   # [K, 2]
phase = ms.lens_phase(sites[:, 0], sites[:, 1], 20.0, 0.633)
indices, realised = ms.choose_cells(phase, library)
lens = hog.metasurface_component(sites, indices, hog.disc_polygons(radii), layer=(1, 0))
```

- **Libraries.** `ms.cell_library` takes any `parameter → layers`: pillar
  radius, fin size, or an anisotropic crystal pillar's orientation. It
  returns the full 2 × 2 Jones matrix, so conversion between polarizations
  is kept.
- **Geometric phase.** A form-birefringent or crystalline cell turned by θ,
  `ms.geometric_phase_jones(jones, θ)`, converts circular polarization with
  phase 2θ. Pass `rotations_rad` to `hog.metasurface_component` to draw the
  turned cells.
- **Twisted stacks.** A cell can be several anisotropic layers, each turned
  about z, patterned or not (see below).
- **Layout.** Each library entry is one cell, and each site a reference to it,
  so a lens of thousands of pillars stays a small GDS.
- **Checking.** `ms.focus` propagates the designed field by the angular
  spectrum, a check on the design rather than a full-wave simulation of the
  finite lens.

Two half-wave layers twisted by α rotate polarization by 2α, so circular
light keeps its handedness and gains a phase of −2α. The twist alone then
spans 2π:

```python
import jax.numpy as jnp

from hyperbolic_optics import bcm

n_o, n_e = 1.50, 1.60
eps = jnp.diag(jnp.array([n_e**2, n_o**2, n_o**2], dtype=complex))
d = 1.55 / (2 * (n_e - n_o))                                  # half-wave


def twisted(alpha):
    turns = [bcm.rotation_matrix(0.0, 0.0, k * alpha) for k in range(2)]
    return [rcwa2d.uniform(bcm.rotate_tensor(eps, r), d) for r in turns]


alphas = np.linspace(0, np.pi, 13)
library = ms.cell_library(twisted, alphas, (0.5, 0.5), 1.55, 1.55**2, 1.55**2, (0, 0))
```

Unpatterned cells need only the zeroth order, `(0, 0)`. For twisted
pillars, give each layer's `Shape` its own turned tensor.

`examples/metalens_gdsfactory.py` designs a geometric-phase lens of TiO₂
nanofins at 660 nm, from the fin scan to the GDS. Each fin converts 0.94 of
the light. The spot is diffraction-limited, and the on-axis maximum falls
at the Li–Wolf focal shift for its Fresnel number.

## SAX models of PDK stacks

`hog.stack_sax_model(vertical_stack(component, layer_stack, point))` turns the
unpatterned stack at a point of a layout into a SAX model, with every material
evaluated with its dispersion at SAX's `wl`. See
[Scattering Matrices and SAX](sax.md).

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

- **One etch:** every patterned layer shares the same shapes, for line
  gratings and for 2D patterns alike.
- **2D-periodic layouts** use the [crossed-grating solver](crossed-gratings.md)
  on any lattice (see above).
- **Infinitely periodic, plane-wave results.** The number of periods drawn does
  not enter, so this describes the interior of a large array.
