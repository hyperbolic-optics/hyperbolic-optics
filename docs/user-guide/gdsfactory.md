# gdsfactory Integration

`hyperbolic_optics.integrations.gdsfactory` (`hog`) connects the solvers to
[gdsfactory](https://gdsfactory.github.io/) layouts:
- read a PDK `LayerStack` at a point, under a line grating or under a 2D
  pattern, with crystals in the cut you state;
- draw metasurfaces whose cells differ site by site (shape, material, crystal
  orientation, twist, number of layers) and read them back;
- turn any of it into SAX models and gplugins-style S-parameters.

```bash
pip install "hyperbolic-optics[gdsfactory]"     # or: uv sync --extra gdsfactory
```

gdsfactory needs an active PDK before it draws anything:

```python
import gdsfactory as gf
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
gf.gpdk.PDK.activate()     # or your own PDK

from hyperbolic_optics.integrations import gdsfactory as hog
```

## Stacks from a PDK

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

- **The column at a point.** Levels on `WAFER` are everywhere; others where
  their layer has shapes, derived layers included. Where levels overlap, the
  lowest `mesh_order` wins. The lowest level is the exit half-space, and
  `ambient` (air) the incident one. Gaps and mesh-order ties between
  materials are refused.
- **Gratings and 2D patterns.** `grating_stack` and `crossed_grating_stack`
  compare the column inside the shapes with the column around them. Where
  they differ the layer is patterned, so etch depth and groove fill come from
  the PDK.
- **Materials.** `materials` maps the PDK's names onto `DEFAULT_MATERIALS`
  (`si`, `sio2`, `sin`, `air`). A value is a built-in name (`"LiNbO3"`,
  `"TiO2"`, `"hBN"`, `"MoO3"`, …, with dispersion), a number or 3×3 tensor, a
  callable `λ → ε`, or `Oriented(material, rotation_deg)`.
- **Orientation.** An anisotropic PDK material has no default, because the
  PDK can't know the wafer cut. Use `x_cut(azimuth_deg)`, `z_cut()`, or any
  payload-style rotation.
- **Results.** `PlanarResponse` over `[wavelength, angle, azimuth]` and
  `GratingResponse` / `CrossedGratingResponse` over `[wavelength, angle,
  order]` carry `r`, `t`, `R` and `T` (`[..., out, in]`, 0 = p, 1 = s), plus
  `mueller()` and `ellipsometry()`. `solver="numpy"` runs the eigenmode
  engine instead of JAX; the two agree to rounding.

`examples/tfln_gdsfactory.py` runs this on
[Luxtelligence's LNOI400 PDK](https://github.com/Luxtelligence/lxt_pdk_gf)
(`uv run --with lnoi400 …`). It reads the rib waveguide and a partially etched
grating whose grooves are filled with SiO₂.

## Line gratings and periodic arrays

```python
from hyperbolic_optics import rcwa, rcwa2d

component = hog.line_grating(period_um=2.65, fill=0.72, n_periods=30, layer=(1, 0))
component.write_gds("absorber.gds")
grating = hog.extract_line_grating(gf.import_gds("absorber.gds"), layer=(1, 0))
stack = [hog.StackLayer(0.148, "hBN", background=1.0), hog.StackLayer(1.48, 1.85)]
result = rcwa.solve_grating(hog.grating_layers(grating, stack, 1e4 / 1500), grating.period_um,
                            1e4 / 1500, 0.0, 1.0, 11.7, 40)

disc = tuple(map(tuple, rcwa2d.ellipse((0.3, 0.25), (0.12, 0.08), 48).tolist()))
array = hog.periodic_array(period_um=(0.6, 0.5), polygons=(disc,), n_cells=(20, 20),
                           layer=(2, 0))
cell, stack, response = hog.simulate_crossed_grating(
    array, (2, 0), tfln, wavelengths_um=[1.55], angles_deg=[0.0, 10.0],
    max_order=(5, 5), materials=materials,
)
response.R[0, 0, response.order(0, 0)]      # [out, in] of the zeroth order
```

- The PCells snap every vertex and lattice vector to the layout grid, so the
  GDS is exactly periodic.
- `extract_line_grating` and `extract_unit_cell` read the pattern back from
  any layout. `extract_unit_cell` infers the lattice: rectangular, hexagonal
  or oblique, dimers included. Anything that is not exactly periodic is
  refused, never approximated.
- `grating_layers` and `crossed_grating_layers` take JAX values throughout
  (fill, polygons, thicknesses, wavelength), so designs can be optimised and
  then drawn.

`examples/moo3_metasurface_gdsfactory.py` optimises elliptical α-MoO₃ pillars
and the crystal's in-plane angle for cross-polarized reflection (mean R_ps
0.08 → 0.86 over 880–920 cm⁻¹). It then draws the design, reads it back
through a `LayerStack` and re-simulates it.

## Metasurfaces with different cells at every site

A cell is a `VerticalStack` whose layers carry their own `polygons` (in cell
coordinates about its centre). Each layer has its own material, thickness,
shapes and crystal orientation, and cells may have different numbers of
layers.

```python
from hyperbolic_optics import metasurfaces as ms

ln = hog.Oriented("LiNbO3", hog.x_cut(0.0))
fin = (np.array([[-0.125, -0.05], [0.125, -0.05], [0.125, 0.05], [-0.125, 0.05]]),)


def cell(twist_deg, n_twisted):
    """n LiNbO3 films, each turned twist_deg further, under a TiO2 fin."""
    films = tuple(hog.StackLayer(0.1, ln, rotation_deg=(0, 0, k * twist_deg))
                  for k in range(n_twisted))
    top = hog.StackLayer(0.3, 2.4**2, background=1.0, polygons=fin, name="fin")
    return hog.VerticalStack(films + (top,), incident=1.0, exit=1.45**2)


lattice = ((0.4, 0.0), (0.0, 0.4))
sites = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
cells = [cell(0, 1), cell(30, 2), cell(45, 3)]
layout = hog.metasurface_component(sites, cells, [(10, 0), (11, 0), (12, 0), (13, 0)],
                                   lattice, rotations_rad=[0.0, 0.3, 0.6])
layout.write_gds("metasurface.gds")
sites_back, cells_back, _ = hog.read_metasurface(gf.import_gds("metasurface.gds"))

model = hog.cell_sax_model(cells_back[0], lattice, max_order=(3, 3))   # SAX: wl, kx, azimuth
model(wl=0.633)["top@p", "bottom@s"]
```

- **Layout.** Each distinct cell is drawn once: level `k` goes on
  `layers[k]`, and an unpatterned level is drawn as the cell's footprint.
  Its levels (material names, thicknesses, orientations) are recorded in the
  cell's `info`. Every site is a reference, turned if asked; a turned
  reference turns the whole cell, crystal included. The same design gives the
  same cached component.
- **Reading back.** `read_metasurface` gives each site's cell, with geometry
  from the polygons and materials from `info` (named materials are resolved
  through `materials=`). What is re-simulated is what was drawn.
  Flattening the layout discards `info`.
- **Solving.** `hog.cell_layers(cell, λ)` gives a cell's
  [rcwa2d](crossed-gratings.md) layers. `ms.cell_library(f, parameters, …)`
  solves a batch of cells in one compiled call, differentiable in the
  parameters: a discrete library, or every site of a design. A layer of
  zero thickness changes nothing, so padding lets cells with different
  layer counts share a batch.
- **Designing.** `ms.lens_phase`, `ms.choose_cells`, `ms.sample_grid` and
  `ms.focus` (angular spectrum, in JAX) cover a lens. `ms.focus` is also a
  differentiable objective in per-site parameters. All of this is local
  periodicity: each cell as in an infinite array of itself, not a full-wave
  simulation of the finite surface.

`examples/metalens_gdsfactory.py` designs a geometric-phase lens of TiO₂
nanofins. The ideal rotator (`e^{2iθ}` at 0.95 conversion) assumes the lattice
turns with the fin. Solving the fin turned in its fixed lattice gives 0.68
between about 25° and 65°, with phase errors up to 0.3 rad. Choosing each site's
angle from that real library gives a diffraction-limited focus at 0.61
focusing efficiency, against 0.77 for ideal rotators. The GDS reads back and
re-simulates to 1e-13.

## SAX models and S-parameters

- **`hog.stack_sax_model(stack)`**: an unpatterned stack (for example from
  `vertical_stack`) as a SAX model over `top@p`, `top@s`, `bottom@p` and
  `bottom@s`, with dispersion evaluated at SAX's `wl`. See
  [Scattering Matrices and SAX](sax.md).
- **`hog.cell_sax_model(cell, lattice)`**: a metasurface cell's
  zeroth-order S-matrix
  (`rcwa2d.crossed_grating_smatrix`) on the same ports. Only the zeroth
  order is a port, so it describes a subwavelength array. Connected in SAX
  to a stack below it across a spacer thick enough for the evanescent
  orders to decay, it equals the whole structure solved at once (tested to
  1e-9).
- **`hog.write_sparameters_stack(component, layer_stack, point, wavelengths)`**:
  the PDK stack's S-matrix at a point, cached gplugins-style as `.npz` under
  `gf.config.PATH.sparameters` and keyed `"port@mode,port@mode"`.
  `hog.model_from_npz` reads it into SAX with the polarization modes kept
  (gplugins' reader drops them).

```python
strip = gf.components.straight(length=10, width=0.5)
sp = hog.write_sparameters_stack(strip, gf.get_active_pdk().layer_stack, (5.0, 0.0),
                                 np.linspace(1.5, 1.6, 3), angle_deg=20.0)
hog.model_from_npz(sp)(wl=1.55)[("top@p", "bottom@s")]
```

## Scope

- Results are for infinitely periodic structures under plane waves. A
  finite device's edges are outside them.
- A line grating or `periodic_array` has one etch shared by every patterned
  layer. Metasurface cells lift this: each layer has its own polygons.
- Gradients with respect to edge positions can ripple on hyperbolic ridges;
  see [Gratings](gratings.md#gradients-with-respect-to-geometry-on-hyperbolic-ridges).
- `benchmarks/calcite_negative_refraction_gdsfactory.py` is a calcite grating
  kept for a cross-check against the full-wave code
  [FDTDX](https://github.com/ymahlau/fdtdx).
