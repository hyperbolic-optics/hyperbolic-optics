# gdsfactory Integration

`hyperbolic_optics.integrations.gdsfactory` (`hog`) reads PDK layer stacks and
[gdsfactory](https://gdsfactory.github.io/) layouts into the solvers, with
crystal cuts stated, draws metasurfaces cell by cell, and exports SAX models.

```bash
pip install "hyperbolic-optics[gdsfactory]"     # or: uv sync --extra gdsfactory
```

It needs gdsfactory `>=9.51,<10` and JAX `>=0.9.2,<0.12`; SAX is optional
(see [SAX](sax.md)).

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
  `ambient` (air) the incident one. Gaps and mesh-order ties between materials
  are refused. For 2D-periodic cells the exit may be a crystal or absorbing
  wafer, whose two transmitted modes replace p and s.
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

`examples/moo3_metasurface_gdsfactory.py` optimises α-MoO₃ pillars and the
crystal's in-plane angle for cross-polarized reflection, then draws, reads back
and re-simulates the design.

## Metasurfaces with different cells at every site

A cell is a `VerticalStack` whose layers carry their own `polygons`, about
the cell's centre; each layer has its own material, thickness, shapes and
orientation. `cell_from_pdk` reads one from a centred layout through the PDK:
gaps become ambient spacers, shapes crossing the cell's edge wrap around, the
lowest interval must be uniform, and each other interval may hold at most two
materials.

```python
mask = gf.Component()
mask.add_polygon([(-0.12, -0.05), (0.12, -0.05), (0.12, 0.05), (-0.12, 0.05)], layer=(2, 0))
pdk_cell = hog.cell_from_pdk(mask, tfln, ((0.4, 0.0), (0.0, 0.4)), materials=materials)
```

`sidewall_slices=N` models tapered levels by N slices, from the PDK's
`sidewall_angle`, `width_to_z` and scalar `bias`; otherwise walls are vertical
and a warning names the ignored process geometry.

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

- **Layout.** Each distinct cell is drawn once, level `k` on `layers[k]`,
  with its materials in the cell's `info`; every site is a reference, turned
  if asked (the lattice never turns). Crystals turn with the reference unless
  `rotate_materials=False` (a mask on a wafer with fixed axes; `hog.turned`
  takes the same flag). Shapes that, turned, overlap each other or their
  periodic images are refused; off-grid vertices warn.
- **Reading back.** `read_metasurface` gives each site's cell, geometry from
  the drawn polygons and materials from `info` (callables by name, through
  `materials=`). Nested layouts, arrays, mirrors and magnification are refused.
- **From a PDK.** Cells may instead be `gf.Component`s drawn on PDK layers;
  `read_metasurface(layout, materials, layer_stack=…)` reads each site through
  `cell_from_pdk`, turned as placed, crystals on the wafer's axes.
- **Solving.** `hog.cell_layers(cell, λ)` gives a cell's rcwa2d layers;
  `ms.cell_library` solves a batch in one compiled, differentiable call
  (zero-thickness layers pad cells to a common layer count). `ms.lens_phase`,
  `ms.choose_cells`, `ms.sample_grid` and `ms.focus` design a lens by local
  periodicity: each cell as in an infinite array of itself. Six x-cut LiNbO₃
  fins turned by kπ/6 on a fixed wafer, solved as one supercell, put 0.40 of
  the power into the deflected order where their library entries predict 0.37.

`examples/metalens_gdsfactory.py` turns 300 × 110 × 600 nm TiO₂ fins in a fixed
325 nm lattice at 660 nm. Converged (`--max-order 16`), a fin converts 0.86 at
0° but about 0.675 near 45°, and the lens designed from that library focuses
0.59 of the power within 1.5 FWHM, against 0.70 for ideal rotators.

## SAX models and S-parameters

- `hog.stack_sax_model(stack)` and `hog.cell_sax_model(cell, lattice)` give
  SAX models of a stack and of a cell's zeroth order (see [SAX](sax.md)).
- `hog.write_sparameters_stack` and `hog.write_sparameters_cell` cache them
  gplugins-style as `.npz`, keyed on the evaluated materials and thicknesses;
  `hog.model_from_npz` reads one back with the polarization modes kept.

```python
strip = gf.components.straight(length=10, width=0.5)
sp = hog.write_sparameters_stack(strip, gf.get_active_pdk().layer_stack, (5.0, 0.0),
                                 np.linspace(1.5, 1.6, 3), angle_deg=20.0, dirpath="sparameters")
hog.model_from_npz(sp)(wl=1.55)[("top@p", "bottom@s")]

cell_sp = hog.write_sparameters_cell(cells_back[0], lattice, [0.63, 0.64],
                                    max_order=(1, 1), dirpath="sparameters")
hog.model_from_npz(cell_sp)(wl=0.633)[("top@p", "bottom@s")]
```

For a PDK cell, pass `hog.cell_from_pdk(component, layer_stack, lattice, materials)`
to `write_sparameters_cell`.

## Scope

| Capability | Status |
|---|---|
| PDK layers, crystal cuts, anisotropic periodic-cell gradients | Supported |
| Linear sidewalls and scalar bias | Sliced approximation; refinement required |
| Cached four-port plane-wave models | Supported; closed model only below diffraction thresholds |
| Finite metasurface | Local periodicity and scalar propagation |
| Guided waveguide ports or a waveguide/free-space bridge | Not provided |
| Independent full-wave comparison | One rotated biaxial periodic cell; see `benchmarks/validation/README.md` in the repository |

- A line grating or `periodic_array` has one etch shared by every patterned
  layer; metasurface cells give each layer its own polygons.
- Gradients with respect to edge positions can ripple on hyperbolic ridges;
  see [Gratings](gratings.md#gradients-with-respect-to-geometry-on-hyperbolic-ridges).
