"""gdsfactory bridge: PDK stacks and layouts into the solvers, gplugins-style.

Everything is importable from here::

    from hyperbolic_optics.integrations import gdsfactory as hog

- **Geometry** (:mod:`.geometry`, :mod:`.lattice`): :func:`line_grating` and
  :func:`periodic_array`, grid-snapped PCells, and :func:`extract_line_grating`
  and :func:`extract_unit_cell`, which read a line grating or a 2D-periodic
  pattern back from any component or GDS and refuse anything else.
- **Materials** (:mod:`.materials`): material specifications, :class:`Oriented`
  crystals with :func:`x_cut` / :func:`z_cut`, and the mapping from PDK material
  names (:data:`DEFAULT_MATERIALS`).
- **Stacks** (:mod:`.stack`): :class:`VerticalStack` and :class:`StackLayer`,
  written by hand or read from a PDK ``LayerStack`` at a point of a layout
  (:func:`vertical_stack`), under a line grating (:func:`grating_stack`) or
  under a 2D-periodic pattern (:func:`crossed_grating_stack`).
- **Simulation** (:mod:`.simulate`): :func:`simulate_stack`,
  :func:`simulate_grating` and :func:`simulate_crossed_grating` take a layout
  and PDK and return :class:`PlanarResponse`, :class:`GratingResponse` or
  :class:`CrossedGratingResponse`, with Jones, Mueller and ellipsometric
  analysis.

- **S-parameters** (:mod:`.sparameters`), gplugins-style:
  :func:`write_sparameters_grating_coupler` for a PDK grating-coupler cell and
  :func:`write_sparameters_stack`, cached as ``.npz`` by a hash of the geometry
  and settings; :func:`model_from_npz` reads them into SAX, modes kept.

Results describe an infinitely extended stack or grating under plane-wave
illumination (couplers: the leaky-mode model of a finite grating). Requires the
``gdsfactory`` extra and an active PDK.
"""

from hyperbolic_optics.integrations.gdsfactory.geometry import (
    GratingTeeth,
    LineGrating,
    extract_grating_teeth,
    extract_line_grating,
    line_grating,
)
from hyperbolic_optics.integrations.gdsfactory.lattice import (
    UnitCell,
    crossed_grating_layers,
    crossed_grating_stack,
    extract_unit_cell,
    periodic_array,
)
from hyperbolic_optics.integrations.gdsfactory.materials import (
    DEFAULT_MATERIALS,
    MaterialSpec,
    Oriented,
    material_tensor,
    resolve_material,
    x_cut,
    z_cut,
)
from hyperbolic_optics.integrations.gdsfactory.simulate import (
    CrossedGratingResponse,
    GratingResponse,
    PlanarResponse,
    simulate_component,
    simulate_crossed_grating,
    simulate_grating,
    simulate_stack,
    solve_crossed_grating,
    solve_grating,
    solve_stack,
    stack_sax_model,
)
from hyperbolic_optics.integrations.gdsfactory.sparameters import (
    model_from_npz,
    write_sparameters_grating_coupler,
    write_sparameters_stack,
)
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    grating_layers,
    grating_stack,
    patterned_stack,
    teeth_stack,
    vertical_profile,
    vertical_stack,
)

__all__ = [
    "CrossedGratingResponse",
    "DEFAULT_MATERIALS",
    "GratingResponse",
    "GratingTeeth",
    "LineGrating",
    "MaterialSpec",
    "Oriented",
    "PlanarResponse",
    "StackLayer",
    "UnitCell",
    "VerticalStack",
    "crossed_grating_layers",
    "crossed_grating_stack",
    "extract_grating_teeth",
    "extract_line_grating",
    "extract_unit_cell",
    "grating_layers",
    "grating_stack",
    "line_grating",
    "material_tensor",
    "model_from_npz",
    "patterned_stack",
    "periodic_array",
    "resolve_material",
    "simulate_component",
    "simulate_crossed_grating",
    "simulate_grating",
    "simulate_stack",
    "solve_crossed_grating",
    "solve_grating",
    "solve_stack",
    "stack_sax_model",
    "teeth_stack",
    "vertical_profile",
    "vertical_stack",
    "write_sparameters_grating_coupler",
    "write_sparameters_stack",
    "x_cut",
    "z_cut",
]
