"""gdsfactory bridge, imported as ``hog``: PDK stacks, periodic layouts and metasurfaces.

Needs the ``gdsfactory`` extra and an active PDK; see the gdsfactory user guide.
"""

from hyperbolic_optics.integrations.gdsfactory.geometry import (
    LineGrating,
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
    x_cut,
    z_cut,
)
from hyperbolic_optics.integrations.gdsfactory.metasurface import (
    cell_from_pdk,
    cell_layers,
    cell_sax_model,
    metasurface_component,
    read_metasurface,
    turned,
)
from hyperbolic_optics.integrations.gdsfactory.simulate import (
    CrossedGratingResponse,
    GratingResponse,
    PlanarResponse,
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
    write_sparameters_cell,
    write_sparameters_stack,
)
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    grating_layers,
    grating_stack,
    vertical_stack,
)

__all__ = [
    "CrossedGratingResponse",
    "DEFAULT_MATERIALS",
    "GratingResponse",
    "LineGrating",
    "MaterialSpec",
    "Oriented",
    "PlanarResponse",
    "StackLayer",
    "UnitCell",
    "cell_layers",
    "cell_from_pdk",
    "cell_sax_model",
    "VerticalStack",
    "crossed_grating_layers",
    "crossed_grating_stack",
    "extract_line_grating",
    "extract_unit_cell",
    "grating_layers",
    "grating_stack",
    "line_grating",
    "material_tensor",
    "metasurface_component",
    "model_from_npz",
    "periodic_array",
    "read_metasurface",
    "simulate_crossed_grating",
    "simulate_grating",
    "simulate_stack",
    "solve_crossed_grating",
    "solve_grating",
    "solve_stack",
    "stack_sax_model",
    "turned",
    "vertical_stack",
    "write_sparameters_stack",
    "write_sparameters_cell",
    "x_cut",
    "z_cut",
]
