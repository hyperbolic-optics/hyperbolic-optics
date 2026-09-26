"""gdsfactory bridge: PDK stacks and layouts into the solvers, gplugins-style.

Everything is importable from here::

    from hyperbolic_optics.integrations import gdsfactory as hog

- **Geometry** (:mod:`.geometry`): :func:`line_grating`, a grid-snapped PCell,
  and :func:`extract_line_grating`, which reads a line grating back from any
  component or GDS and refuses anything else.
- **Materials** (:mod:`.materials`): material specifications, :class:`Oriented`
  crystals with :func:`x_cut` / :func:`z_cut`, and the mapping from PDK material
  names (:data:`DEFAULT_MATERIALS`).
- **Stacks** (:mod:`.stack`): :class:`VerticalStack` and :class:`StackLayer`,
  written by hand or read from a PDK ``LayerStack`` at a point of a layout
  (:func:`vertical_stack`) or under a line grating (:func:`grating_stack`).
- **Simulation** (:mod:`.simulate`): :func:`simulate_stack` and
  :func:`simulate_grating` take a layout and PDK and return
  :class:`PlanarResponse` / :class:`GratingResponse`, with Jones, Mueller and
  ellipsometric analysis.

Results describe an infinitely extended stack or grating under plane-wave
illumination. Requires the ``gdsfactory`` extra and an active PDK.
"""

from hyperbolic_optics.integrations.gdsfactory.geometry import (
    LineGrating,
    extract_line_grating,
    line_grating,
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
    GratingResponse,
    PlanarResponse,
    simulate_component,
    simulate_grating,
    simulate_stack,
    solve_grating,
    solve_stack,
)
from hyperbolic_optics.integrations.gdsfactory.stack import (
    StackLayer,
    VerticalStack,
    grating_layers,
    grating_stack,
    vertical_profile,
    vertical_stack,
)

__all__ = [
    "DEFAULT_MATERIALS",
    "GratingResponse",
    "LineGrating",
    "MaterialSpec",
    "Oriented",
    "PlanarResponse",
    "StackLayer",
    "VerticalStack",
    "extract_line_grating",
    "grating_layers",
    "grating_stack",
    "line_grating",
    "material_tensor",
    "resolve_material",
    "simulate_component",
    "simulate_grating",
    "simulate_stack",
    "solve_grating",
    "solve_stack",
    "vertical_profile",
    "vertical_stack",
    "x_cut",
    "z_cut",
]
