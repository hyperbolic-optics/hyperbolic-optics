"""Sweep a single layer's thickness as an extra output dimension.

The canonical engine sweeps incident angle, azimuth, and frequency as the batch
axes ``[A, B, F]`` (see :mod:`hyperbolic_optics.axes`). A layer's *thickness* is a
per-layer property, not a scenario axis, so this helper sweeps it the pragmatic
way: re-run the structure once per thickness and stack the results along a new
**leading** index. Each run is cheap, and the output gains a thickness axis in
front of the usual presentation shape — e.g. an ``Incident`` ``r_pp`` of shape
``(F, angle)`` becomes ``(n_thickness, F, angle)``.

This keeps the core convention untouched (no golden re-baseline). If you ever need
thickness as a *vectorized* batch axis alongside angle/azimuth/frequency, that is
a separate, larger change to the canonical layout.

Example:
    >>> payload = {
    ...     "ScenarioData": {"type": "Incident"},
    ...     "Layers": [
    ...         {"type": "Ambient Incident Layer", "permittivity": 50.0},
    ...         {"type": "Crystal Layer", "material": "Calcite",
    ...          "thickness": 1.0, "rotationY": 90},
    ...         {"type": "Semi Infinite Isotropic Layer", "permittivity": 1.0},
    ...     ],
    ... }
    >>> sweep = ThicknessSweep(payload, layer_index=1, thicknesses=[0.5, 1.0, 2.0])
    >>> sweep.r_pp.shape          # (3, F, angle)
    >>> sweep.transmittance("p")  # (3, F, angle)
    >>> sweep.layer_absorption("p")  # per-interior-layer, each (3, F, angle)
"""

import copy
from typing import Any

import numpy as np

from hyperbolic_optics.fields import FieldProfile
from hyperbolic_optics.structure import Structure

# Layer types that carry a finite, sweepable thickness.
_THICKNESS_LAYERS = {"Isotropic Middle-Stack Layer", "Crystal Layer"}


class ThicknessSweep:
    """Run a payload across a range of thicknesses for one layer and stack outputs.

    The reflection coefficients and every :class:`~hyperbolic_optics.fields.FieldProfile`
    power quantity are exposed with a leading ``thickness`` axis. Thicknesses are
    in microns, matching the ``thickness`` field in a layer payload.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        layer_index: int,
        thicknesses: np.ndarray | list[float],
    ) -> None:
        """Execute ``payload`` once per thickness for the chosen layer.

        Args:
            payload: A complete ``Structure.execute`` payload.
            layer_index: Index into ``payload["Layers"]`` of the layer to sweep;
                must be a finite layer (``Crystal Layer`` or ``Isotropic
                Middle-Stack Layer``).
            thicknesses: Thicknesses to sweep, in microns.

        Raises:
            IndexError: If ``layer_index`` is out of range.
            ValueError: If the selected layer has no sweepable thickness, or
                ``thicknesses`` is empty.
        """
        layers = payload.get("Layers", [])
        if not 0 <= layer_index < len(layers):
            raise IndexError(f"layer_index {layer_index} out of range (0..{len(layers) - 1}).")
        layer_type = layers[layer_index].get("type")
        if layer_type not in _THICKNESS_LAYERS:
            raise ValueError(
                f"Layer {layer_index} ({layer_type!r}) has no sweepable thickness; "
                f"choose a finite layer of type {sorted(_THICKNESS_LAYERS)}."
            )
        self.thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=np.float64))
        if self.thicknesses.size == 0:
            raise ValueError("thicknesses must contain at least one value.")

        self.layer_index = layer_index
        self.layer_type = layer_type
        self.structures: list[Structure] = []
        for thickness in self.thicknesses:
            single = copy.deepcopy(payload)
            single["Layers"][layer_index]["thickness"] = float(thickness)
            structure = Structure()
            structure.execute(single)
            self.structures.append(structure)
        self._fields = [FieldProfile(structure) for structure in self.structures]

    def __len__(self) -> int:
        """Number of thicknesses swept."""
        return len(self.thicknesses)

    @staticmethod
    def _stack(values: list[np.ndarray]) -> np.ndarray:
        """Stack per-thickness arrays along a new leading axis."""
        return np.stack([np.asarray(value) for value in values], axis=0)

    # -- reflection coefficients ---------------------------------------------

    @property
    def r_pp(self) -> np.ndarray:
        """Stacked p->p reflection coefficient, shape ``(n_thickness, *r_pp)``."""
        return self._stack([s.r_pp for s in self.structures])

    @property
    def r_ss(self) -> np.ndarray:
        """Stacked s->s reflection coefficient."""
        return self._stack([s.r_ss for s in self.structures])

    @property
    def r_ps(self) -> np.ndarray:
        """Stacked p->s reflection coefficient."""
        return self._stack([s.r_ps for s in self.structures])

    @property
    def r_sp(self) -> np.ndarray:
        """Stacked s->p reflection coefficient."""
        return self._stack([s.r_sp for s in self.structures])

    # -- power quantities (via FieldProfile) ---------------------------------

    def reflectance(self, polarization: str | tuple[complex, complex] = "p") -> np.ndarray:
        """Stacked power reflectance ``R`` vs thickness."""
        return self._stack([f.reflectance(polarization) for f in self._fields])

    def transmittance(self, polarization: str | tuple[complex, complex] = "p") -> np.ndarray:
        """Stacked power transmittance ``T`` vs thickness."""
        return self._stack([f.transmittance(polarization) for f in self._fields])

    def total_absorption(self, polarization: str | tuple[complex, complex] = "p") -> np.ndarray:
        """Stacked total interior absorption vs thickness."""
        return self._stack([f.summary(polarization)["total_absorption"] for f in self._fields])

    def layer_absorption(
        self, polarization: str | tuple[complex, complex] = "p"
    ) -> list[dict[str, Any]]:
        """Per-interior-layer absorptance, each stacked over thickness.

        Returns a list (top->bottom) of ``{"index", "type", "absorptance"}`` where
        ``absorptance`` has a leading thickness axis. Empty when there are no finite
        interior layers. The swept layer's own absorptance varies with thickness.
        """
        per_run = [f.layer_absorption(polarization) for f in self._fields]
        stacked = []
        for entries in zip(*per_run, strict=True):
            stacked.append(
                {
                    "index": entries[0]["index"],
                    "type": entries[0]["type"],
                    "absorptance": self._stack([e["absorptance"] for e in entries]),
                }
            )
        return stacked

    def transmission_coefficients(self) -> dict[str, np.ndarray]:
        """Amplitude transmission coefficients, each stacked over thickness."""
        per_run = [f.transmission_coefficients() for f in self._fields]
        return {key: self._stack([run[key] for run in per_run]) for key in per_run[0]}
