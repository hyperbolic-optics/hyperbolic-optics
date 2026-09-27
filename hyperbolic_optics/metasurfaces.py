"""Metasurface design: unit-cell libraries, lens phase maps, and the layout.

The local-periodicity approach: each unit cell of a slowly varying
metasurface transmits as it would in an infinite array of itself, so a
library of cells -- their zeroth-order transmission Jones matrices against a
geometric parameter, from :mod:`hyperbolic_optics.rcwa2d` -- lets a target
phase map be realised cell by cell.

- :func:`cell_library`: ``t(parameter)`` ``[P, 2, 2]`` (``[out, in]``, 0 = x/p,
  1 = y/s at normal incidence) for any parametrised unit cell: pillar
  radius, fin width, rotation, or anything else. Anisotropic pillars
  (birefringent crystals, form-birefringent fins) give non-diagonal or
  unequal-diagonal Jones matrices.
- :func:`square_sites`: cell centres inside a circular aperture.
- :func:`lens_phase`: the hyperbolic phase of a lens focusing a normally
  incident plane wave at ``f``.
- :func:`choose_cells`: the parameter at each site whose transmission best
  matches the target phase (the amplitude, as a weight).
- :func:`geometric_phase_jones`: the Jones matrix of a cell turned by ``θ``,
  whose circular-polarization conversion carries the Pancharatnam–Berry phase
  ``±2θ``.
- :func:`focus`: the designed transmission propagated to the focal plane by
  the angular spectrum, a self-consistency check on the design (not a
  substitute for a full-wave simulation of the finite metasurface).

The layout itself is drawn by
:func:`hyperbolic_optics.integrations.gdsfactory.metasurface_component`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import rcwa2d

__all__ = [
    "cell_library",
    "disc_cell",
    "choose_cells",
    "focus",
    "geometric_phase_jones",
    "lens_phase",
    "square_sites",
]


def cell_library(
    cell: Callable[[float], list],
    parameters: Sequence[float],
    lattice,
    wavelength_um: float,
    eps_incident: float,
    eps_exit: float,
    max_order: tuple[int, int] = (4, 4),
    **kwargs,
) -> np.ndarray:
    """Zeroth-order transmission Jones matrices of a family of unit cells.

    Args:
        cell: ``parameter → layers`` (:class:`~hyperbolic_optics.rcwa2d.PatternedLayer` s
            from the incident side down).
        parameters: The parameter values ``[P]``.
        lattice: The lattice (a :class:`~hyperbolic_optics.rcwa2d.Lattice` or ``(Λx, Λy)``).
        wavelength_um, eps_incident, eps_exit, max_order, **kwargs: As in
            :func:`hyperbolic_optics.rcwa2d.solve_crossed_grating`, at normal incidence.

    Returns:
        ``t`` ``[P, 2, 2]`` (``[out, in]``, p = x and s = y at normal
        incidence): complex amplitudes, whose ``|t|²·(n_exit/n_in)`` is the
        transmitted power.
    """
    out = []
    for parameter in parameters:
        result = rcwa2d.solve_crossed_grating(
            cell(parameter), lattice, wavelength_um, 0.0, eps_incident, eps_exit, max_order,
            **kwargs,
        )  # fmt: skip
        zeroth = int(np.flatnonzero((result.orders == 0).all(axis=1))[0])
        out.append(np.asarray(result.t[zeroth]))
    return np.stack(out)


def square_sites(aperture_radius_um: float, pitch_um: float) -> np.ndarray:
    """Cell centres ``[K, 2]`` of a square lattice inside a circular aperture, centred on 0."""
    n = int(np.floor(aperture_radius_um / pitch_um))
    grid = np.arange(-n, n + 1) * pitch_um
    x, y = np.meshgrid(grid, grid, indexing="ij")
    inside = x**2 + y**2 <= aperture_radius_um**2
    return np.stack([x[inside], y[inside]], axis=-1)


def lens_phase(x_um, y_um, focal_um: float, wavelength_um: float, n_exit: float = 1.0):
    """``φ(x, y) = −(2π n/λ)(√(x² + y² + f²) − f)``: a plane wave focused at ``f``."""
    k = 2 * np.pi * n_exit / wavelength_um
    return -k * (np.sqrt(np.asarray(x_um) ** 2 + np.asarray(y_um) ** 2 + focal_um**2) - focal_um)


def geometric_phase_jones(jones: np.ndarray, angle_rad) -> np.ndarray:
    """The Jones matrix of a cell turned by ``angle_rad`` about the normal: ``R·J·Rᵀ``.

    For circular input, the converted (opposite-handedness) output of a
    turned cell carries the Pancharatnam–Berry phase ``±2θ``.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.moveaxis(np.array([[c, -s], [s, c]]), (0, 1), (-2, -1))
    return rotation @ np.asarray(jones) @ np.swapaxes(rotation, -1, -2)


def choose_cells(target_phase, library_t: np.ndarray, polarization=(1.0, 0.0), analyzer=None):
    """The library index at each site that best realises the target phase.

    Chooses ``argmax Re(a† t p · e^{−iφ})`` -- the cell whose transmitted
    field, projected on the analyzer, is largest in phase with the target.

    Args:
        target_phase: ``φ`` at each site ``[...]``.
        library_t: :func:`cell_library` ``[P, 2, 2]``.
        polarization: The incident Jones vector ``p``.
        analyzer: The detected polarization ``a``; by default ``p``.

    Returns:
        ``(indices [...], realised complex transmission [...])``.
    """
    p = np.asarray(polarization, dtype=complex)
    a = p if analyzer is None else np.asarray(analyzer, dtype=complex)
    amplitude = np.einsum("i,kij,j->k", np.conj(a), np.asarray(library_t), p)  # [P]
    target = np.exp(1j * np.asarray(target_phase))
    score = np.real(amplitude[None, :] * np.conj(target).reshape(-1, 1))
    indices = np.argmax(score, axis=1).reshape(np.shape(target_phase))
    return indices, amplitude[indices]


def focus(x_um, y_um, transmission, wavelength_um: float, distance_um: float, n_exit: float = 1.0):
    """Propagate a field on a square grid by the angular spectrum (scalar).

    Args:
        x_um, y_um: Sample coordinates ``[N]``, ``[M]`` (uniform).
        transmission: The field just after the metasurface ``[N, M]``.
        wavelength_um, n_exit: The medium the light propagates in.
        distance_um: How far.

    Returns:
        The field at ``distance_um`` ``[N, M]``.
    """
    x, y = np.asarray(x_um), np.asarray(y_um)
    kx = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
    ky = 2 * np.pi * np.fft.fftfreq(len(y), y[1] - y[0])
    k = 2 * np.pi * n_exit / wavelength_um
    kz = np.sqrt(k**2 - kx[:, None] ** 2 - ky[None, :] ** 2 + 0j)
    return np.fft.ifft2(np.fft.fft2(transmission) * np.exp(1j * kz * distance_um))


def disc_cell(eps_pillar, height_um, lattice_constant_um, eps_background=1.0, n_vertices=48):
    """A convenience ``radius → layers`` for :func:`cell_library`: one disc per cell."""
    eye = jnp.eye(3, dtype=complex)
    pillar = jnp.asarray(eps_pillar) * (eye if np.ndim(eps_pillar) == 0 else 1)

    def cell(radius):
        centre = (lattice_constant_um / 2, lattice_constant_um / 2)
        shape = rcwa2d.Shape(rcwa2d.ellipse(centre, (radius, radius), n_vertices), pillar)
        return [rcwa2d.PatternedLayer(height_um, eps_background * eye, (shape,))]

    return cell
