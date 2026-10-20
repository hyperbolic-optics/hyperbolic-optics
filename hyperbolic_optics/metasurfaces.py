"""Metasurface design by local periodicity: cell libraries, phase maps, focusing.

Each cell transmits as it would in an infinite array of itself, so its
zeroth-order Jones matrix from :mod:`hyperbolic_optics.rcwa2d` stands for it.
Layouts: :func:`hyperbolic_optics.integrations.gdsfactory.metasurface_component`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import bcm, rcwa2d

__all__ = [
    "cell_library",
    "choose_cells",
    "focus",
    "geometric_phase_jones",
    "lens_phase",
    "sample_grid",
    "square_sites",
]


def cell_library(
    cell: Callable,
    parameters,
    lattice,
    wavelength_um: float,
    eps_incident: float,
    eps_exit: float,
    max_order: tuple[int, int] = (4, 4),
    batch_size: int | None = None,
    **kwargs,
) -> jax.Array:
    """Zeroth-order transmission Jones matrices of a batch of cells, at normal incidence.

    Args:
        cell: ``parameters → layers`` (:class:`~hyperbolic_optics.rcwa2d.PatternedLayer` s
            from the incident side down), written in ``jax.numpy``. Every entry of
            the batch must give the same structure (layers, shapes, vertex
            counts); pad with zero-thickness layers to vary the number of layers.
        parameters: A batch ``[P, ...]`` (an array, or a pytree of arrays with a
            common leading axis).
        lattice: A :class:`~hyperbolic_optics.rcwa2d.Lattice` or ``(Λx, Λy)``.
        batch_size: Cells solved at once (``jax.lax.map``); all at once if omitted.
        wavelength_um, eps_incident, eps_exit, max_order, **kwargs: As in
            :func:`hyperbolic_optics.rcwa2d.solve_crossed_grating`.

    Returns:
        ``t`` ``[P, 2, 2]`` (``[out, in]``, x/p and y/s), whose
        ``|t|²·n_exit/n_incident`` is the transmitted power. Differentiable in
        ``parameters``.
    """

    def one(p):
        result = rcwa2d.solve_crossed_grating(
            cell(p), lattice, wavelength_um, 0.0, eps_incident, eps_exit, max_order, **kwargs
        )
        zeroth = int(np.flatnonzero((result.orders == 0).all(axis=1))[0])
        return result.t[zeroth]

    parameters = jax.tree.map(jnp.asarray, parameters)
    size = jax.tree.leaves(parameters)[0].shape[0]
    return jax.lax.map(jax.jit(one), parameters, batch_size=batch_size or size)


def choose_cells(target_phase, library_t, polarization=(1.0, 0.0), analyzer=None):
    """The library entry at each site that best realises the target phase.

    Chooses ``argmax Re(a† t p · e^{−iφ})``: the cell whose transmitted field,
    projected on the analyzer ``a`` (by default the incident ``p``), is largest
    in phase with the target. Returns ``(indices [...], realised amplitude [...])``.
    """
    p = np.asarray(polarization, dtype=complex)
    a = p if analyzer is None else np.asarray(analyzer, dtype=complex)
    amplitude = np.einsum("i,kij,j->k", np.conj(a), np.asarray(library_t), p)
    target = np.exp(1j * np.asarray(target_phase))
    score = np.real(amplitude[None, :] * np.conj(target).reshape(-1, 1))
    indices = np.argmax(score, axis=1).reshape(np.shape(target_phase))
    return indices, amplitude[indices]


def geometric_phase_jones(jones, angle_rad):
    """Ideal normal-incidence rotation ``R·J·Rᵀ``, with the lattice turning too.

    Turning only a shape in a fixed lattice generally requires a new cell solve.
    """
    rotation = bcm.rotation_matrix(0.0, 0.0, angle_rad)[..., :2, :2]
    return rotation @ jnp.asarray(jones) @ jnp.swapaxes(rotation, -1, -2)


def square_sites(aperture_radius_um: float, pitch_um: float) -> np.ndarray:
    """Cell centres ``[K, 2]`` of a square lattice inside a circular aperture about 0."""
    n = int(np.floor(aperture_radius_um / pitch_um))
    grid = np.arange(-n, n + 1) * pitch_um
    x, y = np.meshgrid(grid, grid, indexing="ij")
    inside = x**2 + y**2 <= aperture_radius_um**2
    return np.stack([x[inside], y[inside]], axis=-1)


def lens_phase(x_um, y_um, focal_um: float, wavelength_um: float, n_exit: float = 1.0):
    """``φ = −(2π n/λ)(√(x² + y² + f²) − f)``: a plane wave focused at ``f``."""
    k = 2 * np.pi * n_exit / wavelength_um
    return -k * (jnp.sqrt(jnp.asarray(x_um) ** 2 + jnp.asarray(y_um) ** 2 + focal_um**2) - focal_um)


def sample_grid(sites_um, values, pitch_um: float, subsamples: int = 3, pad: int = 3):
    """Site values on a square grid, each cell ``subsamples²`` pixels, zero-padded.

    Returns ``(coordinates [N], field [N, N])`` with ``N = pad·(2n + 1)·subsamples``
    for sites within ``±n`` pitches of the origin. Differentiable in ``values``.
    """
    index = np.round(np.asarray(sites_um) / pitch_um).astype(int)
    n = int(np.abs(index).max())
    cells = jnp.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
    cells = cells.at[index[:, 0] + n, index[:, 1] + n].set(values)
    field = jnp.kron(cells, jnp.ones((subsamples, subsamples)))
    size = field.shape[0]
    margin = (pad - 1) * size // 2
    field = jnp.pad(field, margin)
    coordinates = (np.arange(field.shape[0]) - margin - (size - 1) / 2) * pitch_um / subsamples
    return coordinates, field


def focus(x_um, y_um, field, wavelength_um: float, distance_um: float, n_exit: float = 1.0):
    """A field on a uniform grid ``[N, M]`` propagated by ``distance_um`` (angular spectrum)."""
    x, y = np.asarray(x_um), np.asarray(y_um)
    kx = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
    ky = 2 * np.pi * np.fft.fftfreq(len(y), y[1] - y[0])
    k = 2 * np.pi * n_exit / wavelength_um
    kz = jnp.sqrt(k**2 - kx[:, None] ** 2 - ky[None, :] ** 2 + 0j)
    return jnp.fft.ifft2(jnp.fft.fft2(field) * jnp.exp(1j * kz * distance_um))
