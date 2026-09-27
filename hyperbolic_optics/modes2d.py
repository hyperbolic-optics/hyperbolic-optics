"""Full-vector modes of 2D waveguide cross-sections, anisotropic cores included.

A waveguide along z with a cross-section ``ε(x, y)``: its modes ``∝ exp(iβz)``.
The transverse fields obey, with lengths in units of ``1/k₀`` and μ = 1,

    β·[Ex, Ey] = M₁·[Hx, Hy],    β·[Hx, Hy] = M₂·[Ex, Ey],

(from Maxwell's curl equations with ``Ez`` and ``Hz`` eliminated), so that
``β²`` is an eigenvalue of ``M₁·M₂`` acting on ``[Ex, Ey]``. They are
discretised on a Yee grid in the cross-section -- ``Ex`` at ``(i+½, j)``,
``Ey`` at ``(i, j+½)``, ``Ez`` at ``(i, j)``, H dual -- with the domain's
boundary a perfect electric conductor, and each tensor component sampled at
its own field's position, averaged over the grid cell around it.

The permittivity may be anisotropic with its principal axes along x, y and z
(``εxx``, ``εyy``, ``εzz``): an x-cut LiNbO₃ rib along a crystal axis, a
z-cut one, a biaxial film. Off-diagonal terms (a waveguide at an angle to the
axes) are refused rather than approximated.

:func:`solve_modes` returns the modes nearest a target index, with their
fields; :func:`waveguide` rasterises a cross-section from rectangles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["CrossSection", "Mode", "rectangles", "solve_modes"]


class CrossSection(NamedTuple):
    """A sampled waveguide cross-section.

    Attributes:
        x_um, y_um: Cell edges ``[Nx + 1]``, ``[Ny + 1]`` (uniform spacing).
        eps: ``(εxx, εyy, εzz)``, each sampled at its own Yee position:
            ``[Nx, Ny]`` arrays (``εxx`` at ``(i+½, j)``, ``εyy`` at
            ``(i, j+½)``, ``εzz`` at ``(i, j)`` with ``i`` counting x edges).
    """

    x_um: np.ndarray
    y_um: np.ndarray
    eps: tuple[np.ndarray, np.ndarray, np.ndarray]


class Mode(NamedTuple):
    """One guided mode of a cross-section.

    Attributes:
        effective_index: ``β/k₀``.
        fields: ``(Ex, Ey, Ez, Hx, Hy, Hz)`` averaged onto the grid nodes
            ``(xᵢ, yⱼ)``, ``[Nx, Ny]`` each, normalised to unit power along z
            (``½ Re ∫ (Ex·Hy* − Ey·Hx*) dA = 1``, H scaled by the free-space
            impedance, lengths in µm).
        te_fraction: ``∫|Ex|² / ∫(|Ex|² + |Ey|²)``: near 1 for a TE-like
            (x-polarized) mode, near 0 for a TM-like one.
    """

    effective_index: complex
    fields: tuple
    te_fraction: float


def rectangles(
    x_um: np.ndarray,
    y_um: np.ndarray,
    background,
    boxes: Sequence[tuple[tuple[float, float, float, float], object]],
    subsamples: int = 4,
) -> CrossSection:
    """A cross-section from axis-aligned boxes over a background.

    Args:
        x_um, y_um: Cell edges (uniform).
        background: ``ε`` of the background (number or ``(εxx, εyy, εzz)``).
        boxes: ``((x0, x1, y0, y1), ε)``, later boxes over earlier ones.
        subsamples: Samples per cell side for the cell average at each Yee
            position.
    """

    def diagonal(value):
        value = np.asarray(value, dtype=complex)
        if value.ndim == 0:
            return np.full(3, value)
        if value.shape == (3, 3):
            if np.any(np.abs(value - np.diag(np.diag(value))) > 1e-12 * np.abs(value).max()):
                raise ValueError(
                    "modes2d takes tensors diagonal in the waveguide's frame (x, y across, "
                    "z along it); this one has off-diagonal terms"
                )
            return np.diag(value)
        return value

    background = diagonal(background)
    boxes = [(bounds, diagonal(value)) for bounds, value in boxes]

    def eps_at(px, py, component):
        out = np.full(np.broadcast(px, py).shape, background[component], dtype=complex)
        for (x0, x1, y0, y1), value in boxes:
            inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            out = np.where(inside, value[component], out)
        return out

    return sampled(x_um, y_um, eps_at, subsamples)


def sampled(x_um, y_um, eps_at: Callable, subsamples: int = 4) -> CrossSection:
    """A cross-section from ``eps_at(x, y, component)`` (component 0, 1, 2 = xx, yy, zz)."""
    x, y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)
    dx, dy = x[1] - x[0], y[1] - y[0]
    nx, ny = len(x) - 1, len(y) - 1
    offsets = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    # Yee positions (cell index i, j): Ex (i+½, j), Ey (i, j+½), Ez (i, j), on
    # the grid of cell corners x[i], y[j]; the field arrays are [Nx, Ny]
    positions = {
        0: (x[:nx] + dx / 2, y[:ny]),
        1: (x[:nx], y[:ny] + dy / 2),
        2: (x[:nx], y[:ny]),
    }
    eps = []
    for component, (cx, cy) in positions.items():
        px = cx[:, None, None, None] + dx * offsets[None, None, :, None]
        py = cy[None, :, None, None] + dy * offsets[None, None, None, :]
        values = eps_at(px, py, component)
        eps.append(values.mean(axis=(-2, -1)))
    return CrossSection(x_um=x, y_um=y, eps=tuple(eps))


def _differences(n: int, step: float):
    """Forward difference ``[n, n]`` with the field zero beyond the last point (PEC)."""
    main = -np.ones(n)
    upper = np.ones(n - 1)
    forward = sp.diags([main, upper], [0, 1], shape=(n, n)) / step
    return forward.tocsr()


def solve_modes(
    cross_section: CrossSection,
    wavelength_um: float,
    n_modes: int = 4,
    target_index: float | None = None,
) -> list[Mode]:
    """The guided modes of a cross-section nearest ``target_index``.

    Args:
        cross_section: From :func:`rectangles` or :func:`sampled`.
        wavelength_um: Vacuum wavelength.
        n_modes: How many modes to return.
        target_index: Look for effective indices near this; by default the
            highest index in the cross-section.

    Returns:
        :class:`Mode` s, highest effective index first.
    """
    x, y = cross_section.x_um, cross_section.y_um
    exx, eyy, ezz = (np.asarray(e, dtype=complex) for e in cross_section.eps)
    nx, ny = exx.shape
    k0 = 2 * np.pi / wavelength_um
    dx, dy = (x[1] - x[0]) * k0, (y[1] - y[0]) * k0  # in units of 1/k₀

    ix, iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    dxf = sp.kron(_differences(nx, dx), iy, format="csr")  # d/dx, flattened C-order (i, j)
    dyf = sp.kron(ix, _differences(ny, dy), format="csr")
    dxb, dyb = -dxf.T.tocsr(), -dyf.T.tocsr()  # the adjoints: backward differences
    eye = sp.identity(nx * ny, format="csr")

    def diag(values):
        return sp.diags(values.ravel())

    inv_zz = diag(1.0 / ezz)
    m1 = sp.bmat(
        [
            [-dxf @ inv_zz @ dyb, eye + dxf @ inv_zz @ dxb],
            [-eye - dyf @ inv_zz @ dyb, dyf @ inv_zz @ dxb],
        ]
    )  # fmt: skip  -- [Ex, Ey] = M₁ [Hx, Hy] / β
    m2 = sp.bmat(
        [[dxb @ dyf, -diag(eyy) - dxb @ dxf], [diag(exx) + dyb @ dyf, -dyb @ dxf]]
    )  # fmt: skip  -- [Hx, Hy] = M₂ [Ex, Ey] / β
    operator = (m1 @ m2).tocsc()

    if target_index is None:
        target_index = float(
            np.sqrt(np.max(np.real(np.concatenate([e.ravel() for e in (exx, eyy)]))))
        )
    shift = target_index**2
    values, vectors = spla.eigs(operator, k=n_modes, sigma=shift, which="LM")
    order = np.argsort(-np.real(values))

    area = (x[1] - x[0]) * (y[1] - y[0])
    modes = []
    for index in order:
        beta = np.sqrt(values[index])
        e_t = vectors[:, index]
        h_t = (m2 @ e_t) / beta
        ex, ey = e_t[: nx * ny].reshape(nx, ny), e_t[nx * ny :].reshape(nx, ny)
        hx, hy = h_t[: nx * ny].reshape(nx, ny), h_t[nx * ny :].reshape(nx, ny)
        ez = (1j * (inv_zz @ (dxb @ hy.ravel() - dyb @ hx.ravel()))).reshape(nx, ny)
        hz = (-1j * (dxf @ ey.ravel() - dyf @ ex.ravel())).reshape(nx, ny)
        # power along z, from the transverse fields at their own positions (Ex with Hy,
        # both at (i+½, j); Ey with Hx, both at (i, j+½))
        power = 0.5 * np.real(np.sum(ex * np.conj(hy) - ey * np.conj(hx))) * area
        scale = 1 / np.sqrt(abs(power))
        fields = tuple(_to_nodes(f * scale, component) for component, f in
                       enumerate((ex, ey, ez, hx, hy, hz)))  # fmt: skip
        te = float(np.sum(np.abs(ex) ** 2) / np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
        modes.append(Mode(effective_index=complex(beta), fields=fields, te_fraction=te))
    return modes


def _to_nodes(field: np.ndarray, component: int) -> np.ndarray:
    """A Yee field ``[Nx, Ny]`` averaged onto the nodes ``(xᵢ, yⱼ)`` (the ``Ez`` points).

    Components (Ex, Ey, Ez, Hx, Hy, Hz) sit at (i+½, j), (i, j+½), (i, j),
    (i, j+½), (i+½, j), (i+½, j+½); each half-step is averaged with its
    neighbour (the field is zero beyond the boundary).
    """

    def half(values, axis):
        shifted = np.zeros_like(values)
        if axis == 0:
            shifted[1:] = values[:-1]
        else:
            shifted[:, 1:] = values[:, :-1]
        return 0.5 * (values + shifted)

    half_x, half_y = {0: (1, 0), 1: (0, 1), 2: (0, 0), 3: (0, 1), 4: (1, 0), 5: (1, 1)}[component]
    out = field
    if half_x:
        out = half(out, 0)
    if half_y:
        out = half(out, 1)
    return out
