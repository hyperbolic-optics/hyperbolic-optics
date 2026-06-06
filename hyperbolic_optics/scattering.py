"""Numerically-stable scattering-matrix backend.

The default engine multiplies per-layer *transfer* matrices ``Mᵢ = Vᵢ·Pᵢ·Vᵢ⁻¹``
with ``Pᵢ = diag(exp(-i·kz·k₀·d))`` (``waves.get_matrix``). For thick / lossy /
strongly-evanescent layers the propagation diagonal mixes exponentially *growing*
and *decaying* terms, so the products overflow and lose precision (energy
conservation drifts, eventually ``NaN``).

This module reformulates the same 4×4 anisotropic problem as a **scattering
matrix** cascaded with the **Redheffer star product**, where every propagation
factor is a *decaying* exponential (``exp(-|Im(kz)|·k₀·d) ≤ 1``) — the growing
matrices are never materialised — so it is stable in all cases. It is the
opt-in backend behind ``Structure.execute(payload, backend="scattering")`` and
produces the same reflection/transmission coefficients as the transfer method
where both are well-conditioned, and *correct* coefficients where the transfer
method fails.

Algorithm and conventions follow PyLlama (Bay, Lafait & Lequime / Bay et al.,
J. Opt. Soc. Am. A 39, 1431 (2022); arXiv:2012.05945), eqs 18-28, adapted to this
codebase's field-vector ordering ``[Ex, Ey, Hx, Hy]`` and partial-wave sort
(``WaveProfile.tangential_modes``: columns ``[t0, t1, r0, r1]`` = forward,
forward, backward, backward). Per-medium eigenvectors come from each layer's
``profile`` (interior layers and a semi-infinite crystal exit) or from the
ambient dynamical matrices ``A_inc`` / ``A_exit`` (prism, isotropic exit).
"""

import numpy as np

from hyperbolic_optics.axes import assert_canonical

# Ambient (isotropic) dynamical matrices have columns [s_fwd, s_bwd, p_fwd,
# p_bwd]; reorder to the forward/backward grouping [p_fwd, s_fwd, p_bwd, s_bwd]
# used here (forward = columns 0,1; backward = columns 2,3).
_AMBIENT_REORDER = [2, 0, 3, 1]


def _diag(values: np.ndarray) -> np.ndarray:
    """Build batched diagonal 4×4 matrices from ``values`` shaped ``[..., 4]``."""
    out = np.zeros(values.shape + (4,), dtype=np.complex128)
    idx = np.arange(4)
    out[..., idx, idx] = values
    return out


def _medium(layer) -> tuple[np.ndarray, np.ndarray | None, object]:
    """Return ``(P, q, d)`` for a layer: eigenvectors, propagation kz, thickness.

    ``q``/``d`` are ``None`` for half-spaces (prism, exit), whose propagation is
    the identity. Columns of ``P`` are grouped forward (0,1) / backward (2,3).
    """
    profile = getattr(layer, "profile", None)
    if profile is not None and layer.thickness is not None:
        # interior finite layer (crystal or isotropic air gap)
        eigenvectors, kz = profile.tangential_modes()
        return eigenvectors, kz, layer.thickness
    if profile is not None:
        # semi-infinite anisotropic exit: half-space, no propagation
        eigenvectors, _ = profile.tangential_modes()
        return eigenvectors, None, None
    # ambient isotropic half-space: prism.matrix = A_inc^{-1}; exit.matrix = A_exit.
    matrix = layer.matrix
    dynamical = np.linalg.inv(matrix) if layer.type == "Ambient Incident Layer" else matrix
    return dynamical[..., :, _AMBIENT_REORDER], None, None


def _interface_scattering(
    p_left: np.ndarray,
    q_left: np.ndarray | None,
    d_left: object,
    p_right: np.ndarray,
    k_0: np.ndarray,
) -> np.ndarray:
    """Scattering matrix ``S`` between two media (PyLlama eqs 19-23).

    ``[E_{right,→}; E_{left,←}] = S · [E_{left,→}; E_{right,←}]``. The propagation
    of the *left* medium is applied as decaying exponentials only — the forward
    factor ``exp(+i·q·k₀·d)`` (forward modes have ``Im(q) > 0``) and the backward
    *inverse* ``exp(-i·q·k₀·d)`` (backward modes have ``Im(q) < 0``) are both ≤ 1,
    and the growing matrices are never formed.
    """
    p_left, p_right = np.broadcast_arrays(p_left, p_right)
    # P_out = [p_left,0, p_left,1, -p_right,2, -p_right,3]
    p_out = np.stack(
        [p_left[..., :, 0], p_left[..., :, 1], -p_right[..., :, 2], -p_right[..., :, 3]], axis=-1
    )
    # P_in = [p_right,0, p_right,1, -p_left,2, -p_left,3]
    p_in = np.stack(
        [p_right[..., :, 0], p_right[..., :, 1], -p_left[..., :, 2], -p_left[..., :, 3]], axis=-1
    )

    core = np.linalg.inv(p_in) @ p_out
    if q_left is None:  # half-space: identity propagation
        return core

    k0_modes = k_0[..., np.newaxis]  # add a trailing mode axis -> [..., 1, 1]
    thickness = d_left[..., np.newaxis] if np.ndim(d_left) > 0 else d_left
    arg = q_left * k0_modes * thickness  # [..., 4]
    ones = np.ones_like(arg[..., 0])
    # Exponentiate only the modes that decay in their natural direction so the
    # growing exponentials are never even formed: forward factor exp(+i·arg) for
    # the forward modes (Im q > 0) and the backward *inverse* exp(-i·arg) for the
    # backward modes (Im q < 0). Both are bounded by 1.
    forward = np.exp(1j * arg[..., :2])
    backward_inv = np.exp(-1j * arg[..., 2:])
    q_forward = _diag(np.stack([forward[..., 0], forward[..., 1], ones, ones], axis=-1))
    q_backward_inv = _diag(
        np.stack([ones, ones, backward_inv[..., 0], backward_inv[..., 1]], axis=-1)
    )
    return q_backward_inv @ core @ q_forward


def _star(s_a: np.ndarray, s_b: np.ndarray) -> np.ndarray:
    """Redheffer star product of two scattering matrices (PyLlama eqs 25-26)."""
    s00a, s01a, s10a, s11a = s_a[..., :2, :2], s_a[..., :2, 2:], s_a[..., 2:, :2], s_a[..., 2:, 2:]
    s00b, s01b, s10b, s11b = s_b[..., :2, :2], s_b[..., :2, 2:], s_b[..., 2:, :2], s_b[..., 2:, 2:]
    identity = np.eye(2, dtype=np.complex128)
    c_inv = np.linalg.inv(identity - s01a @ s10b)
    s00 = s00b @ c_inv @ s00a
    s01 = s01b + s00b @ c_inv @ s01a @ s11b
    s10 = s10a + s11a @ s10b @ c_inv @ s00a
    s11 = s11a @ s11b + s11a @ s10b @ c_inv @ s01a @ s11b
    top = np.concatenate([s00, s01], axis=-1)
    bottom = np.concatenate([s10, s11], axis=-1)
    return np.concatenate([top, bottom], axis=-2)


def scattering_coefficients(layers: list, k_0: np.ndarray) -> dict[str, np.ndarray]:
    """Reflection/transmission coefficients via the stable scattering-matrix method.

    Args:
        layers: The executed structure's layers (``structure.layers``), each
            exposing eigenvectors via ``profile`` or an ambient dynamical matrix.
        k_0: Canonical free-space wavenumber ``[1, 1, F, 1]``.

    Returns:
        Dict of the eight complex coefficients ``r_pp, r_ss, r_ps, r_sp, t_pp,
        t_ss, t_ps, t_sp`` in canonical ``[A, B, F, T]`` layout (un-presented).

    Note:
        For an isotropic exit the transmission coefficients are in the clean s/p
        basis; for a semi-infinite anisotropic exit they are in that crystal's
        eigenmode basis (as in the transfer method), while the reflection
        coefficients are always in the prism's s/p basis.
    """
    media = [_medium(layer) for layer in layers]
    scattering = None
    for (p_left, q_left, d_left), (p_right, _, _) in zip(media[:-1], media[1:], strict=True):
        interface = _interface_scattering(p_left, q_left, d_left, p_right, k_0)
        scattering = interface if scattering is None else _star(scattering, interface)

    assert_canonical(scattering, matrix_ndim=2, name="scattering_matrix")
    s = scattering
    # PyLlama r_kj is (out k, in j); this codebase uses r_{in->out}, so the cross
    # terms transpose (verified against the transfer backend).
    return {
        "r_pp": s[..., 2, 0],
        "r_ss": s[..., 3, 1],
        "r_ps": s[..., 3, 0],
        "r_sp": s[..., 2, 1],
        "t_pp": s[..., 0, 0],
        "t_ss": s[..., 1, 1],
        "t_ps": s[..., 1, 0],
        "t_sp": s[..., 0, 1],
    }
