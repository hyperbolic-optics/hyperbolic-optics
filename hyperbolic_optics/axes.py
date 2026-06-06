"""Canonical batch-axis convention for all scenario arrays.

Every batched quantity in the pipeline has shape ``[A, B, F, T, *matrix_axes]``,
where un-swept axes are size 1 (never absent):

- ``A`` -- incident angle points (kx)
- ``B`` -- azimuthal angle points (sample rotation)
- ``F`` -- frequency points (material dispersion)
- ``T`` -- layer-thickness points (a single swept layer; size 1 unless a layer's
  ``thickness`` is given as a list)

NumPy never materialises size-1 broadcast axes, so carrying them costs nothing in
time or memory, and it lets the physics code broadcast over leading axes without
knowing which scenario produced the arrays. The thickness axis only becomes
non-trivial for the one layer being swept; every other quantity keeps ``T = 1``
and broadcasts.
"""

import numpy as np

A, B, F, T = 0, 1, 2, 3  # incident angle, azimuthal angle, frequency, thickness
N_BATCH = 4  # number of leading batch axes


def assert_canonical(arr: np.ndarray, matrix_ndim: int = 2, name: str = "array") -> None:
    """Assert ``arr`` has ``N_BATCH`` leading batch axes + ``matrix_ndim`` trailing.

    Args:
        arr: Array to check.
        matrix_ndim: Number of trailing matrix axes (2 for a 4x4 / 3x3 block,
            0 for a pure batch quantity such as a reflection coefficient).
        name: Label used in the error message.

    Raises:
        AssertionError: If ``arr`` does not have the expected rank.
    """
    expected = N_BATCH + matrix_ndim
    if arr.ndim != expected:
        raise AssertionError(
            f"{name}: expected {expected}D [A, B, F, ...], got {arr.ndim}D {arr.shape}"
        )


def canonicalize(
    arr: np.ndarray, batch_axes: tuple[int, ...] = (), matrix_ndim: int = 0
) -> np.ndarray:
    """Lift a raw array into canonical ``[A, B, F, T, *matrix]`` layout.

    The array's leading axes (everything but the trailing ``matrix_ndim`` axes) map,
    in order, onto the canonical positions named in ``batch_axes``; every other
    batch position becomes size 1. Trailing matrix axes are preserved.

    This is the single boundary adapter for raw quantities (``kx``, ``k0``, fetched
    material tensors, rotation angles, ambient transfer matrices). Being parametric
    in :data:`N_BATCH` is the whole point: adding or removing a batch axis is a
    one-line change to the constants above, not a hunt for ``np.newaxis`` at every
    call site.

    Args:
        arr: Raw array whose leading axes correspond to ``batch_axes``.
        batch_axes: Canonical positions (e.g. ``(A,)``, ``(F,)``, ``(A, B)``) for
            ``arr``'s leading axes, in order. Empty for a fully un-swept quantity.
        matrix_ndim: Number of trailing matrix axes to preserve (2 for a 3x3/4x4
            block, 0 for a scalar field).

    Returns:
        ``arr`` reshaped to ``N_BATCH`` leading axes (un-named positions size 1)
        plus the preserved trailing matrix axes.
    """
    arr = np.asarray(arr)
    n_lead = arr.ndim - matrix_ndim
    if n_lead != len(batch_axes):
        raise ValueError(
            f"canonicalize: expected {len(batch_axes)} leading axes for {batch_axes}, "
            f"got {n_lead} from shape {arr.shape} (matrix_ndim={matrix_ndim})."
        )
    target = [1] * N_BATCH
    for source_pos, canonical_axis in enumerate(batch_axes):
        target[canonical_axis] = arr.shape[source_pos]
    return arr.reshape(tuple(target) + arr.shape[n_lead:])


def present(coefficient: np.ndarray) -> np.ndarray:
    """Map a canonical ``[A, B, F, T]`` batch quantity to its presentation shape.

    Reorders the batch axes to ``(F, A, B, T)`` and squeezes the size-1 axes. With
    no thickness sweep ``T = 1`` squeezes away, exactly reproducing every
    scenario's historical output shape (Incident/Azimuthal -> ``(F, angle)``;
    Dispersion -> ``(A, B)``; FullSweep -> ``(F, A, B)``; Simple -> scalar). A
    thickness sweep appends the trailing ``T`` axis (e.g. Simple -> ``(T,)``,
    Incident -> ``(F, angle, T)``). Shared by the reflection/transmission
    coefficients (``structure.py``) and the field-resolved quantities
    (``fields.py``) so they all present identically.
    """
    return np.squeeze(np.transpose(coefficient, (2, 0, 1, 3)))
