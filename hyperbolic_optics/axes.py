"""Canonical batch-axis convention for all scenario arrays.

Every batched quantity in the pipeline has shape ``[A, B, F, *matrix_axes]``,
where un-swept axes are size 1 (never absent):

- ``A`` -- incident angle points (kx)
- ``B`` -- azimuthal angle points (sample rotation)
- ``F`` -- frequency points (material dispersion)

NumPy never materialises size-1 broadcast axes, so carrying them costs nothing in
time or memory, and it lets the physics code broadcast over leading axes without
knowing which scenario produced the arrays.
"""

import numpy as np

A, B, F = 0, 1, 2  # incident angle, azimuthal angle, frequency
N_BATCH = 3  # number of leading batch axes


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


def present(coefficient: np.ndarray) -> np.ndarray:
    """Map a canonical ``[A, B, F]`` batch quantity to its presentation shape.

    Reorders the batch axes to ``(F, A, B)`` and squeezes the size-1 axes. This
    reproduces every scenario's historical output shape (Incident/Azimuthal ->
    ``(F, angle)``; Dispersion -> ``(A, B)``; FullSweep -> ``(F, A, B)``; Simple
    -> scalar), and is shared by reflection/transmission coefficients
    (``structure.py``) and the field-resolved quantities (``fields.py``) so they
    all present identically.
    """
    return np.squeeze(np.transpose(coefficient, (2, 0, 1)))
