"""Rotation utilities for anisotropic tensor transformations.

A material tensor is rotated by a similarity transform ``M' = R M Rᵀ`` where the
Euler rotation is ``R = Rz(beta) · Ry(phi) · Rx(theta)``.

There is a single rotation function: it builds each elementary rotation with
``np.stack`` of ``cos``/``sin`` (using ``ones_like``/``zeros_like`` so the angles
may be scalars *or* arrays of any broadcastable shape), and applies the transform
via ``matmul``. Because it broadcasts over leading axes, the same function serves
every scenario — feed it canonically-shaped angles (see ``hyperbolic_optics.axes``)
and a canonical ``[A, B, F, 3, 3]`` tensor and it returns a canonical tensor.
"""

import numpy as np


def _euler_rotation_matrix(
    theta: float | np.ndarray, phi: float | np.ndarray, beta: float | np.ndarray
) -> np.ndarray:
    """Build the Euler rotation matrix ``Rz(beta) · Ry(phi) · Rx(theta)``.

    Each angle may be a scalar or an array of any broadcastable shape; the result
    has shape ``[*broadcast(theta, phi, beta), 3, 3]``.
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack([np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)], axis=-1),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    return (rotation_z @ rotation_y @ rotation_x).astype(np.complex128)


def rotate_tensor(
    tensor: np.ndarray,
    theta: float | np.ndarray,
    phi: float | np.ndarray,
    beta: float | np.ndarray,
) -> np.ndarray:
    """Rotate a 3×3 material tensor by Euler angles via ``R · tensor · Rᵀ``.

    Args:
        tensor: Permittivity or permeability tensor, ``[..., 3, 3]``.
        theta: Rotation around x (radians); scalar or broadcastable array.
        phi: Rotation around y (radians); scalar or broadcastable array.
        beta: Rotation around z (radians); scalar or broadcastable array.

    Returns:
        The rotated tensor, broadcast over the combined angle/tensor shape.
    """
    rotation = _euler_rotation_matrix(theta, phi, beta)
    return rotation @ tensor @ np.swapaxes(rotation, -2, -1)
