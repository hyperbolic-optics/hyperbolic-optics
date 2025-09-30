"""Rotation utilities for anisotropic tensor transformations.

This module provides functions for rotating permittivity and permeability
tensors using Euler angles. The rotations are performed using the convention:
R_total = R_z(β) · R_y(φ) · R_x(θ)

The functions support different broadcasting patterns for various scenario types,
enabling efficient batch processing of angle sweeps and dispersion calculations.
"""

import numpy as np


def anisotropy_rotation_one_value(
    matrix: np.ndarray, theta: float | np.ndarray, phi: float | np.ndarray, beta: float | np.ndarray
) -> np.ndarray:
    """Apply Euler angle rotations to a tensor for scalar angle values.

    Performs a sequence of rotations (Rz · Ry · Rx) on a 3×3 tensor using
    Euler angles. This is used for rotating permittivity and permeability
    tensors to account for crystal orientation.

    Args:
        matrix: The 3×3 tensor to rotate (permittivity or permeability)
        theta: Rotation angle around x-axis in radians
        phi: Rotation angle around y-axis in radians
        beta: Rotation angle around z-axis in radians

    Returns:
        The rotated 3×3 tensor with shape matching the input matrix

    Note:
        The rotation is performed as: R · matrix · R^T where R = Rz · Ry · Rx

    Example:
        >>> eps = np.diag([2.5, 2.5, 4.0])
        >>> rotated = anisotropy_rotation_one_value(eps, 0, np.pi/4, 0)
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    total_rotation = (rotation_z @ rotation_y @ rotation_x).astype(np.complex128)
    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result


def anisotropy_rotation_one_axis(
    matrix: np.ndarray, theta: np.ndarray, phi: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Apply Euler angle rotations with broadcasting along one axis.

    Similar to anisotropy_rotation_one_value but handles rotation arrays
    along a single axis, typically used for azimuthal rotation scenarios.

    Args:
        matrix: The 3×3 tensor to rotate with shape [N, 3, 3]
        theta: Rotation angles around x-axis in radians, shape [M]
        phi: Rotation angles around y-axis in radians, shape [M]
        beta: Rotation angles around z-axis in radians, shape [M]

    Returns:
        Rotated tensor with shape [M, N, 3, 3] after broadcasting

    Note:
        This function adds a new axis to the matrix and broadcasts the
        rotation across it, enabling efficient batch processing.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    total_rotation = (rotation_z @ rotation_y @ rotation_x).astype(np.complex128)

    matrix = matrix[:, np.newaxis, :, :]
    total_rotation = total_rotation[np.newaxis, :, :, :]

    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result


def anisotropy_rotation_all_axes(
    matrix: np.ndarray, theta: np.ndarray, phi: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Apply Euler angle rotations with broadcasting along all axes.

    Performs rotation with full broadcasting support for dispersion scenarios
    where both incident angle and azimuthal angle vary independently.

    Args:
        matrix: The 3×3 tensor to rotate with shape [N, 3, 3]
        theta: Rotation angles around x-axis in radians, shape [I]
        phi: Rotation angles around y-axis in radians, shape [J]
        beta: Rotation angles around z-axis in radians, shape [K]

    Returns:
        Rotated tensor with shape [I, J, K, N, 3, 3] after full broadcasting

    Note:
        This is the most general rotation function, supporting independent
        variation of all three Euler angles simultaneously.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.stack(
        [
            np.stack(
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                axis=-1,
            ),
            np.stack([np.zeros_like(theta), cos_theta, -sin_theta], axis=-1),
            np.stack([np.zeros_like(theta), sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.stack(
        [
            np.stack([cos_phi, np.zeros_like(phi), sin_phi], axis=-1),
            np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=-1),
            np.stack([-sin_phi, np.zeros_like(phi), cos_phi], axis=-1),
        ],
        axis=-2,
    )

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    rotation_x = rotation_x[:, np.newaxis, np.newaxis, :, :]
    rotation_y = rotation_y[np.newaxis, :, np.newaxis, :, :]
    rotation_z = rotation_z[np.newaxis, np.newaxis, :, :, :]

    total_rotation = rotation_z @ rotation_y @ rotation_x

    matrix = matrix[:, np.newaxis, np.newaxis, np.newaxis, :, :]
    total_rotation = total_rotation[np.newaxis, ...]

    result = total_rotation @ matrix @ np.swapaxes(total_rotation, -2, -1)

    return result


def anisotropy_rotation_two_axes(
    matrix: np.ndarray, theta: float, phi: float, beta: np.ndarray
) -> np.ndarray:
    """Apply Euler angle rotations with fixed X/Y and array Z (azimuthal).

    Used for FullSweep where crystal orientation (X, Y) is fixed but Z varies
    with azimuthal angle.

    Args:
        matrix: The 3×3 tensor to rotate with shape [N_freq, 3, 3]
        theta: Fixed rotation angle around x-axis in radians
        phi: Fixed rotation angle around y-axis in radians
        beta: Rotation angles around z-axis in radians, shape [N_incident, N_azim]

    Returns:
        Rotated tensor with shape [N_incident, N_azim, N_freq, 3, 3]
    """
    # Fixed rotations
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_theta, -sin_theta], [0.0, sin_theta, cos_theta]],
        dtype=np.complex128,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    rotation_y = np.array(
        [[cos_phi, 0.0, sin_phi], [0.0, 1.0, 0.0], [-sin_phi, 0.0, cos_phi]], dtype=np.complex128
    )

    # Variable Z rotation for incident and azimuthal
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    rotation_z = np.stack(
        [
            np.stack([cos_beta, -sin_beta, np.zeros_like(beta)], axis=-1),
            np.stack([sin_beta, cos_beta, np.zeros_like(beta)], axis=-1),
            np.stack([np.zeros_like(beta), np.zeros_like(beta), np.ones_like(beta)], axis=-1),
        ],
        axis=-2,
    )

    # Combine rotations
    intermediate = rotation_y @ rotation_x
    total_rotation = rotation_z @ intermediate[np.newaxis, np.newaxis, :, :]

    # Apply rotation to tensors
    matrix_expanded = matrix[np.newaxis, np.newaxis, :, :, :]
    total_rotation_expanded = total_rotation[:, :, np.newaxis, :, :]

    result = (
        total_rotation_expanded @ matrix_expanded @ np.swapaxes(total_rotation_expanded, -2, -1)
    )

    return result
