from typing import cast

from numbers import Number

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


try:
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:  # pragma: no cover
    Rotation = None


# @njit(cache=True)
def _matrix_to_euler(matrix: NDArray) -> NDArray:
    """
    Converts a rotation matrices to Euler angles

    Parameters
    ----------
    matrix: NDArray
        A batch of rotation matrices. Given row-wise)

    Returns
    -------
    NDArray
        A batch of Euler angles
    """
    if Rotation is not None:
        R = Rotation.from_matrix(matrix.T)
        return cast(NDArray, R.as_euler("xyz"))

    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if singular:
        x_angle = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y_angle = np.arctan2(-matrix[2, 0], sy)
        z_angle = 0.0
    else:
        x_angle = np.arctan2(matrix[2, 1], matrix[2, 2])
        y_angle = np.arctan2(-matrix[2, 0], sy)
        z_angle = np.arctan2(matrix[1, 0], matrix[0, 0])

    return np.array([x_angle, y_angle, z_angle])


def _validate_position(position: NDArray) -> None:
    """
    Checks if inputted position values are valid

    Paramters
    ---------
    position: NDArray
        Position input (endpoint or centerpoint depending on Object type)

    Raises
    ------
    ValueError
        If the position is the wrong shape or contains NaN values
    """

    if position.shape[0] != 3:
        raise ValueError("The shape of the position is incorrect.")
    if np.isnan(position).any():
        raise ValueError("The position contains NaN values.")


def _validate_radius(radius: float) -> None:
    """
    Checks if inputted radius value is valid

    Parameters:
    -----------
    radius: Float
        Radius input

    Raises
    ------
    ValueError
        If the radius is not positive, or contains NaN values
    """

    if not isinstance(radius, Number) or radius <= 0:
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radius):
        raise ValueError("The radius contains NaN values.")


def _validate_radii(radii: NDArray) -> None:
    """
    Checks if inputted radius value is valid

    Parameters:
    -----------
    radii: NDArray
        Radius input

    Raises
    ------
    ValueError
        If the radius is not positive, or contains NaN values
    """

    if (radii <= 0).any():
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radii).any():
        raise ValueError("The radius contains NaN values.")


def _validate_rotation_matrix(director: NDArray) -> None:
    """
    Checks if inputted rotation matrix is valid

    Parameters:
    -----------
    director: NDArray
        Rotation matrix input

    Raises
    ------
    ValueError
        If the rotation matrix is not a 3x3 matrix, or contains NaN values
    """

    if np.isnan(director).any():
        raise ValueError("The director contains NaN values.")


def compute_fin_segment_bounds(
    pipe_offset: NDArray, pipe_radius: float, overlap_ratio: float = 0.05
) -> NDArray:
    """
    Compute a fin segment boundary that extends slightly into an adjacent
    pipe's footprint, rather than stopping exactly at its edge.

    Given the signed distance from a fin segment's own spine to an adjacent
    pipe's center (``pipe_offset``), returns the distance to the pipe's near
    edge nudged inward by ``overlap_ratio`` of the pipe's diameter -- small
    enough to guarantee no visible gap between the fin segment and the pipe,
    while staying well within the pipe's footprint so it doesn't poke out
    the other side.

    Parameters
    ----------
    pipe_offset : NDArray
        Signed distance from the fin segment's spine to the pipe's center,
        at each node/element the segment shares with the pipe. Positive
        values mean the pipe sits further from the spine in the direction
        the segment's span extends; negative values mean the opposite
        direction.
    pipe_radius : float
        Constant radius of the adjacent pipe.
    overlap_ratio : float, optional
        Fraction of the pipe's diameter (``2 * pipe_radius``) to extend the
        boundary into the pipe's footprint. Default is 0.05.

    Returns
    -------
    NDArray
        The fin segment boundary value (e.g. ``left_span`` or
        ``right_span``, depending on the sign convention used for
        ``pipe_offset``) to use adjacent to this pipe.
    """
    pipe_offset = np.asarray(pipe_offset, dtype=float)
    sign = np.sign(pipe_offset)
    sign = np.where(sign == 0, 1.0, sign)
    overlap = overlap_ratio * (2.0 * pipe_radius)
    return cast(NDArray, pipe_offset - sign * (pipe_radius - overlap))
