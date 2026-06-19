"""Core geometry types for the VLM task planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = ["RigidTransform", "CameraIntrinsics"]


@dataclass(frozen=True)
class RigidTransform:
    """A rigid-body pose consisting of a position and rotation matrix.

    Attributes:
        position: 3D position vector of shape (3,).
        rotation: 3×3 rotation matrix of shape (3, 3).
    """

    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3)

    def inv(self) -> RigidTransform:
        """Compute the inverse of this transform."""
        inv_rotation = self.rotation.T
        inv_position = -inv_rotation @ self.position
        return RigidTransform(position=inv_position, rotation=inv_rotation)

    def __mul__(self, other: Union[RigidTransform, np.ndarray]) -> Union[RigidTransform, np.ndarray]:
        """Apply this transform to a RigidTransform, a 3D point, or a point cloud.

        Args:
            other: A ``RigidTransform`` for pose composition, a shape ``(3,)``
                array for a single point, or a shape ``(N, 3)`` array for a
                point cloud.

        Returns:
            The composed transform or transformed point(s).

        Raises:
            ValueError: If *other* is an array with an unsupported shape.
        """
        # Case 1: Transforming another RigidTransform (Pose composition)
        if isinstance(other, RigidTransform):
            new_rotation = self.rotation @ other.rotation
            new_position = self.position + self.rotation @ other.position
            return RigidTransform(position=new_position, rotation=new_rotation)

        # Case 2: Transforming NumPy arrays (Points)
        elif isinstance(other, np.ndarray):
            # 2a. Single 3D point: shape (3,)
            if other.shape == (3,):
                return self.position + self.rotation @ other
            # 2b. Point Cloud / Batch of points: shape (N, 3)
            elif other.ndim == 2 and other.shape[1] == 3:
                # Use matrix transpose trick for fast batched processing
                return (other @ self.rotation.T) + self.position
            else:
                raise ValueError(f"Expected array of shape (3,) or (N, 3), got {other.shape}")

        # Case 3: Unsupported type
        return NotImplemented


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length along the x-axis.
        fy: Focal length along the y-axis.
        ppx: Principal point x-coordinate.
        ppy: Principal point y-coordinate.
    """

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
