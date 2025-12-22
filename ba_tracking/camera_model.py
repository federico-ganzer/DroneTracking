"""
camera_model.py: Pinhole camera model with projection and back-projection
"""

import numpy as np


class PinholeCamera:
    """
    Pinhole camera model with intrinsic and extrinsic parameters.

    Intrinsics: K (3x3 matrix)
    Extrinsics: R (3x3 rotation), t (3x1 translation)

    Projection: uv = K @ (R @ X_world + t)
    """

    def __init__(self, K, pose=None):
        """
        Args:
            K (np.ndarray): 3x3 intrinsic matrix.
            pose (tuple | None): Optional (R, t) extrinsics.
                                 R: 3x3 rotation, t: (3,) or (3,1).
                                 If None, identity pose is used.
        """
        self.K = np.asarray(K, dtype=np.float64)

        if pose is None:
            self.R = np.eye(3, dtype=np.float64)
            self.t = np.zeros((3, 1), dtype=np.float64)
        else:
            R, t = pose
            self.set_pose(R, t)

    def set_pose(self, R, t):
        """Set camera extrinsic parameters.

        Args:
            R (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3x1 or (3,) translation vector
        """
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    def get_pose(self):
        """Return current pose.

        Returns:
            (R, t): rotation matrix and translation vector (3x3, 3x1)
        """
        return self.R.copy(), self.t.copy()

    def project(self, X_world):
        """Project 3D world point(s) to 2D image coordinates.

        Args:
            X_world (np.ndarray): 3D point, shape (3,) or (3, N)

        Returns:
            uv (np.ndarray): 2D pixel coordinates, shape (2,) or (2, N)
        """
        X_world = np.asarray(X_world, dtype=np.float64)

        if X_world.ndim == 1:
            X_world = X_world.reshape(3, 1)
            squeeze = True
        else:
            squeeze = False

        # Transform to camera frame
        X_cam = self.R @ X_world + self.t  # (3, N)

        # Project to image plane
        uv_homogeneous = self.K @ X_cam    # (3, N)

        # Normalize by depth
        uv = uv_homogeneous[:2] / uv_homogeneous[2]

        if squeeze:
            uv = uv.reshape(2)

        return uv

    def unproject(self, uv, depth):
        """Back-project 2D point(s) with depth to 3D world coordinates.

        Args:
            uv (np.ndarray): 2D pixel coordinates, shape (2,) or (2, N)
            depth (float or np.ndarray): depth value(s)

        Returns:
            X_world (np.ndarray): 3D world point(s), shape (3,) or (3, N)
        """
        uv = np.asarray(uv, dtype=np.float64)

        if uv.ndim == 1:
            uv = uv.reshape(2, 1)
            squeeze = True
        else:
            squeeze = False

        # Ensure depth array shape matches
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim == 0:
            depth = np.full((uv.shape[1],), depth, dtype=np.float64)
        depth = depth.reshape(1, -1)

        # Back-project in camera frame
        uv_homogeneous = np.vstack([uv, np.ones((1, uv.shape[1]))])
        K_inv = np.linalg.inv(self.K)
        X_cam = K_inv @ uv_homogeneous * depth  # (3, N)

        # Transform to world frame: X_world = R^T (X_cam - t)
        X_world = self.R.T @ (X_cam - self.t)

        if squeeze:
            X_world = X_world.reshape(3)

        return X_world

    def get_intrinsic_matrix(self):
        """Return intrinsic matrix K."""
        return self.K.copy()

    def __repr__(self):
        return f"PinholeCamera(K=\n{self.K}, R=\n{self.R}, t={self.t.ravel()})"
