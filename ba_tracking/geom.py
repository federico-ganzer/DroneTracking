import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv
from scipy.optimize import least_squares

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float):
    return R.from_rotvec(axis * angle).as_matrix()

def perturb_R(R_true: np.ndarray, angle_sigma: float):
    '''Perturb rotation matrix by random axis-angle noise.'''

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    angle = np.random.randn() * angle_sigma

    dR = rotation_matrix_from_axis_angle(axis, angle)

    R_perturb = dR @ R_true

    return R_perturb

def perturb_t(t_true: np.ndarray, sigma: float):
    return t_true + np.random.randn(3) * sigma

def project_p(K, R, t, X):
    """Project 3D points X into camera with intrinsics K, rotation R and translation t.

    X can be shape (3,) or (3, N) or (N, 3). Returns 2xN array of pixel coordinates.
    """
    X_arr = np.asarray(X)
    # convert to shape (3, N)
    if X_arr.ndim == 1:
        Xc = X_arr.reshape(3, 1)
    elif X_arr.ndim == 2 and X_arr.shape[0] == 3:
        Xc = X_arr
    elif X_arr.ndim == 2 and X_arr.shape[1] == 3:
        Xc = X_arr.T
    else:
        raise ValueError("X must have shape (3,), (3,N) or (N,3)")

    X_cam = R @ Xc + t.reshape(3, 1)

    p_hom = K @ X_cam

    p_2d = p_hom[:2, :] / p_hom[2:3, :]

    if p_2d.shape[1] == 1:
        return p_2d[:, 0]
    return p_2d

def triangulate_points(K1, R1, t1, K2, R2, t2, p1, p2):
    '''
    Triangulate 3D points from two views.
    p1 and p2 are 2xN arrays of pixel coordinates in the two images.
    Returns 3xN array of 3D points.
    '''
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K2 @ np.hstack((R2, t2.reshape(3, 1)))

    p1_hom = np.vstack((p1, np.ones((1, p1.shape[1]))))
    p2_hom = np.vstack((p2, np.ones((1, p2.shape[1]))))

    X_hom = cv.triangulatePoints(P1, P2, p1_hom[:2, :], p2_hom[:2, :])

    X_world = X_hom[:3, :] / X_hom[3:4, :]

    return X_world

def reproj_error(K, R, t, X, p_obs):
    """
    Compute reprojection error.

    X is 3xN array of 3D points.
    p_obs is 2xN array of observed pixel coordinates.
    Returns array of reprojection errors for each point.
    """
    p_proj = project_p(K, R, t, X)

    errors = np.linalg.norm(p_proj - p_obs, axis=0)

    return errors

def rotation_error_deg(R_est, R_gt):
    # relative rotation
    R_rel = R_est @ R_gt.T
    # angle from trace
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.rad2deg(theta)

def pose_from_baseline(baseline_dir, c0, rvec, s):
    '''
    Construct pose from rvec
    '''
    
    baseline_dir = baseline_dir/np.linalg.norm(baseline_dir)
    
    C = c0 + s * baseline_dir
    R_cam, _ = cv.Rodrigues(rvec.astype(float))
    
    t = -R_cam @ C
    
    return R_cam, t