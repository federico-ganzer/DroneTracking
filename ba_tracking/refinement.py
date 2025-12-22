
import numpy as np
from scipy.optimize import least_squares
import cv2 as cv
from config import CONFIG
from geom import project_p, pose_from_baseline 


def pack_params(rvecs, s_vals, X_world):
    N_cams = rvecs.shape[0]
    s_vals = s_vals.reshape(N_cams, 1)
    cam_params = np.hstack([rvecs, s_vals])   # (N_cams, 4)
    return np.hstack([cam_params.ravel(), X_world.ravel()])


def unpack_params(params, N_cams, N_pts):
    cam_flat = params[:4 * N_cams]
    pts_flat = params[4 * N_cams:]

    cam_params = cam_flat.reshape(N_cams, 4)
    rvecs = cam_params[:, :3]      # local rotation increments
    s_vals = cam_params[:, 3]      # local baseline increments

    X_world = pts_flat.reshape(3, N_pts)
    return rvecs, s_vals, X_world


def residuals_constrained(
    params,
    K,
    baseline_dir,
    c0,
    cam_indices,
    pt_indices,
    p_obs_all,
    N_cams,
    N_pts,
    lambda_smooth=0.05,
    X_ref_static=None,
    lambda_static=300,
    N_static=1,
    rvecs_ref=None,
    s_ref=None,
    lambda_pose=10.0,
):
    """
    Residuals for constrained BA with:
    - proper SO(3) updates
    - static-point anchoring
    - optional pose regularization
    """
    rvecs, s_vals, X_world = unpack_params(params, N_cams, N_pts)
    baseline_dir = baseline_dir / np.linalg.norm(baseline_dir)

    M = cam_indices.shape[0]
    sigma_pix = CONFIG['detection_noise']

    reproj_res = np.empty((M, 2), dtype=float)

    for k in range(M):
        ci = cam_indices[k]
        pi = pt_indices[k]

        # --- compose rotation properly ---
        dR, _ = cv.Rodrigues(rvecs[ci])
        R_ref, t_ref = pose_from_baseline(
            baseline_dir, c0, rvecs_ref[ci], s_ref[ci]
        )

        R_cam = dR @ R_ref
        t_cam = t_ref + baseline_dir * (s_vals[ci] - s_ref[ci])

        p_proj = project_p(K, R_cam, t_cam, X_world[:, pi])

        reproj_res[k] = (p_proj - p_obs_all[:, k]) / sigma_pix

    r = reproj_res.ravel()

    # --- smooth dynamic trajectory ---
    if lambda_smooth > 0.0 and N_pts > N_static + 1:
        dX = X_world[:, N_static + 1:] - X_world[:, N_static:-1]
        r = np.concatenate([r, lambda_smooth * dX.ravel()])

    # --- static-point anchor ---
    if lambda_static > 0.0 and X_ref_static is not None:
        static_res = lambda_static * (
            X_world[:, :N_static].T - X_ref_static
        ).ravel()
        r = np.concatenate([r, static_res])

    # --- pose prior (optional) ---
    if lambda_pose > 0.0 and rvecs_ref is not None and s_ref is not None:
        pose_rot_res = lambda_pose * rvecs.ravel()
        pose_s_res   = lambda_pose * (s_vals - s_ref).ravel()
        r = np.concatenate([r, pose_rot_res, pose_s_res])

    return r


def run_constrained_ba(
    K,
    baseline_dir,
    c0,
    rvecs_init,
    s_init,
    X_init,
    X_ref_static,
    cam_indices,
    pt_indices,
    p_obs_all,
):
    N_cams = rvecs_init.shape[0]
    N_pts = X_init.shape[1]
    N_static = X_ref_static.shape[0]

    # local increments start at zero
    rvecs0 = np.zeros_like(rvecs_init)
    s0 = s_init.copy()

    params0 = pack_params(rvecs0, s0, X_init)

    res = least_squares(
        residuals_constrained,
        params0,
        args=(
            K,
            baseline_dir,
            c0,
            cam_indices,
            pt_indices,
            p_obs_all,
            N_cams,
            N_pts,
            0.05,
            X_ref_static,
            300,
            N_static,
            rvecs_init,
            s_init,
            5.0,
        ),
        method="trf",
        x_scale="jac",
        loss="huber",
        f_scale=3.0,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-3,
        max_nfev=200,
        verbose=0,
    )

    drvecs, s_opt, X_opt = unpack_params(res.x, N_cams, N_pts)

    # compose final rotations
    rvecs_opt = np.zeros_like(rvecs_init)
    for i in range(N_cams):
        dR, _ = cv.Rodrigues(drvecs[i])
        R_ref, _ = cv.Rodrigues(rvecs_init[i])
        R = dR @ R_ref
        rvecs_opt[i] = cv.Rodrigues(R)[0].ravel()

    return rvecs_opt, s_opt, X_opt

'''
def pack_params(rvecs, s_vals, X_world):
    """
    rvecs: (N_cams, 3)
    s_vals: (N_cams,) or (N_cams, 1)
    X_world: (3, N_pts)
    Returns: 1D parameter vector.
    """
    N_cams = rvecs.shape[0]
    s_vals = s_vals.reshape(N_cams, 1)
    cam_params = np.hstack([rvecs, s_vals])   # (N_cams, 4)
    return np.hstack([cam_params.ravel(), X_world.ravel()])

def unpack_params(params, N_cams, N_pts):
    """
    Inverse of pack_params.
    """
    cam_flat = params[:4 * N_cams]
    pts_flat = params[4 * N_cams:]

    cam_params = cam_flat.reshape(N_cams, 4)      # (N_cams, 4)
    rvecs = cam_params[:, :3]
    s_vals = cam_params[:, 3]

    X_world = pts_flat.reshape(3, N_pts)          # (3, N_pts)
    return rvecs, s_vals, X_world

def residuals_constrained(params, K, baseline_dir, c0,
                          cam_indices, pt_indices, p_obs_all,
                          N_cams, N_pts,
                          lambda_smooth=0.05,
                          X_ref_static=None,
                          lambda_static = 1000,
                          N_static= 1,
                          rvecs_ref= None,
                          s_ref= None,
                          lambda_pose = 10):
    """
    Returns: ((2*M) [+ 3*(N_pts-1)] ,) residual vector with applied constraints.
    Constraints:
        - baseline_dir is conserved
        - X_ref_static static prior is tightly constrained
    """
    rvecs, s_vals, X_world = unpack_params(params, N_cams, N_pts)
    baseline_dir = baseline_dir / np.linalg.norm(baseline_dir)

    M = cam_indices.shape[0]

    # --- reprojection residuals (M, 2) ---
    r_weighted = np.empty((M, 2), dtype=float)

    for k in range(M):
        ci = cam_indices[k]
        pi = pt_indices[k]

        rvec = rvecs[ci]
        s = s_vals[ci]
        X = X_world[:, pi]

        R_cam, t_cam = pose_from_baseline(baseline_dir, c0, rvec, s)
        p_proj = project_p(K, R_cam, t_cam, X)

        r_weighted[k] = (p_proj - p_obs_all[:, k])/CONFIG['detection_noise']


    if lambda_smooth > 0.0 and N_pts > N_static + 1:
        # X_world: (3, N_pts)
        dX = X_world[:, N_static + 1:] - X_world[:, N_static:-1]  # use dynamic points only
        smooth_res = lambda_smooth * dX.ravel()    
        r_weighted = np.concatenate([r_weighted, smooth_res])

    if lambda_static > 0.0 and X_ref_static is not None:
        static_res = lambda_static * (X_world[:, :N_static].T - X_ref_static).ravel()
        r_weighted = np.concatenate([r_weighted, static_res])
        
    if lambda_pose > 0.0 and rvecs_ref is not None and s_ref is not None:
        pose_res_rot = lambda_pose * (rvecs - rvecs_ref).ravel()
        pose_res_s   = lambda_pose * (s_vals - s_ref).ravel()
        r_weighted = np.concatenate([r_weighted, pose_res_rot, pose_res_s])
    
    return r_weighted


def run_constrained_ba(
        K,
        baseline_dir,
        c0,
        rvecs_init,
        s_init,
        X_init,
        X_ref_static,
        cam_indices,
        pt_indices,
        p_obs_all):
    """
    K: 3x3 intrinsics
    baseline_dir: (3,) baseline direction
    c0: (3,) reference point on baseline
    rvecs_init: (N_cams, 3) initial Rodrigues rotations
    s_init: (N_cams,) initial scalar offsets along baseline
    X_init: (3, N_pts) initial 3D points
    cam_indices: (M,) camera index per observation
    pt_indices: (M,) point index per observation
    p_obs_all: (2, M) observed pixel coords

    Returns: optimized (rvecs, s_vals, X_world)
    """
    N_cams = rvecs_init.shape[0]
    N_pts = X_init.shape[1]

    params0 = pack_params(rvecs_init, s_init, X_init)
    N_static = X_ref_static.shape[0]
    
    res = least_squares(
        residuals_constrained,
        params0,
        args=(K, baseline_dir, c0, cam_indices, pt_indices, p_obs_all,
              N_cams, N_pts, 0.05, X_ref_static, 300, N_static, rvecs_init, s_init, 500),
        method="trf",
        x_scale="jac",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-3,
        max_nfev=200,
        verbose=0,
        loss='huber',
        f_scale= 3.0
        )

    rvecs_opt, s_opt, X_opt = unpack_params(res.x, N_cams, N_pts)
    return rvecs_opt, s_opt, X_opt



'''