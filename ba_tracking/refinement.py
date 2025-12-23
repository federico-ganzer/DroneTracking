import numpy as np
from scipy.optimize import least_squares
import cv2 as cv
from config import CONFIG
from geom import project_p, pose_from_baseline

def pack_params(rvecs, s_vals, X_dyn):
    N_cams = rvecs.shape[0]
    cam_params = np.hstack([rvecs, s_vals.reshape(N_cams, 1)])
    return np.hstack([cam_params.ravel(), X_dyn.ravel()])


def unpack_params(params, N_cams, N_dyn):
    
    cam_flat = params[:4 * N_cams]
    pts_flat = params[4 * N_cams:]

    cam_params = cam_flat.reshape(N_cams, 4)
    rvecs = cam_params[:, :3]     
    s_vals = cam_params[:, 3]     

    X_dyn = pts_flat.reshape(3, N_dyn)
    
    return rvecs, s_vals, X_dyn


def residuals_constrained(
    params,
    K,
    baseline_dir,
    c0,
    cam_indices,
    pt_indices,
    p_obs_all,
    N_cams,
    N_dyn,
    X_static,
    rvecs_ref,
    s_ref,
    lambda_smooth=0.05,
    lambda_pose=5.0,
):
    """
    Constrained BA residuals with:
    - frozen static points
    - SO(3) rotation increments
    - baseline-only translation DOF
    """

    rvecs, s_vals, X_dyn = unpack_params(params, N_cams, N_dyn)
    baseline_dir = baseline_dir / np.linalg.norm(baseline_dir)

    X_all = np.hstack([X_static, X_dyn])

    m = cam_indices.shape[0]
    
    sigma_pix = CONFIG["detection_noise"]

    reproj_res = np.empty((m, 2), dtype=float)

    for k in range(m):
        ci = cam_indices[k]
        pi = pt_indices[k]

        if ci == 0:
            rvec = np.zeros(3)
            s = s_ref[0]
        else:
            rvec = rvecs[ci]
            s = s_vals[ci]

        R_cam, t_cam = pose_from_baseline(baseline_dir, c0, rvecs_ref[ci] + rvec, s)


        p_proj = project_p(K, R_cam, t_cam, X_all[:, pi])
        reproj_res[k] = (p_proj - p_obs_all[:, k]) / sigma_pix

    r = reproj_res.ravel()

    # Smoothness (dynamic points only)
    if lambda_smooth > 0.0 and N_dyn > 1:
        dX = X_dyn[:, 1:] - X_dyn[:, :-1]
        r = np.concatenate([r, lambda_smooth * dX.ravel()])

    
    # Pose regularization (keeps increments small)
    if lambda_pose > 0.0:
        r_pose_rot = lambda_pose * rvecs.ravel()
        r_pose_s   = lambda_pose * (s_vals - s_ref).ravel()
        r = np.concatenate([r, r_pose_rot, r_pose_s])

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

    X_static = X_ref_static.T                # (3, N_static)
    N_static = X_static.shape[1]

    X_dyn0 = X_init[:, N_static:]            # (3, N_dyn)
    N_dyn = X_dyn0.shape[1]

    rvecs0 = np.zeros_like(rvecs_init)
    s0 = s_init.copy()

    params0 = pack_params(rvecs0, s0, X_dyn0)

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
            N_dyn,
            X_static,
            rvecs_init,
            s_init,
            0.01,
            0.2,
        ),
        method="trf",
        loss="huber",
        f_scale=3.0,
        x_scale="jac",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-3,
        max_nfev=200,
        verbose=0,
    )

    # unpack solution
    drvecs, s_opt, X_dyn_opt = unpack_params(res.x, N_cams, N_dyn)

    # compose final rotations
    rvecs_opt = np.zeros_like(rvecs_init)
    for i in range(N_cams):
        dR, _ = cv.Rodrigues(drvecs[i])
        R_ref, _ = cv.Rodrigues(rvecs_init[i])
        R = dR @ R_ref
        rvecs_opt[i] = cv.Rodrigues(R)[0].ravel()

    # full structure output
    X_opt = np.hstack([X_static, X_dyn_opt])

    return rvecs_opt, s_opt, X_opt
