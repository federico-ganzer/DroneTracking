# experiment.py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from config import CONFIG
from camera_model import PinholeCamera
import geom
import refinement as ba
from trajectory_gen import generate_circular_trajectory


def setup_cameras(config):
    """Create true and perturbed camera poses."""
    n_cams = 2
    c0 = np.array([0., 0., 0.])
    baseline = np.array([0., 1., 0.])
    baseline_u = baseline / np.linalg.norm(baseline)
    f = config['focal_length']

    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]], dtype=float)

    s_true = np.array([0.0, 10.0])
    rvec_true = np.zeros((n_cams, 3), dtype=float)

    R_true, t_true = [], []
    for i in range(n_cams):
        R_i, t_i = geom.pose_from_baseline(
            baseline_dir=baseline_u, c0=c0, rvec=rvec_true[i], s=s_true[i]
        )
        R_true.append(R_i)
        t_true.append(t_i)

    # perturbations
    angle_sigma = np.deg2rad(config.get("rotation_noise_deg", 3.0))
    t_sigma = config.get("translation_noise", 0.07)

    R_init = [geom.perturb_R(R, angle_sigma=angle_sigma) for R in R_true]
    t_init = [geom.perturb_t(t, sigma=t_sigma) for t in t_true]

    return K, baseline_u, c0, R_true, t_true, R_init, t_init


def simulate_observations(K, baseline_u, R_init, t_init, X_true, X_static_true, config):
    """Simulate dynamic and static observations for both cameras.
    
    Args:
        X_static_true: shape (N_static, 3)
    
    Returns:
        obs_per_cam, static_obs_per_cam, X_est, X_static_est
    """
    n_cams = 2
    num_frames = X_true.shape[0]
    detection_noise = config.get('detection_noise', 0.5)
    N_static = X_static_true.shape[0]

    R1, R2 = R_init
    t1, t2 = t_init

    obs_per_cam = [dict() for _ in range(n_cams)]
    # static_obs_per_cam[cam_id][static_idx] = list of 2D observations
    static_obs_per_cam = [[[] for _ in range(N_static)] for _ in range(n_cams)]

    X_est = np.zeros_like(X_true)

    for frame in range(num_frames):
        Xw = X_true[frame]

        # dynamic point observations
        p1 = geom.project_p(K, R1, t1, Xw)
        p2 = geom.project_p(K, R2, t2, Xw)
        if detection_noise:
            p1 += np.random.randn(2) * detection_noise
            p2 += np.random.randn(2) * detection_noise
        p1 = p1.reshape(2, 1)
        p2 = p2.reshape(2, 1)

        obs_per_cam[0][frame] = p1
        obs_per_cam[1][frame] = p2

        # static points
        for j in range(N_static):
            Xs = X_static_true[j]
            p1_stat = geom.project_p(K, R1, t1, Xs)
            p2_stat = geom.project_p(K, R2, t2, Xs)
            if detection_noise:
                p1_stat += np.random.randn(2) * detection_noise
                p2_stat += np.random.randn(2) * detection_noise
            static_obs_per_cam[0][j].append(p1_stat.reshape(2, 1))
            static_obs_per_cam[1][j].append(p2_stat.reshape(2, 1))

        # triangulate dynamic
        X_tri = geom.triangulate_points(K, R1, t1, K, R2, t2, p1, p2)
        X_est[frame] = X_tri[:, 0]

    # triangulate static points from first frame
    X_static_est = np.zeros_like(X_static_true)
    for j in range(N_static):
        p1s = static_obs_per_cam[0][j][0]
        p2s = static_obs_per_cam[1][j][0]
        Xs_est = geom.triangulate_points(K, R1, t1, K, R2, t2, p1s, p2s)[:, 0]
        X_static_est[j] = Xs_est

    return obs_per_cam, static_obs_per_cam, X_est, X_static_est


def pose_error(R_est, t_est, R_true, t_true):
    """Rotation error [deg] and position error [norm] per camera."""
    rot_err = []
    pos_err = []
    for Re, te, Rt, tt in zip(R_est, t_est, R_true, t_true):
        theta = geom.rotation_error_deg(Re, Rt)
        Ce = -Re.T @ te
        Ct = -Rt.T @ tt
        pe = np.linalg.norm(Ce - Ct)
        rot_err.append(theta)
        pos_err.append(pe)
    # return mean over cameras for a scalar summary
    return np.mean(rot_err), np.mean(pos_err)


def run_ba_once(K, baseline_u, c0, R_init, t_init,
                obs_per_cam, static_obs_per_cam,
                X_est, X_static_est, X_static_true, config):
    """Run one BA over a window and return refined poses."""
    num_frames = X_est.shape[0]
    N_static = X_static_true.shape[0]
    R1, R2 = R_init
    t1, t2 = t_init

    # choose a window near the end of the trajectory
    W = config.get("ba_window_size", 30)
    frame = min(num_frames - 1, W)
    start_frame = max(0, frame - W + 1)
    window_ids = list(range(start_frame, frame + 1))

    N_dyn = len(window_ids)
    N_pts_window = N_static + N_dyn

    cam_idxs = []
    p_idxs = []
    p_list = []

    # static observations -> indices 0..N_static-1
    num_stat_used = min(len(static_obs_per_cam[0][0]), 5)
    for cam_id in range(2):
        for j in range(N_static):
            obs_list = static_obs_per_cam[cam_id][j]
            for p_stat in obs_list[-num_stat_used:]:
                cam_idxs.append(cam_id)
                p_idxs.append(j)
                p_list.append(p_stat.reshape(2,))

    # dynamic observations -> indices N_static..N_static+N_dyn-1
    frame_to_local = {fid: (i + N_static) for i, fid in enumerate(window_ids)}

    for cam_id, obs in enumerate(obs_per_cam):
        for pid, p in obs.items():
            if pid < start_frame or pid > frame:
                continue
            cam_idxs.append(cam_id)
            p_idxs.append(frame_to_local[pid])
            p_list.append(p.reshape(2,))

    if len(p_list) < 4:
        # not enough constraints; return original poses
        return R_init, t_init, X_est

    cam_idxs = np.array(cam_idxs, dtype=int)
    p_idxs = np.array(p_idxs, dtype=int)
    p_obs_all = np.column_stack(p_list)

    # camera centers
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    c0_used = C1
    s_init = np.array([0.0, np.dot(C2 - c0_used, baseline_u)])

    rvec1, _ = cv.Rodrigues(R1)
    rvec2, _ = cv.Rodrigues(R2)
    rvecs_init = np.vstack([rvec1.ravel(), rvec2.ravel()])

    # build X_init: first N_static are static, rest are dynamic
    X_init = np.zeros((3, N_pts_window))
    X_init[:, :N_static] = X_static_est.T
    X_init[:, N_static:] = X_est[start_frame:frame+1].T

    # run your constrained BA
    rvecs_opt, s_opt, X_opt = ba.run_constrained_ba(
        K,
        baseline_u,
        c0_used,
        rvecs_init,
        s_init,
        X_init,
        X_static_true,
        cam_idxs,
        p_idxs,
        p_obs_all,
    )

    R1_new, t1_new = geom.pose_from_baseline(baseline_u, c0_used, rvecs_opt[0], s_opt[0])
    R2_new, t2_new = geom.pose_from_baseline(baseline_u, c0_used, rvecs_opt[1], s_opt[1])

    X_est_new = X_est.copy()
    X_est_new[start_frame:frame+1] = X_opt[:, N_static:].T

    return [R1_new, R2_new], [t1_new, t2_new], X_est_new


def run_single_experiment(seed, base_config):
    np.random.seed(seed)
    config = dict(base_config)

    # 1) cameras
    K, baseline_u, c0, R_true, t_true, R_init, t_init = setup_cameras(config)

    # 2) trajectory
    X_true = generate_circular_trajectory(config)

    # 3) static reference points (N_static arbitrary)
    X_static_true = np.array([
        [15.0, 15.0, 80.0],
        [20.0, 10.0, 78.0],
        [10.0, 20.0, 82.0],
        [18.0, 18.0, 79.0],
    ], dtype=float)

    # 4) observations
    obs_per_cam, static_obs_per_cam, X_est, X_static_est = \
        simulate_observations(K, baseline_u, R_init, t_init, X_true, X_static_true, config)

    # 5) error before BA
    rot_before, pos_before = pose_error(R_init, t_init, R_true, t_true)

    # 6) run BA once
    R_ba, t_ba, X_est_ba = run_ba_once(
        K, baseline_u, c0, R_init, t_init,
        obs_per_cam, static_obs_per_cam,
        X_est, X_static_est, X_static_true, config
    )

    # 7) error after BA
    rot_after, pos_after = pose_error(R_ba, t_ba, R_true, t_true)

    return rot_before, rot_after, pos_before, pos_after


def main():
    N_trials = CONFIG.get("experiment_trials", 100)

    rot_before_all = []
    rot_after_all = []
    pos_before_all = []
    pos_after_all = []

    print(f"Running {N_trials} experiments...")
    for seed in range(N_trials):
        rb, ra, pb, pa = run_single_experiment(seed, CONFIG)
        rot_before_all.append(rb)
        rot_after_all.append(ra)
        pos_before_all.append(pb)
        pos_after_all.append(pa)
        if (seed + 1) % 10 == 0:
            print(f"  Completed {seed + 1}/{N_trials}")

    rot_before = np.array(rot_before_all)
    rot_after = np.array(rot_after_all)
    pos_before = np.array(pos_before_all)
    pos_after = np.array(pos_after_all)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Rotation error [deg]:")
    print(f"  Before BA: mean = {rot_before.mean():.4f}, median = {np.median(rot_before):.4f}, std = {rot_before.std():.4f}")
    print(f"  After BA:  mean = {rot_after.mean():.4f}, median = {np.median(rot_after):.4f}, std = {rot_after.std():.4f}")
    print(f"  Improvement: {(rot_before.mean() - rot_after.mean()):.4f} deg")

    print(f"\nPosition error:")
    print(f"  Before BA: mean = {pos_before.mean():.4f}, median = {np.median(pos_before):.4f}, std = {pos_before.std():.4f}")
    print(f"  After BA:  mean = {pos_after.mean():.4f}, median = {np.median(pos_after):.4f}, std = {pos_after.std():.4f}")
    print(f"  Improvement: {(pos_before.mean() - pos_after.mean()):.4f}")
    print("="*60 + "\n")

    # boxplots
    fig, (ax_rot, ax_pos) = plt.subplots(1, 2, figsize=(12, 5))
    ax_rot.boxplot([rot_before, rot_after], labels=["before BA", "after BA"])
    ax_rot.set_title("Rotation error [deg]")
    ax_rot.set_ylabel("Error [deg]")
    ax_rot.grid(True, alpha=0.3)

    ax_pos.boxplot([pos_before, pos_after], labels=["before BA", "after BA"])
    ax_pos.set_title("Position error")
    ax_pos.set_ylabel("Error")
    ax_pos.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # scatter plots
    fig, (ax_rot_scatter, ax_pos_scatter) = plt.subplots(1, 2, figsize=(12, 5))

    # rotation scatter
    ax_rot_scatter.scatter(rot_before, rot_after, alpha=0.6, s=50)
    mn = min(rot_before.min(), rot_after.min())
    mx = max(rot_before.max(), rot_after.max())
    ax_rot_scatter.plot([mn, mx], [mn, mx], "k--", label="no improvement")
    ax_rot_scatter.set_xlabel("Rot error before BA [deg]")
    ax_rot_scatter.set_ylabel("Rot error after BA [deg]")
    ax_rot_scatter.set_title("Per-trial rotation error")
    ax_rot_scatter.legend()
    ax_rot_scatter.grid(True, alpha=0.3)

    # position scatter
    ax_pos_scatter.scatter(pos_before, pos_after, alpha=0.6, s=50, color='orange')
    mn = min(pos_before.min(), pos_after.min())
    mx = max(pos_before.max(), pos_after.max())
    ax_pos_scatter.plot([mn, mx], [mn, mx], "k--", label="no improvement")
    ax_pos_scatter.set_xlabel("Pos error before BA")
    ax_pos_scatter.set_ylabel("Pos error after BA")
    ax_pos_scatter.set_title("Per-trial position error")
    ax_pos_scatter.legend()
    ax_pos_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
