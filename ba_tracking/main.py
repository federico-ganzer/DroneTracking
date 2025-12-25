import numpy as np
import matplotlib.pyplot as plt
from plotter import PlotState, camera_frustum, plot_frustum
import refinement as ba
import geom
from config import CONFIG
from trajectory_gen import generate_circular_trajectory
import cv2 as cv

n_cams = 2
c0 = np.array([0., 0., 0.])
baseline = np.array([0., 1., 0.])
baseline_u = baseline / np.linalg.norm(baseline)
f = CONFIG['focal_length']

K = np.array([[f, 0, 0],
              [0, f, 0],
              [0, 0, 1]], dtype=float)

s_true = np.array([0.0, 20.0])
rvec_true = np.zeros((n_cams, 3), dtype=float)

def initialise_camera_poses(s_true, rvec_true, baseline_dir, c0):
    R_true, t_true = [], [] # actual poses
    for i in range(n_cams):
        R_i, t_i = geom.pose_from_baseline(baseline_dir=baseline_dir,
                                           c0=c0,
                                           rvec=rvec_true[i],
                                           s=s_true[i])
        R_true.append(R_i)
        t_true.append(t_i)

    R_init = [geom.perturb_R(R, angle_sigma=np.deg2rad(CONFIG['rotation_noise_deg'])) for R in R_true]
    t_init = [geom.perturb_t(t, sigma=CONFIG['translation_noise']) for t in t_true]

    # current working poses (will be updated by BA)
    R1, R2 = R_true[0], R_init[1] 
    t1, t2 = t_true[0], t_init[1] # measured misclaibrated relative pose used

    return R1, R2, t1, t2, R_true, t_true

R1, R2, t1, t2, R_true, t_true = initialise_camera_poses(s_true, rvec_true, baseline_dir=baseline_u, c0=c0)

X_true = generate_circular_trajectory(CONFIG)
num_frames = CONFIG['num_frames']
X_est = np.zeros_like(X_true)

#region Plot Set
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
max_range = np.ptp(X_true, axis=0).max()
mid = X_true.mean(axis=0)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range / 2)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range / 2)
ax.set_zlim(0, mid[2] + max_range /2)
rot_err_cam1 = np.zeros(num_frames)
rot_err_cam2 = np.zeros(num_frames)
pos_err_cam1 = np.zeros(num_frames)
pos_err_cam2 = np.zeros(num_frames)
fig_err, (ax_rot, ax_pos) = plt.subplots(2, 1, sharex=True)
ax_rot.set_ylabel("Rot error [deg]")
ax_pos.set_ylabel("Pos error")
ax_pos.set_xlabel("Frame")

rot_line1, = ax_rot.plot([], [], color="C0", label="cam1 rot")
rot_line2, = ax_rot.plot([], [], color="C1", label="cam2 rot")

pos_line1, = ax_pos.plot([], [], color="C0", label="cam1 pos")
pos_line2, = ax_pos.plot([], [], color="C1", label="cam2 pos")

ax_rot.legend()
ax_pos.legend()

fr_scale = 2.0

fr_true_1 = camera_frustum(R_true[0], t_true[0], fr_scale)
fr_true_2 = camera_frustum(R_true[1], t_true[1], fr_scale)

fr_est_1  = camera_frustum(R1, t1, fr_scale)
fr_est_2  = camera_frustum(R2, t2, fr_scale)

true_lines_1 = plot_frustum(ax, fr_true_1, color="green", alpha=0.3)
true_lines_2 = plot_frustum(ax, fr_true_2, color="green", alpha=0.3)

est_lines_1  = plot_frustum(ax, fr_est_1,  color="red",   alpha=0.9)
est_lines_2  = plot_frustum(ax, fr_est_2,  color="red",   alpha=0.9)


#endregion

obs_per_cam = [dict() for _ in range(n_cams)]
id_to_idx = {}

#These are additional static points to be triangulated to stabilise and constrain BA. Various depths and positions need to be used
X_static_true = np.array([
    [-10, -10, 40],
    [-10,  10, 40],
    [ 10, -10, 40],
    [ 10,  10, 40],
    [  0,   0, 30],
    [  0,   0, 60],
], dtype=float)


def initialize_static_points(X_static_true):
    N_static = X_static_true.shape[0]
    static_obs_per_cam = [ [[] for _ in range(N_static)]  # cam 0
                           for _ in range(n_cams) ] 

    X_static_est = np.zeros_like(X_static_true)
    for j in range(N_static):
        Xs = X_static_true[j]
        p1_stat = geom.project_p(K, R_true[0], t_true[0], Xs)
        p2_stat = geom.project_p(K, R_true[1], t_true[1], Xs)
        if CONFIG['detection_noise']:
            noise = CONFIG['detection_noise']
            p1_stat += np.random.randn(2) * noise
            p2_stat += np.random.randn(2) * noise
        static_obs_per_cam[1][j].append(p2_stat.reshape(2, 1))
        static_obs_per_cam[0][j].append(p1_stat.reshape(2, 1))
        X_static_est[j] = geom.triangulate_points(K, R1, t1,
                                                  K, R2, t2,
                                                  p1_stat.reshape(2,1),
                                                  p2_stat.reshape(2,1))[:, 0]
        
    return X_static_est, N_static, static_obs_per_cam

X_static_est, N_static, static_obs_per_cam = initialize_static_points(X_static_true=X_static_true)

def create_3d_kalman(dt=1/CONFIG['frame_rate'], process_noise=0.01, meas_noise=2.0):
    kf = cv.KalmanFilter(6, 3)  # 6 states, 3 measurements

    # State: [x, y, z, vx, vy, vz]
    kf.transitionMatrix = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    # Measurement: [x, y, z]
    kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * meas_noise
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    kf.statePost = np.zeros((6,1), dtype=np.float32)

    return kf

kf = create_3d_kalman(dt=1.0)
kf_initialized = False

X_kf_est = np.zeros_like(X_est)  # store KF smoothed trajectory

def build_ba_window(obs_per_cam,
                    static_obs_per_cam,
                    X_est,
                    X_static_est,
                    start_frame,
                    frame,
                    N_static,
                    max_static_obs=5):

    cam_idxs, p_idxs, p_list = [], [], []

    # static points
    for cam_id in range(2):
        for j in range(N_static):
            obs_list = static_obs_per_cam[cam_id][j]
            for p in obs_list[-max_static_obs:]:
                cam_idxs.append(cam_id)
                p_idxs.append(j)
                p_list.append(p.reshape(2,))

    frame_to_local = {
        fid: i + N_static
        for i, fid in enumerate(range(start_frame, frame + 1))
    }

    # dynamic points
    for cam_id, obs in enumerate(obs_per_cam):
        for pid, p in obs.items():
            if start_frame <= pid <= frame:
                cam_idxs.append(cam_id)
                p_idxs.append(frame_to_local[pid])
                p_list.append(p.reshape(2,))

    return (np.array(cam_idxs),
            np.array(p_idxs),
            np.column_stack(p_list))

plotter = PlotState(ax, ax_rot, ax_pos,
                    X_true, R_true, t_true,
                    fr_true_1, fr_true_2,
                    est_lines_1, est_lines_2,
                    fr_scale)

for frame in range(num_frames):
    # observe
    Xw = X_true[frame]
    p1 = geom.project_p(K, R_true[0], t_true[0], Xw)
    p2 = geom.project_p(K, R_true[1], t_true[1], Xw)

    if CONFIG['detection_noise']:
        p1 += np.random.randn(2) * CONFIG['detection_noise']
        p2 += np.random.randn(2) * CONFIG['detection_noise']

    obs_per_cam[0][frame] = p1.reshape(2, 1)
    obs_per_cam[1][frame] = p2.reshape(2, 1)

    # triangulate
    X_est[frame] = geom.triangulate_points(
        K, R1, t1, K, R2, t2,
        obs_per_cam[0][frame],
        obs_per_cam[1][frame]
    )[:, 0]

    z = X_est[frame].reshape(3,1).astype(np.float32)
    
    if not kf_initialized:
        kf.statePost[:3, 0] = z[:, 0]
        kf.statePost[3:, 0] = 0.0
        kf_initialized = True
        
    else:
        kf.predict()
        kf.correct(z)
    
    X_kf_est[frame] = kf.statePost[:3, 0]
    
    plotter.update(frame, X_kf_est, R1, t1, R2, t2)

    if (frame > 0 and
        frame % CONFIG['ba_interval'] == 0 and
        CONFIG['refinement_enabled'] and
        frame < CONFIG['stop_refinement']):

        start = max(0, frame - CONFIG['ba_window_size'] + 1)

        cam_idxs, p_idxs, p_obs_all = build_ba_window(
            obs_per_cam,
            static_obs_per_cam,
            X_est,
            X_static_est,
            start,
            frame,
            N_static
        )

        if p_obs_all.shape[1] >= 4:
            C1 = -R1.T @ t1
            C2 = -R2.T @ t2

            rvecs_init = np.vstack([
                cv.Rodrigues(R1)[0].ravel(),
                cv.Rodrigues(R2)[0].ravel()
            ])

            s_init = np.array([0.0, np.dot(C2 - C1, baseline_u)])

            X_init = np.zeros((3, N_static + frame - start + 1))
            X_init[:, :N_static] = X_static_est.T
            X_init[:, N_static:] = X_kf_est[start:frame+1].T

            rvecs_opt, s_opt, X_opt = ba.run_constrained_ba(
                K, baseline_u, C1,
                rvecs_init, s_init,
                X_init, X_static_est,
                cam_idxs, p_idxs, p_obs_all
            )

            R1 = cv.Rodrigues(rvecs_opt[0])[0]
            R2 = cv.Rodrigues(rvecs_opt[1])[0]

            C1 = c0
            C2 = c0 + s_opt[1] * baseline_u

            t1 = -R1 @ C1
            t2 = -R2 @ C2

            X_kf_est[start:frame+1] = X_opt[:, N_static:].T

    plt.pause(0.01)


plt.ioff()
plt.show()

