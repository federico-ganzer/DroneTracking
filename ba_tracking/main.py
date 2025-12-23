import numpy as np
import matplotlib.pyplot as plt
from camera_model import PinholeCamera, plot_frustum, camera_frustum
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

def initialise_camera_poses(K, s_true, rvec_true, baseline_dir, c0):
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

R1, R2, t1, t2, R_true, t_true = initialise_camera_poses(K, s_true, rvec_true, baseline_dir=baseline_u, c0=c0)

X_true = generate_circular_trajectory(CONFIG)
num_frames = CONFIG['num_frames']
X_est = np.zeros_like(X_true)

#region Plot Set
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
true_line, = ax.plot([], [], [], color="C0", label="true trajectory")
est_line,  = ax.plot([], [], [], color="C1", label="estimated trajectory")
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
rot_line1, = ax_rot.plot([], [], label="cam1 rot")
rot_line2, = ax_rot.plot([], [], label="cam2 rot")
pos_line1, = ax_pos.plot([], [], label="cam1 pos")
pos_line2, = ax_pos.plot([], [], label="cam2 pos")
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
X_static_true = np.array([[15.0, 15.0, 80.0],
                          [20.0, 10.0, 75.0],
                          [10.0, 20.0, 82.0],
                          [18.0, 18.0, 79.0],
                          [0, 0, 100],
                          [5, 5, 50]], dtype=float) 

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

for frame in range(num_frames):
    Xw = X_true[frame]

    # project with current estimated poses
    p1 = geom.project_p(K, R_true[0], t_true[0], Xw)
    p2 = geom.project_p(K, R_true[1], t_true[1], Xw)

    if CONFIG['detection_noise']:
        noise = CONFIG['detection_noise']
        p1 += np.random.randn(2) * noise
        p2 += np.random.randn(2) * noise

    p1 = p1.reshape(2, 1)
    p2 = p2.reshape(2, 1)
    
    pid = frame
    id_to_idx[pid] = frame
    obs_per_cam[0][pid] = p1
    obs_per_cam[1][pid] = p2
    
    # simple triangulation from current poses
    X_tri = geom.triangulate_points(K, R1, t1, K, R2, t2, p1, p2)
    X_est[frame] = X_tri[:, 0]
    z = X_tri[:, 0]
    X_est[frame] = z
    
    def plot_update():
        true_line.set_data(X_true[:frame+1, 0], X_true[:frame+1, 1])
        true_line.set_3d_properties(X_true[:frame+1, 2])

        est_line.set_data(X_est[:frame+1, 0], X_est[:frame+1, 1])
        est_line.set_3d_properties(X_est[:frame+1, 2])

        rot_err_cam1[frame] = geom.rotation_error_deg(R1, R_true[0])
        rot_err_cam2[frame] = geom.rotation_error_deg(R2, R_true[1])

        C1_est = -R1.T @ t1
        C2_est = -R2.T @ t2
        C1_gt  = -R_true[0].T @ t_true[0]
        C2_gt  = -R_true[1].T @ t_true[1]

        pos_err_cam1[frame] = np.linalg.norm(C1_est - C1_gt)
        pos_err_cam2[frame] = np.linalg.norm(C2_est - C2_gt)

        # update error plots
        frames = np.arange(frame + 1)
        rot_line1.set_data(frames, rot_err_cam1[:frame+1])
        rot_line2.set_data(frames, rot_err_cam2[:frame+1])
        pos_line1.set_data(frames, pos_err_cam1[:frame+1])
        pos_line2.set_data(frames, pos_err_cam2[:frame+1])

        ax_rot.relim(); ax_rot.autoscale_view()
        ax_pos.relim(); ax_pos.autoscale_view()
        
        fr_est_1 = camera_frustum(R1, t1, fr_scale)
        fr_est_2 = camera_frustum(R2, t2, fr_scale)

        def update_frustum(lines, fr):
            c = fr[0]
            for i in range(4):
                lines[i].set_data([c[0], fr[i+1][0]],
                                  [c[1], fr[i+1][1]])
                lines[i].set_3d_properties([c[2], fr[i+1][2]])

            idx = [1, 2, 3, 4, 1]
            lines[4].set_data(fr[idx, 0], fr[idx, 1])
            lines[4].set_3d_properties(fr[idx, 2])

        update_frustum(est_lines_1, fr_est_1)
        update_frustum(est_lines_2, fr_est_2)
    plot_update()

    if (frame > 0 and frame % CONFIG['ba_interval'] == 0
            and CONFIG['refinement_enabled']
            and frame < CONFIG['stop_refinement']):

        
        start_frame = max(0, frame - CONFIG['ba_window_size'] + 1)
        window_ids = [i for i in range(start_frame, frame + 1)]

        N_dyn = len(window_ids)
        N_pts_window = N_dyn + N_static
        
        cam_idxs = []
        p_idxs = []
        p_list = []

        num_stat_used = min(len(static_obs_per_cam[0][0]), 5)
        for cam_id in range(n_cams):
            for j in range(N_static):
                obs_list = static_obs_per_cam[cam_id][j]
                for p_stat in obs_list[-num_stat_used:]:
                    cam_idxs.append(cam_id)
                    p_idxs.append(j)                      
                    p_list.append(p_stat.reshape(2,))
        
        
        frame_to_local = {fid: (i + N_static) for i, fid in enumerate(window_ids)}
        
        for cam_id, obs in enumerate(obs_per_cam):
            for pid, p in obs.items():
                if pid < start_frame or pid > frame:
                    continue
                cam_idxs.append(cam_id)
                p_idxs.append(frame_to_local[pid])
                p_list.append(p.reshape(2,))

        if len(p_list) >= 4:  # need some constraints
            cam_idxs = np.array(cam_idxs, dtype=int)
            p_idxs = np.array(p_idxs, dtype=int)
            p_obs_all = np.column_stack(p_list)  # (2, M)

            C1 = -R1.T @ t1
            C2 = -R2.T @ t2

            c0_used = C1
            s_init = np.array([0.0, np.dot(C2 - c0_used, baseline_u)])

            rvec1, _ = cv.Rodrigues(R1)
            rvec2, _ = cv.Rodrigues(R2)
            rvecs_init = np.vstack([rvec1.ravel(), rvec2.ravel()])

            # Build X_init
            X_init = np.zeros((3, N_pts_window))
            X_init[:, :N_static] = X_static_est.T
            X_init[:, N_static:] = X_est[start_frame:frame+1].T  # (3, N_pts_window)

            rvecs_opt, s_opt, X_opt = ba.run_constrained_ba(
                K,
                baseline_u,
                c0_used,
                rvecs_init,
                s_init,
                X_init,
                X_static_est,
                cam_idxs,
                p_idxs,
                p_obs_all,
            )

            R1, t1 = geom.pose_from_baseline(baseline_u, c0_used, rvecs_opt[0], s_opt[0])
            R2, t2 = geom.pose_from_baseline(baseline_u, c0_used, rvecs_opt[1], s_opt[1])
            
            X_est[start_frame:frame+1] = X_opt[:, N_static:].T

    plt.pause(0.01)

plt.ioff()
plt.show()

