import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from copy import deepcopy
import time

import geom
import refinement as ba
from main import (CONFIG, K, baseline_u, c0, n_cams, 
                  generate_circular_trajectory, camera_frustum,
                  initialise_camera_poses, build_ba_window, create_3d_kalman)


def run_single_trial(config, trial_num=0, verbose=False):
    """
    Run a single trial with given configuration
    Returns metrics: trajectory errors, pose errors
    """
    np.random.seed(trial_num)  # For reproducibility
    
    num_frames = config['num_frames']
    
    # Initialize true trajectory
    X_true = generate_circular_trajectory(config)
    X_est = np.zeros_like(X_true)
    
    # Initialize camera poses
    s_true = np.array([0.0, 20.0])
    rvec_true = np.zeros((n_cams, 3), dtype=float)
    
    R1, R2, t1, t2, R_true, t_true = initialise_camera_poses(
        s_true, rvec_true, baseline_dir=baseline_u, c0=c0
    )
    
    # Initialize static points
    X_static_true = np.array([
        [-10, -10, 40],
        [-10,  10, 40],
        [ 10, -10, 40],
        [ 10,  10, 40],
        [  0,   0, 30],
        [  0,   0, 60],
    ], dtype=float)

    N_static = X_static_true.shape[0]
    static_obs_per_cam = [[[] for _ in range(N_static)] for _ in range(n_cams)]
    X_static_est = np.zeros_like(X_static_true)
    
    # Project and triangulate static points
    for j in range(N_static):
        Xs = X_static_true[j]
        p1_stat = geom.project_p(K, R_true[0], t_true[0], Xs)
        p2_stat = geom.project_p(K, R_true[1], t_true[1], Xs)
        
        if config['detection_noise']:
            noise = config['detection_noise']
            p1_stat += np.random.randn(2) * noise
            p2_stat += np.random.randn(2) * noise
        
        static_obs_per_cam[0][j].append(p1_stat.reshape(2, 1))
        static_obs_per_cam[1][j].append(p2_stat.reshape(2, 1))
        X_static_est[j] = geom.triangulate_points(
            K, R1, t1, K, R2, t2,
            p1_stat.reshape(2, 1),
            p2_stat.reshape(2, 1)
        )[:, 0]
    
    # Initialize Kalman filter
    kf = create_3d_kalman(dt=1.0)
    X_kf_est = np.zeros_like(X_est)
    kf_initialized = False
    
    # Tracking arrays
    obs_per_cam = [dict() for _ in range(n_cams)]
    traj_errors = np.zeros(num_frames)
    rot_errors = np.zeros((num_frames, 2))
    pos_errors = np.zeros((num_frames, 2))
    ba_refinements = []
    
    # Main loop
    for frame in range(num_frames):
        # Observe
        Xw = X_true[frame]
        p1 = geom.project_p(K, R_true[0], t_true[0], Xw)
        p2 = geom.project_p(K, R_true[1], t_true[1], Xw)
        
        if config['detection_noise']:
            p1 += np.random.randn(2) * config['detection_noise']
            p2 += np.random.randn(2) * config['detection_noise']
        
        obs_per_cam[0][frame] = p1.reshape(2, 1)
        obs_per_cam[1][frame] = p2.reshape(2, 1)
        
        # Triangulate
        X_est[frame] = geom.triangulate_points(
            K, R1, t1, K, R2, t2,
            obs_per_cam[0][frame],
            obs_per_cam[1][frame]
        )[:, 0]
        
        # Apply Kalman filter
        z = X_est[frame].reshape(3, 1).astype(np.float32)
        if not kf_initialized:
            kf.statePost[:3, 0] = z[:, 0]
            kf.statePost[3:, 0] = 0.0
            kf_initialized = True
        else:
            kf.predict()
            kf.correct(z)
        
        X_kf_est[frame] = kf.statePost[:3, 0]
        
        # Compute errors using Kalman-filtered estimates
        traj_errors[frame] = np.linalg.norm(X_kf_est[frame] - X_true[frame])
        
        for cam_id in range(2):
            R_curr = R1 if cam_id == 0 else R2
            t_curr = t1 if cam_id == 0 else t2
            R_gt = R_true[cam_id]
            t_gt = t_true[cam_id]
            
            # Rotation error (Frobenius norm)
            R_err = R_curr @ R_gt.T
            rot_errors[frame, cam_id] = np.rad2deg(
                np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            )
            
            # Position error
            C_curr = -R_curr.T @ t_curr
            C_gt = -R_gt.T @ t_gt
            pos_errors[frame, cam_id] = np.linalg.norm(C_curr - C_gt)
        
        # Bundle Adjustment
        if (frame > 0 and 
            frame % config['ba_interval'] == 0 and
            config['refinement_enabled'] and
            frame < config.get('stop_refinement', num_frames)):
            
            start = max(0, frame - config['ba_window_size'] + 1)
            
            # Build BA window
            cam_idxs, p_idxs, p_obs_all = build_ba_window(
                obs_per_cam, static_obs_per_cam,
                X_kf_est, X_static_est,
                start, frame, N_static
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
                ba_refinements.append(frame)
                
                if verbose:
                    print(f"  BA at frame {frame}")
    
    return {
        'traj_errors': traj_errors,
        'rot_errors': rot_errors,
        'pos_errors': pos_errors,
        'ba_refinements': ba_refinements,
        'X_est': X_kf_est,
        'X_true': X_true,
        'final_mean_error': np.mean(traj_errors[-10:])  # Last 10 frames
    }


def run_experiment(n_trials=10, config_base=None):
    """
    Run complete experiment comparing BA vs non-BA
    """
    if config_base is None:
        config_base = deepcopy(CONFIG)
    
    print(f"Running experiment with {n_trials} trials...")
    print(f"Configuration: {config_base}")
    
    results_with_ba = []
    results_without_ba = []
    
    # Run trials WITH BA
    print("\n=== Running trials WITH Bundle Adjustment ===")
    config_ba = deepcopy(config_base)
    config_ba['refinement_enabled'] = True
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials} (BA enabled)")
        start_time = time.time()
        result = run_single_trial(config_ba, trial_num=trial, verbose=False)
        elapsed = time.time() - start_time
        result['time'] = elapsed
        results_with_ba.append(result)
    
    # Run trials WITHOUT BA
    print("\n=== Running trials WITHOUT Bundle Adjustment ===")
    config_no_ba = deepcopy(config_base)
    config_no_ba['refinement_enabled'] = False
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials} (BA disabled)")
        start_time = time.time()
        result = run_single_trial(config_no_ba, trial_num=trial, verbose=False)
        elapsed = time.time() - start_time
        result['time'] = elapsed
        results_without_ba.append(result)
    
    print("\n=== Experiment Complete ===")
    
    return results_with_ba, results_without_ba


def plot_comparison(results_with_ba, results_without_ba, save_prefix='ba_comparison'):
    """
    Create comprehensive comparison plots (without 3D trajectory plot)
    """
    n_trials = len(results_with_ba)
    num_frames = len(results_with_ba[0]['traj_errors'])
    
    # Setup figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Mean Trajectory Error Over Time
    ax1 = plt.subplot(2, 3, 1)
    traj_ba = np.array([r['traj_errors'] for r in results_with_ba])
    traj_no_ba = np.array([r['traj_errors'] for r in results_without_ba])
    
    mean_ba = np.mean(traj_ba, axis=0)
    std_ba = np.std(traj_ba, axis=0)
    mean_no_ba = np.mean(traj_no_ba, axis=0)
    std_no_ba = np.std(traj_no_ba, axis=0)
    
    frames = np.arange(num_frames)
    ax1.plot(frames, mean_ba, 'b-', label='With BA', linewidth=2)
    ax1.fill_between(frames, mean_ba - std_ba, mean_ba + std_ba, alpha=0.3, color='b')
    ax1.plot(frames, mean_no_ba, 'r-', label='Without BA', linewidth=2)
    ax1.fill_between(frames, mean_no_ba - std_no_ba, mean_no_ba + std_no_ba, alpha=0.3, color='r')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Trajectory Error (units)')
    ax1.set_title('Mean Trajectory Error Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rotation Error - Camera 2
    ax2 = plt.subplot(2, 3, 2)
    rot_ba_cam2 = np.array([r['rot_errors'][:, 1] for r in results_with_ba])
    rot_no_ba_cam2 = np.array([r['rot_errors'][:, 1] for r in results_without_ba])
    
    mean_rot_ba = np.mean(rot_ba_cam2, axis=0)
    std_rot_ba = np.std(rot_ba_cam2, axis=0)
    mean_rot_no_ba = np.mean(rot_no_ba_cam2, axis=0)
    std_rot_no_ba = np.std(rot_no_ba_cam2, axis=0)
    
    ax2.plot(frames, mean_rot_ba, 'b-', label='With BA', linewidth=2)
    ax2.fill_between(frames, mean_rot_ba - std_rot_ba, mean_rot_ba + std_rot_ba, 
                     alpha=0.3, color='b')
    ax2.plot(frames, mean_rot_no_ba, 'r-', label='Without BA', linewidth=2)
    ax2.fill_between(frames, mean_rot_no_ba - std_rot_no_ba, mean_rot_no_ba + std_rot_no_ba,
                     alpha=0.3, color='r')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Rotation Error (deg)')
    ax2.set_title('Camera 2 Rotation Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position Error - Camera 2
    ax3 = plt.subplot(2, 3, 3)
    pos_ba_cam2 = np.array([r['pos_errors'][:, 1] for r in results_with_ba])
    pos_no_ba_cam2 = np.array([r['pos_errors'][:, 1] for r in results_without_ba])
    
    mean_pos_ba = np.mean(pos_ba_cam2, axis=0)
    std_pos_ba = np.std(pos_ba_cam2, axis=0)
    mean_pos_no_ba = np.mean(pos_no_ba_cam2, axis=0)
    std_pos_no_ba = np.std(pos_no_ba_cam2, axis=0)
    
    ax3.plot(frames, mean_pos_ba, 'b-', label='With BA', linewidth=2)
    ax3.fill_between(frames, mean_pos_ba - std_pos_ba, mean_pos_ba + std_pos_ba,
                     alpha=0.3, color='b')
    ax3.plot(frames, mean_pos_no_ba, 'r-', label='Without BA', linewidth=2)
    ax3.fill_between(frames, mean_pos_no_ba - std_pos_no_ba, mean_pos_no_ba + std_pos_no_ba,
                     alpha=0.3, color='r')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position Error (units)')
    ax3.set_title('Camera 2 Position Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Error Distribution
    ax4 = plt.subplot(2, 3, 4)
    final_errors_ba = [r['final_mean_error'] for r in results_with_ba]
    final_errors_no_ba = [r['final_mean_error'] for r in results_without_ba]
    
    bp = ax4.boxplot([final_errors_ba, final_errors_no_ba],
                      labels=['With BA', 'Without BA'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Final Mean Error (last 10 frames)')
    ax4.set_title('Final Trajectory Error Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Computation Time Comparison
    ax5 = plt.subplot(2, 3, 5)
    times_ba = [r['time'] for r in results_with_ba]
    times_no_ba = [r['time'] for r in results_without_ba]
    
    bp2 = ax5.boxplot([times_ba, times_no_ba],
                       labels=['With BA', 'Without BA'],
                       patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax5.set_ylabel('Computation Time (s)')
    ax5.set_title('Computation Time Comparison')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistical Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    improvement_pct = ((np.mean(final_errors_no_ba) - np.mean(final_errors_ba)) / 
                       np.mean(final_errors_no_ba) * 100)
    
    summary_text = f"""
    STATISTICAL SUMMARY
    {'='*40}
    Number of Trials: {n_trials}
    Number of Frames: {num_frames}
    
    Final Mean Error (last 10 frames):
    With BA:    {np.mean(final_errors_ba):.3f} ± {np.std(final_errors_ba):.3f}
    Without BA: {np.mean(final_errors_no_ba):.3f} ± {np.std(final_errors_no_ba):.3f}
    
    Improvement: {improvement_pct:.1f}%
    
    Overall Mean Trajectory Error:
    With BA:    {np.mean(mean_ba):.3f}
    Without BA: {np.mean(mean_no_ba):.3f}
    
    Final Rotation Error (cam 2):
    With BA:    {mean_rot_ba[-1]:.3f}° ± {std_rot_ba[-1]:.3f}°
    Without BA: {mean_rot_no_ba[-1]:.3f}° ± {std_rot_no_ba[-1]:.3f}°
    
    Computation Time:
    With BA:    {np.mean(times_ba):.2f}s ± {np.std(times_ba):.2f}s
    Without BA: {np.mean(times_no_ba):.2f}s ± {np.std(times_no_ba):.2f}s
    """
    
    ax6.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_full.png', dpi=300, bbox_inches='tight')
    
    # Print summary to console
    print(summary_text)
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Run experiment
    N_TRIALS = 10
    
    results_ba, results_no_ba = run_experiment(n_trials=N_TRIALS, config_base=CONFIG)
    
    # Generate plots
    plot_comparison(results_ba, results_no_ba, save_prefix='ba_comparison')
    
    print("\nExperiment complete!")
