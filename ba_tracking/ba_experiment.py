"""
Experiment script to compare Bundle Adjustment (BA) enabled vs disabled performance
Runs multiple trials and generates comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from copy import deepcopy
import time

# Import your modules (adjust as needed)
import geom
import refinement as ba
from main import (CONFIG, K, baseline_u, c0, n_cams, 
                  generate_circular_trajectory, camera_frustum)


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
    
    # True poses
    R_true, t_true = [], []
    for i in range(n_cams):
        R_i, t_i = geom.pose_from_baseline(
            baseline_dir=baseline_u,
            c0=c0,
            rvec=rvec_true[i],
            s=s_true[i]
        )
        R_true.append(R_i)
        t_true.append(t_i)
    
    # Initialize with noise
    R_init = [geom.perturb_R(R, angle_sigma=np.deg2rad(config['rotation_noise_deg'])) 
              for R in R_true]
    t_init = [geom.perturb_t(t, sigma=config['translation_noise']) 
              for t in t_true]
    
    # Working poses
    R1, R2 = R_true[0], R_init[1]
    t1, t2 = t_true[0], t_init[1]
    
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
        
        # Compute errors
        traj_errors[frame] = np.linalg.norm(X_est[frame] - X_true[frame])
        
        for cam_id in range(2):
            R_curr = R1 if cam_id == 0 else R2
            t_curr = t1 if cam_id == 0 else t2
            R_gt = R_true[cam_id]
            t_gt = t_true[cam_id]
            
            # Rotation error (Frobenius norm)
            R_err = R_curr @ R_gt.T
            rot_errors[frame, cam_id] = np.rad2deg(
                np.arccos((np.trace(R_err) - 1) / 2)
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
                X_est, X_static_est,
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
                X_init[:, N_static:] = X_est[start:frame+1].T
                
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
                
                X_est[start:frame+1] = X_opt[:, N_static:].T
                ba_refinements.append(frame)
                
                if verbose:
                    print(f"  BA at frame {frame}")
    
    return {
        'traj_errors': traj_errors,
        'rot_errors': rot_errors,
        'pos_errors': pos_errors,
        'ba_refinements': ba_refinements,
        'X_est': X_est,
        'X_true': X_true,
        'final_mean_error': np.mean(traj_errors[-10:])  # Last 10 frames
    }


def build_ba_window(obs_per_cam, static_obs_per_cam, X_est, X_static_est,
                   start_frame, frame, N_static, max_static_obs=5):
    """Build BA observation window"""
    cam_idxs, p_idxs, p_list = [], [], []
    
    # Static points
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
    
    # Dynamic points
    for cam_id, obs in enumerate(obs_per_cam):
        for pid, p in obs.items():
            if start_frame <= pid <= frame:
                cam_idxs.append(cam_id)
                p_idxs.append(frame_to_local[pid])
                p_list.append(p.reshape(2,))
    
    return (np.array(cam_idxs), np.array(p_idxs), np.column_stack(p_list))


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
    Create comprehensive comparison plots
    """
    n_trials = len(results_with_ba)
    num_frames = len(results_with_ba[0]['traj_errors'])
    
    # Setup figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Mean Trajectory Error Over Time
    ax1 = plt.subplot(3, 3, 1)
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
    ax2 = plt.subplot(3, 3, 2)
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
    ax3 = plt.subplot(3, 3, 3)
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
    ax4 = plt.subplot(3, 3, 4)
    final_errors_ba = [r['final_mean_error'] for r in results_with_ba]
    final_errors_no_ba = [r['final_mean_error'] for r in results_without_ba]
    
    positions = [1, 2]
    bp = ax4.boxplot([final_errors_ba, final_errors_no_ba],
                      labels=['With BA', 'Without BA'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Final Mean Error (last 10 frames)')
    ax4.set_title('Final Trajectory Error Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative Error
    ax5 = plt.subplot(3, 3, 5)
    cum_ba = np.cumsum(mean_ba)
    cum_no_ba = np.cumsum(mean_no_ba)
    ax5.plot(frames, cum_ba, 'b-', label='With BA', linewidth=2)
    ax5.plot(frames, cum_no_ba, 'r-', label='Without BA', linewidth=2)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Cumulative Error')
    ax5.set_title('Cumulative Trajectory Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Improvement Percentage
    ax6 = plt.subplot(3, 3, 6)
    improvement = ((mean_no_ba - mean_ba) / mean_no_ba) * 100
    ax6.plot(frames, improvement, 'g-', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('BA Error Reduction (% improvement)')
    ax6.grid(True, alpha=0.3)
    ax6.fill_between(frames, 0, improvement, where=(improvement > 0), 
                     alpha=0.3, color='g', label='Improvement')
    ax6.fill_between(frames, 0, improvement, where=(improvement < 0),
                     alpha=0.3, color='r', label='Degradation')
    ax6.legend()
    
    # 7. Example Trajectory (first trial)
    ax7 = plt.subplot(3, 3, 7, projection='3d')
    X_true = results_with_ba[0]['X_true']
    X_est_ba = results_with_ba[0]['X_est']
    X_est_no_ba = results_without_ba[0]['X_est']
    
    ax7.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], 
             'k-', label='Ground Truth', linewidth=2)
    ax7.plot(X_est_ba[:, 0], X_est_ba[:, 1], X_est_ba[:, 2],
             'b--', label='With BA', linewidth=1.5, alpha=0.7)
    ax7.plot(X_est_no_ba[:, 0], X_est_no_ba[:, 1], X_est_no_ba[:, 2],
             'r--', label='Without BA', linewidth=1.5, alpha=0.7)
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    ax7.set_title('Example Trajectory (Trial 1)')
    ax7.legend()
    
    # 8. Computation Time Comparison
    ax8 = plt.subplot(3, 3, 8)
    times_ba = [r['time'] for r in results_with_ba]
    times_no_ba = [r['time'] for r in results_without_ba]
    
    bp2 = ax8.boxplot([times_ba, times_no_ba],
                       labels=['With BA', 'Without BA'],
                       patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax8.set_ylabel('Computation Time (s)')
    ax8.set_title('Computation Time Comparison')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Statistical Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    STATISTICAL SUMMARY
    {'='*40}
    Number of Trials: {n_trials}
    Number of Frames: {num_frames}
    
    Final Mean Error (last 10 frames):
    With BA:    {np.mean(final_errors_ba):.3f} ± {np.std(final_errors_ba):.3f}
    Without BA: {np.mean(final_errors_no_ba):.3f} ± {np.std(final_errors_no_ba):.3f}
    
    Improvement: {((np.mean(final_errors_no_ba) - np.mean(final_errors_ba)) / np.mean(final_errors_no_ba) * 100):.1f}%
    
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
    
    ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_full.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {save_prefix}_full.png")
    
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
