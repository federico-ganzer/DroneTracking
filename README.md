# Stereo Bundle Adjustment Proof of Concept

This repository contains a small simulation framework and proof-of-concept of constrained Bundle Adjustment (BA) on stereo trajectory reconstruction using simulated data. It simulates a drone flying on a circular path observed by a stereo rig with noisy calibration and image measurements, and evaluates how much BA improves camera poses and 3D reconstruction.

The code is organized so that `main.py` runs an interactive single‑run visualization, while `ba_experiment.py` runs multiple Monte‑Carlo trials and produces quantitative comparison plots between BA and a simple non‑BA triangulation baseline.

## Features
- Synthetic circular “drone” trajectory in 3D with configurable radius, height, angular velocity, and frame rate. (Random path has not been tested yet)
- Noisy initialization of the stereo rig (rotation and translation perturbations) and noisy image measurements to mimic real‑world calibration and detection errors.
- Constrained BA that:
  - optimizes camera pose increments along a known baseline direction with reguralizers (Uses temporal smoothness and pose increment penalty). 
  - refines dynamic 3D points while keeping static anchor points fixed,
  - uses robust `scipy.optimize.least_squares` with Huber loss. 
- Benchmark scripts that compare BA vs. a non‑BA triangulation pipeline over many trials, and visualize trajectory, pose, and runtime statistics.

## Code Overview

- **`main.py`**  
  Runs a single stereo reconstruction with BA enabled. It:
  - sets up the stereo rig from a baseline parameterization (`pose_from_baseline`),
  - generates a circular 3D trajectory with `generate_circular_trajectory`,
  - projects points into both cameras with noise and triangulates them frame‑by‑frame,
  - periodically builds a sliding BA window (static + recent dynamic points) and calls `run_constrained_ba` to refine the right‑camera pose and the structure,
  - visualizes the true and estimated trajectory, the camera frustums, and per‑frame rotation/position errors in real time via `PlotState`.

- **`ba_experiment.py`** 
  
  Provides a full experiment driver to *benchmark* BA against a non‑BA baseline:
  - `run_single_trial` runs one noisy sequence, either with BA enabled or with refinement disabled, and records:
    - per‑frame trajectory error,
    - rotation and position error for each camera,
    - times at which BA is run, and the final mean error over the last frames.
  - `run_experiment` runs multiple trials with and without BA using the same configuration and different seeds, measuring total runtime per trial.
  - `plot_comparison` aggregates all trials and produces the summary figure (`ba_comparison_full.jpg`) showing:
    - mean ± std trajectory error over time,
    - mean ± std camera‑2 rotation and position error,
    - final trajectory error distribution (boxplots),
    - cumulative trajectory error with and without BA,
    - frame‑wise percentage improvement from BA,
    - example 3D trajectory (ground truth vs BA vs no‑BA),
    - computation‑time boxplots for BA vs non‑BA, plus a textual statistical summary.

- **`refinement.py`**  
  Implements the constrained BA solver:
  - packs/unpacks parameter vectors consisting of camera rotation increments, baseline parameters, and dynamic 3D points,
  - defines the residual function combining reprojection error, temporal smoothness of dynamic points, and pose regularization,
  - calls `scipy.optimize.least_squares` with robust loss and returns optimized poses and structure.

- **`trajectory_gen.py`**  
  Generates the ground‑truth circular drone trajectory given the kinematic parameters in `CONFIG`.

- **`geom.py`**  
  Geometry utilities: pose construction from baseline, Rodrigues conversions, projection, triangulation, and rotation‑error computation.

- **`plotter.py`**  
  Real‑time visualization helper used by `main.py` to update the 3D trajectory, camera frustums, and error plots as the sequence progresses.

- **`background_subtraction.py`** (optional)  
  Contains utilities for simple foreground/background separation in real image streams, which can be used when extending the framework to real data.

- **`config.py`**  
  Central configuration dictionary controlling camera intrinsics, baseline length, trajectory parameters, noise levels, and BA scheduling (window size, interval, refinement stop frame).

## Usage

### Single visual run (interactive BA)

**`main.py`**

This opens interactive Matplotlib windows that show the evolving 3D trajectory, camera poses, and rotation/position errors while BA refines the right‑camera pose over time.

### Benchmark BA vs non‑BA

**`ba_experiment.py`**

This runs multiple trials with and without BA using the same configuration, then saves a comprehensive comparison figure (e.g. `ba_comparison_full.jpg`) and prints a textual statistical summary of accuracy and runtime.

To adjust the scenario (noise levels, trajectory, BA window size, number of frames, etc.), edit the values in `config.py`.

## Dependencies

- Python 3.x  
- NumPy  
- SciPy (for `optimize.least_squares`)  
- OpenCV (`cv2`)
- Matplotlib  

Install them with:

pip install numpy scipy opencv-python matplotlib



This repository is intended as a compact, reproducible proof of concept, desined for experimenting with constrained BA in a simulated stereo setting. 
