"""
trajectory_gen.py: Generate synthetic drone trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def generate_circular_trajectory(config):
    """
    Generate a circular drone trajectory in a plane perpendicular to camera baseline.
    ...
    """
    num_frames = config['num_frames']
    frame_rate = config['frame_rate']
    dt = 1.0 / frame_rate

    center_x = config['drone_center'][0]
    center_y = config['drone_center'][1]
    center_z = config['drone_center'][2]
    radius = config['drone_radius']
    angular_velocity = config['drone_angular_velocity']

    trajectory = np.zeros((num_frames, 3), dtype=np.float64)

    for frame_idx in range(num_frames):
        t = frame_idx * dt
        theta = angular_velocity * t

        x = center_x + radius * np.sin(theta)
        y = center_y + radius * np.cos(theta)
        z = center_z 

        trajectory[frame_idx] = [x, y, z]

    return trajectory


def get_trajectory_velocity_acceleration(trajectory, frame_rate):
    """
    Compute velocity and acceleration from trajectory.
    ...
    """
    dt = 1.0 / frame_rate
    num_frames = trajectory.shape[0]

    velocity = np.zeros_like(trajectory)
    velocity[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * dt)
    velocity[0] = velocity[1]
    velocity[-1] = velocity[-2]

    acceleration = np.zeros_like(trajectory)
    acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * dt)
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]

    return velocity, acceleration


def plot_trajectory_3d(trajectory, show=True, ax=None, color="C0", label="trajectory"):
    """
    Plot a 3D trajectory.

    Args:
        trajectory (np.ndarray): shape (N, 3) array of [x, y, z] points.
        show (bool): if True, calls plt.show() at the end.
        ax (mpl_toolkits.mplot3d.Axes3D or None): existing 3D axis to draw on.
        color (str): line color.
        label (str): label for the trajectory.

    Returns:
        ax: the 3D axis with the plot.
    """
    traj = np.asarray(trajectory)
    assert traj.ndim == 2 and traj.shape[1] == 3, "trajectory must be (N, 3)"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

    ax.plot(x, y, z, color=color, label=label)
    ax.scatter(x[0], y[0], z[0], color="green", s=50, label="start")
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=50, label="end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Equal-ish aspect ratio
    max_range = np.ptp(traj, axis=0).max()
    mid = traj.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2, m + max_range / 2)

    if show:
        plt.show()

    return ax

from noise import pnoise1

def smooth_random_walk_bounded(num_steps=1000, scale=0.02, speed=0.05, 
                                bounds=(0, 10, 0, 10), seed=None):
    if seed is not None:
        np.random.seed(seed)

    xmin, xmax, ymin, ymax = bounds

    # Random offsets for independent noise streams
    offset_angle = np.random.randint(0, 10000)

    # Start in the middle
    x, y = (xmax - xmin) / 2, (ymax - ymin) / 2
    x_positions = [x]
    y_positions = [y]

    # Initial direction
    angle = 0.0

    for i in range(1, num_steps):
        
        angle_change = pnoise1((i + offset_angle) * scale) * 0.5  # radians
        angle += angle_change

        x += np.cos(angle) * speed
        y += np.sin(angle) * speed

        if x < xmin:
            x = xmin
            angle = np.pi - angle
        elif x > xmax:
            x = xmax
            angle = np.pi - angle

        if y < ymin:
            y = ymin
            angle = -angle
        elif y > ymax:
            y = ymax
            angle = -angle

        x_positions.append(x)
        y_positions.append(y)

    return np.array(x_positions), np.array(y_positions)

'''
if __name__ == "__main__":
    x, y = smooth_random_walk_bounded(num_steps=2000, scale=0.02, speed=0.05, 
                                      bounds=(0, 10, 0, 10), seed=None)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color="blue")
    plt.scatter(x[0], y[0], color="green", label="Start", zorder=5)
    plt.scatter(x[-1], y[-1], color="red", label="End", zorder=5)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Smooth Random Walk (Bounded Drone Motion)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()
'''