import numpy as np
import geom

class PlotState:
    def __init__(self, ax, ax_rot, ax_pos,
                 X_true, R_true, t_true,
                 fr_true_1, fr_true_2,
                 est_lines_1, est_lines_2,
                 fr_scale):

        self.ax = ax
        self.ax_rot = ax_rot
        self.ax_pos = ax_pos

        self.X_true = X_true
        self.R_true = R_true
        self.t_true = t_true
        self.fr_scale = fr_scale

        self.true_line, = ax.plot([], [], [], "C0", label="true")
        self.est_line,  = ax.plot([], [], [], "C1", label="est")

        self.rot_err_cam1 = []
        self.rot_err_cam2 = []
        self.pos_err_cam1 = []
        self.pos_err_cam2 = []

        self.rot_line1, = ax_rot.plot([], [], label="cam1 rot")
        self.rot_line2, = ax_rot.plot([], [], label="cam2 rot")
        self.pos_line1, = ax_pos.plot([], [], label="cam1 pos")
        self.pos_line2, = ax_pos.plot([], [], label="cam2 pos")

        self.est_lines_1 = est_lines_1
        self.est_lines_2 = est_lines_2

    def update_frustum(self, lines, fr):
        c = fr[0]
        for i in range(4):
            lines[i].set_data([c[0], fr[i+1][0]],
                              [c[1], fr[i+1][1]])
            lines[i].set_3d_properties([c[2], fr[i+1][2]])

        idx = [1, 2, 3, 4, 1]
        lines[4].set_data(fr[idx, 0], fr[idx, 1])
        lines[4].set_3d_properties(fr[idx, 2])

    def update(self, frame, X_est, R1, t1, R2, t2):
        # trajectory
        self.true_line.set_data(self.X_true[:frame+1, 0],
                                self.X_true[:frame+1, 1])
        self.true_line.set_3d_properties(self.X_true[:frame+1, 2])

        self.est_line.set_data(X_est[:frame+1, 0],
                               X_est[:frame+1, 1])
        self.est_line.set_3d_properties(X_est[:frame+1, 2])

        # errors
        self.rot_err_cam1.append(
            geom.rotation_error_deg(R1, self.R_true[0])
        )
        self.rot_err_cam2.append(
            geom.rotation_error_deg(R2, self.R_true[1])
        )

        C1_est = -R1.T @ t1
        C2_est = -R2.T @ t2
        
        C1_gt  = -self.R_true[0].T @ self.t_true[0]
        C2_gt  = -self.R_true[1].T @ self.t_true[1]

        self.pos_err_cam1.append(np.linalg.norm(C1_est - C1_gt))
        self.pos_err_cam2.append(np.linalg.norm(C2_est - C2_gt))

        frames = np.arange(len(self.rot_err_cam1))
        self.rot_line1.set_data(frames, self.rot_err_cam1)
        self.rot_line2.set_data(frames, self.rot_err_cam2)
        self.pos_line1.set_data(frames, self.pos_err_cam1)
        self.pos_line2.set_data(frames, self.pos_err_cam2)

        self.ax_rot.relim(); self.ax_rot.autoscale_view()
        self.ax_pos.relim(); self.ax_pos.autoscale_view()

        # frustums
        fr1 = camera_frustum(R1, t1, self.fr_scale)
        fr2 = camera_frustum(R2, t2, self.fr_scale)

        self.update_frustum(self.est_lines_1, fr1)
        self.update_frustum(self.est_lines_2, fr2)


def camera_frustum(R, t, scale=2.0):
    
    # Camera frame frustum
    frustum_cam = np.array([
        [0, 0, 0],          # camera center
        [ 1,  1,  2],
        [-1,  1,  2],
        [-1, -1,  2],
        [ 1, -1,  2],
    ]) * scale
    # Transform to world frame
    frustum_world = (R.T @ frustum_cam.T).T + t
    
    return frustum_world

def plot_frustum(ax, fr, color="red", alpha=0.8):
    """
    fr: (5,3) array [center, 4 image-plane corners]
    returns: list of 5 Line3D objects
    """
    lines = []

    c = fr[0]
    # pyramid edges
    for i in range(4):
        l, = ax.plot(
            [c[0], fr[i+1][0]],
            [c[1], fr[i+1][1]],
            [c[2], fr[i+1][2]],
            color=color,
            alpha=alpha
        )
        lines.append(l)

    # image plane square
    idx = [1, 2, 3, 4, 1]
    l, = ax.plot(
        fr[idx, 0],
        fr[idx, 1],
        fr[idx, 2],
        color=color,
        alpha=alpha
    )
    lines.append(l)

    return lines

