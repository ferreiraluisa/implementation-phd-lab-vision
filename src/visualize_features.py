import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


""" 
This code provides visualization functions for 2D(in 2d space) and 3D human pose data, as well to test it out our reprojection function.
Used to test it out dataset_features.py output, which contains the 2048 array features,  2D GT joints, 3D joints and camera intrinsics (K) to calculate reprojection without distorsion.
Coded by Luisa Ferreira, 2026.
"""
H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

def _as_numpy(x):
    # helper: torch -> numpy if needed
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def project_with_K(P_cam, K):
    # P_cam: (N,3) in camera coordinates
    # K: (3,3)
    P_h = (K @ P_cam.T).T              # (N,3)
    uv = P_h[:, :2] / P_h[:, 2:3]      # divide by w (=Z scaled)
    return uv                          # (N,2)

def plot_batch_sample_2d_2dproj_3d_no_video(
    joints3d, joints2d, K,
    sample_idx=0, fps=10, point_size=18, line_width=2,
    invert_y_2d=True, equal_aspect=True
):
    """
    joints3d: (B,T,J,3)
    joints2d: (B,T,J,2)
    K:       (B,3,3)  (or (3,3) broadcastable)
    """

    joints3d = _as_numpy(joints3d)
    joints2d = _as_numpy(joints2d)
    K = _as_numpy(K)

    js3 = joints3d[sample_idx]  # (T,J,3)
    js2 = joints2d[sample_idx]  # (T,J,2)

    T, J, _ = js3.shape

    # If K is (3,3), broadcast it; if (B,3,3) pick sample
    if K.ndim == 2:
        K_s = K
    else:
        K_s = K[sample_idx]

    # project per frame
    proj2 = np.zeros_like(js2, dtype=np.float32)  # (T,J,2)
    for tt in range(T):
        proj2[tt] = project_with_K(js3[tt], K_s)

    # ---- figure: 3 panels ----
    fig = plt.figure(figsize=(15, 4))
    ax_gt  = fig.add_subplot(1, 3, 1)
    ax_prj = fig.add_subplot(1, 3, 2)
    ax_3d  = fig.add_subplot(1, 3, 3, projection="3d")

    ax_gt.set_title("GT 2D joints")
    ax_prj.set_title("Reprojected 2D joints")
    ax_3d.set_title("3D skeleton")

    # ---- 2D scatters ----
    scat_gt  = ax_gt.scatter(js2[0, :, 0], js2[0, :, 1], s=point_size)
    scat_prj = ax_prj.scatter(proj2[0, :, 0], proj2[0, :, 1], s=point_size)

    # ---- 2D skeleton lines ----
    lines2d_gt, lines2d_prj = [], []
    for a, b in H36M_EDGES:
        l_gt, = ax_gt.plot([js2[0, a, 0], js2[0, b, 0]],
                           [js2[0, a, 1], js2[0, b, 1]],
                           linewidth=line_width)
        l_pr, = ax_prj.plot([proj2[0, a, 0], proj2[0, b, 0]],
                            [proj2[0, a, 1], proj2[0, b, 1]],
                            linewidth=line_width)
        lines2d_gt.append(l_gt)
        lines2d_prj.append(l_pr)

    # ---- 2D axis limits (auto from data, consistent across frames) ----
    all2d = np.concatenate([js2.reshape(-1, 2), proj2.reshape(-1, 2)], axis=0)
    xmin, ymin = all2d.min(axis=0)
    xmax, ymax = all2d.max(axis=0)
    padx = 0.05 * (xmax - xmin + 1e-9)
    pady = 0.05 * (ymax - ymin + 1e-9)

    ax_gt.set_xlim(xmin - padx, xmax + padx)
    ax_gt.set_ylim(ymin - pady, ymax + pady)
    ax_prj.set_xlim(xmin - padx, xmax + padx)
    ax_prj.set_ylim(ymin - pady, ymax + pady)

    if invert_y_2d:
        ax_gt.invert_yaxis()
        ax_prj.invert_yaxis()

    if equal_aspect:
        ax_gt.set_aspect("equal", adjustable="box")
        ax_prj.set_aspect("equal", adjustable="box")

    ax_gt.grid(True, alpha=0.3)
    ax_prj.grid(True, alpha=0.3)

    # ---- 3D plot ----
    xs, ys, zs = js3[..., 0], js3[..., 1], js3[..., 2]

    def pad(a, b, p=0.05):
        r = (b - a) if (b > a) else 1.0
        return a - p * r, b + p * r

    ax_3d.set_xlim(*pad(xs.min(), xs.max()))
    ax_3d.set_ylim(*pad(ys.min(), ys.max()))
    ax_3d.set_zlim(*pad(zs.min(), zs.max()))
    ax_3d.view_init(elev=25, azim=290)

    scat3d = ax_3d.scatter(js3[0, :, 0], js3[0, :, 1], js3[0, :, 2], s=point_size)

    lines3d = []
    for a, b in H36M_EDGES:
        l3, = ax_3d.plot([js3[0, a, 0], js3[0, b, 0]],
                         [js3[0, a, 1], js3[0, b, 1]],
                         [js3[0, a, 2], js3[0, b, 2]],
                         linewidth=line_width)
        lines3d.append(l3)

    # ---- update ----
    def update(tt):
        # 2D scatters
        scat_gt.set_offsets(js2[tt])
        scat_prj.set_offsets(proj2[tt])

        # 2D lines
        for l_gt, (a, b) in zip(lines2d_gt, H36M_EDGES):
            l_gt.set_data([js2[tt, a, 0], js2[tt, b, 0]],
                          [js2[tt, a, 1], js2[tt, b, 1]])
        for l_pr, (a, b) in zip(lines2d_prj, H36M_EDGES):
            l_pr.set_data([proj2[tt, a, 0], proj2[tt, b, 0]],
                          [proj2[tt, a, 1], proj2[tt, b, 1]])

        # 3D scatter
        scat3d._offsets3d = (js3[tt, :, 0], js3[tt, :, 1], js3[tt, :, 2])

        # 3D lines
        for l3, (a, b) in zip(lines3d, H36M_EDGES):
            l3.set_data([js3[tt, a, 0], js3[tt, b, 0]],
                        [js3[tt, a, 1], js3[tt, b, 1]])
            l3.set_3d_properties([js3[tt, a, 2], js3[tt, b, 2]])

        return [scat_gt, scat_prj, scat3d] + lines2d_gt + lines2d_prj + lines3d

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    plt.tight_layout()
    plt.show()
    return anim
