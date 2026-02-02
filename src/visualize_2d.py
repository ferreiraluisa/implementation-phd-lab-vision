import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
This code provides visualization functions for 2D and 3D human pose data, as well to test it out our reprojection function. 
video sequence with 2D GT plot + 2D projected plot + 3D plot.
Coded by Luisa Ferreira, 2026.
"""

H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# option 2: with radial distortion
def project_point_radial(P, R, t, f, c, all_k):
    # FUNCTION FROM https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/h36/read_human36m.py#L449
    # (LUISA) project 3D points to 2D pixel coordinates
    k = np.array(list(all_k[:2]) + list(all_k[-1:]))  # radial distortion coeffs (k1,k2,k3)
    p = all_k[2:4]                                    # tangential distortion (p1,p2)

    N = P.shape[0]
    X = R.dot(P.T - np.tile(t.reshape((-1, 1)), (1, len(P))))  # (3,N)

    XX = X[:2, :] / np.tile(X[2, :], (2, 1))  # normalized image plane

    r2 = XX[0, :]**2 + XX[1, :]**2
    radial = 1 + np.sum(
        np.tile(k.reshape((-1, 1)), (1, N)) * np.vstack((r2, r2**2, r2**3)),
        axis=0
    )
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + p[::-1].reshape((-1, 1)).dot(r2.reshape((1, -1)))
    proj = (np.tile(f, (N, 1)) * XXX.T) + np.tile(c, (N, 1))  # (N,2)
    return proj

# option 2: without radial distortion
def project_with_K(P_cam, K):
    # P_cam: (N,3) in camera coordinates
    # K: (3,3)
    P_h = (K @ P_cam.T).T              # (N,3)
    uv = P_h[:, :2] / P_h[:, 2:3]      # divide by w (=Z scaled)
    return uv                          # (N,2)

def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    # getting frame back to uint8 RGB for visualization (it was normalized for ResNet)
    f = frame_chw.astype(np.float32)

    # looks like ImageNet-normalized
    if f.min() < -0.5 or f.max() > 1.5:
        f = np.transpose(f, (1, 2, 0))             # HWC
        f = (f * IMAGENET_STD) + IMAGENET_MEAN
        f = np.clip(f, 0.0, 1.0)
        f = (f * 255.0).astype(np.uint8)
        return f

    # float [0..1] or uint8-like
    if f.max() <= 1.5:
        f = f * 255.0
    f = np.clip(f, 0, 255).astype(np.uint8)
    f = np.transpose(f, (1, 2, 0))  # HWC
    return f

def _as_numpy(x):
    # helper: torch -> numpy if needed
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def plot_batch_sample_2d_2dproj_3d(
    video, joints3d, joints2d, K,
    sample_idx=0, fps=10, point_size=18, line_width=2
):
    """
    video:   (B,T,3,H,W)
    joints3d:(B,T,J,3)
    joints2d:(B,T,J,2)   in pixel coords of the (cropped/resized) frames
    f,c,k,R,t: per-sample camera params (B, ...)
    """

    # --- convert to numpy for consistent matplotlib use ---
    video   = _as_numpy(video)
    joints3d = _as_numpy(joints3d)
    joints2d = _as_numpy(joints2d)
    K = _as_numpy(K)

    vid = video[sample_idx]      # (T,3,H,W)
    js3 = joints3d[sample_idx]   # (T,J,3)
    js2 = joints2d[sample_idx]   # (T,J,2)

    T = vid.shape[0]
    _, _, H, W = vid.shape

    # project each frame (more correct than projecting only once)
    proj2 = np.zeros_like(js2, dtype=np.float32)  # (T,J,2)
    for tt in range(T):
        proj2[tt] = project_with_K(js3[tt], K[sample_idx])
        # print(proj2[tt] - js2[tt])  # debug

    # frames to uint8 for display
    frames = [to_uint8_rgb(vid[tt]) for tt in range(T)]

    # ---- figure: 3 panels ----
    fig = plt.figure(figsize=(15, 4))
    ax_gt   = fig.add_subplot(1, 3, 1)
    ax_prj  = fig.add_subplot(1, 3, 2)
    ax_3d   = fig.add_subplot(1, 3, 3, projection="3d")

    ax_gt.set_title("Frame + GT 2D joints")
    ax_prj.set_title("Frame + Reprojected 2D joints")
    ax_3d.set_title("Skeleton 3D")

    # images
    im_gt  = ax_gt.imshow(frames[0])
    im_prj = ax_prj.imshow(frames[0])
    ax_gt.axis("off")
    ax_prj.axis("off")

    # ---- 2D scatters ----
    scat_gt  = ax_gt.scatter(js2[0, :, 0], js2[0, :, 1], s=point_size)
    scat_prj = ax_prj.scatter(proj2[0, :, 0], proj2[0, :, 1], s=point_size)

    # ---- 2D skeleton lines ----
    lines2d_gt = []
    lines2d_prj = []
    for a, b in H36M_EDGES:
        l_gt, = ax_gt.plot([js2[0, a, 0], js2[0, b, 0]],
                           [js2[0, a, 1], js2[0, b, 1]],
                           linewidth=line_width)
        l_pr, = ax_prj.plot([proj2[0, a, 0], proj2[0, b, 0]],
                            [proj2[0, a, 1], proj2[0, b, 1]],
                            linewidth=line_width)
        lines2d_gt.append(l_gt)
        lines2d_prj.append(l_pr)

    # Keep 2D axes consistent with image coords (origin at top-left)
    ax_gt.set_xlim(0, W - 1); ax_gt.set_ylim(H - 1, 0)
    ax_prj.set_xlim(0, W - 1); ax_prj.set_ylim(H - 1, 0)

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

    # ---- update function ----
    def update(tt):
        # images
        im_gt.set_data(frames[tt])
        im_prj.set_data(frames[tt])

        # 2D scatters (matplotlib wants Nx2)
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

        return [im_gt, im_prj, scat_gt, scat_prj, scat3d] + lines2d_gt + lines2d_prj + lines3d

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    plt.tight_layout()
    plt.show()
    return anim
