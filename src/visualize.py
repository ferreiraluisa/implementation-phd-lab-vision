import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
This code provides visualization functions for 3D human pose data. Only video sequence + 3D plot
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

def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    # getting frame back to uint8 RGB for visualization(it was normalized for ResNet)
    f = frame_chw.astype(np.float32)

    if f.min() < -0.5 or f.max() > 1.5:
        f = np.transpose(f, (1, 2, 0))             
        f = (f * IMAGENET_STD) + IMAGENET_MEAN     
        f = np.clip(f, 0.0, 1.0)
        f = (f * 255.0).astype(np.uint8)
        return f

    if f.max() <= 1.5:
        f = f * 255.0
    f = np.clip(f, 0, 255).astype(np.uint8)
    f = np.transpose(f, (1, 2, 0))  # HWC
    return f

def plot_batch_sample(video, joints, sample_idx=0, fps=10):
    vid = video[sample_idx]
    js = joints[sample_idx]

    T = vid.shape[0]

    frames = []
    for t in range(T):
        frame = to_uint8_rgb(vid[t])
        frames.append(frame)

    xs, ys, zs = js[...,0], js[...,1], js[...,2]

    # define limits of plot with padding so the joints are not to tight to borders
    def pad(a,b,p=0.05):
        r = b-a if b>a else 1.0
        return a-p*r, b+p*r

    xlim = pad(xs.min(), xs.max())
    ylim = pad(ys.min(), ys.max())
    zlim = pad(zs.min(), zs.max())

    fig = plt.figure(figsize=(10,4))
    ax_img = fig.add_subplot(1,2,1)
    ax_3d  = fig.add_subplot(1,2,2, projection="3d")

    im = ax_img.imshow(frames[0])
    ax_img.axis("off")
    ax_img.set_title("Video")

    ax_3d.set_xlim(*xlim)
    ax_3d.set_ylim(*ylim)
    ax_3d.set_zlim(*zlim)
    ax_3d.set_title("Skeleton 3D")

    ax_3d.view_init(elev=45, azim=-90) # just change the view angle

    scat = ax_3d.scatter(js[0,:,0], js[0,:,1], js[0,:,2], s=20)

    lines = []
    for a,b in H36M_EDGES:
        line, = ax_3d.plot(
            [js[0,a,0], js[0,b,0]],
            [js[0,a,1], js[0,b,1]],
            [js[0,a,2], js[0,b,2]],
        )
        lines.append(line)

    def update(t):
        im.set_data(frames[t])
        scat._offsets3d = (js[t,:,0], js[t,:,1], js[t,:,2])

        for line, (a,b) in zip(lines, H36M_EDGES):
            line.set_data(
                [js[t,a,0], js[t,b,0]],
                [js[t,a,1], js[t,b,1]]
            )
            line.set_3d_properties(
                [js[t,a,2], js[t,b,2]]
            )
        return [im, scat] + lines

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=False)
    plt.tight_layout()
    plt.show()

    return anim
