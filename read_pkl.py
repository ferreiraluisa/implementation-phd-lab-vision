import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# CONFIG
# ============================================================
GT_PATH = 'Human3.6M_preprocessed/S1/Directions_0/cam_0/gt_poses.pkl'
VIDEO_PATH = 'Human3.6M_preprocessed/S1/Directions_0/cam_0/S1_Directions_0_cam_0.mp4'
SKIP_FRAMES = 2      # must match pose sampling
FPS_INTERVAL = 40   # ms between frames

# ============================================================
# LOAD VIDEO FRAMES
# ============================================================
def load_video_frames(video_path, skip=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        idx += 1

    cap.release()
    return frames

# ============================================================
# LOAD POSES
# ============================================================
with open(GT_PATH, 'rb') as f:
    data = pickle.load(f)

poses3d = np.asarray(data['3d'])   # (T,17,3) or (17,3,T)
print(poses3d.shape)

print("Loaded poses3d shape:", poses3d.shape)

# Ensure shape = (17, 3, T)
if poses3d.shape[0] == 17:
    poses = poses3d
elif poses3d.shape[-1] == 3:
    poses = poses3d.transpose(1, 2, 0)
else:
    raise ValueError("Unexpected poses3d shape")

T = poses.shape[2]

# Center on pelvis
poses = poses - poses[0:1, :, :]

# ============================================================
# LOAD RGB FRAMES
# ============================================================
imgs = load_video_frames(VIDEO_PATH, skip=SKIP_FRAMES)

num_frames = min(T, len(imgs))
print(T, len(imgs)) 
print(f"Animating {num_frames} frames")

# ============================================================
# HUMAN3.6M SKELETON (17 joints)
# ============================================================
H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine + head
    (8, 11), (11, 12), (12, 13),   # Left arm
    (8, 14), (14, 15), (15, 16)    # Right arm
]

# ============================================================
# SETUP FIGURE
# ============================================================
fig = plt.figure(figsize=(12, 6))

ax_img = fig.add_subplot(121)
ax_3d = fig.add_subplot(122, projection='3d')

# --- RGB image ---
img_artist = ax_img.imshow(imgs[0])
ax_img.axis('off')
ax_img.set_title("RGB Frame")

# --- 3D pose ---
radius = np.max(np.abs(poses))
ax_3d.set_xlim([-radius, radius])
ax_3d.set_ylim([-radius, radius])
ax_3d.set_zlim([-radius, radius])

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

ax_3d.view_init(elev=15, azim=70)
ax_3d.invert_zaxis()
ax_3d.set_title("3D Pose (Human3.6M)")

# Initialize skeleton lines
lines = []
for _ in H36M_EDGES:
    line, = ax_3d.plot([], [], [], lw=2)
    lines.append(line)

# ============================================================
# ANIMATION UPDATE
# ============================================================
def update(frame):
    # Update RGB image
    img_artist.set_data(imgs[frame])

    # Update skeleton
    xyz = poses[:, :, frame]

    for line, (i, j) in zip(lines, H36M_EDGES):
        line.set_data(
            [xyz[i, 0], xyz[j, 0]],
            [xyz[i, 1], xyz[j, 1]]
        )
        line.set_3d_properties(
            [xyz[i, 2], xyz[j, 2]]
        )

    ax_3d.set_title(f"3D Pose – Frame {frame}")
    return [img_artist] + lines

# ============================================================
# RUN ANIMATION
# ============================================================
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=FPS_INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()
