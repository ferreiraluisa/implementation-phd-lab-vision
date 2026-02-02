import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Human36MPreprocessedClips

"""
This script tests the Human36MPreprocessedClips dataset by loading a batch of data
and saving it to a .npz file for debugging purposes.
Coded by Luisa Ferreira, 2026.
"""

root = "../../Human3.6M_preprocessed"

train_ds = Human36MPreprocessedClips(
    root,
    subjects=[1],
    seq_len=40,
    stride=10,
    frame_skip=2
)

train_dl = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=False 
)

# option 2
# video, joints3d, joints2d, camera_params = next(iter(train_dl))

# option 1
video, joints3d, joints2d, K = next(iter(train_dl))

print("video:", video.shape, video.dtype)
print("joints3d:", joints3d.shape, joints3d.dtype)
print("joints2d:", joints2d.shape, joints2d.dtype)
print("K:", K.shape, K.dtype)

video_np = video.detach().cpu().numpy()
joints3d_np = joints3d.detach().cpu().numpy()
joints2d_np = joints2d.detach().cpu().numpy()
K = K.detach().cpu().numpy()

# option 2: save all camera parameters to calculate reprojection with distortion
# cam_np = {}
# for k, v in camera_params.items():
#     if torch.is_tensor(v):
#         cam_np[k] = v.detach().cpu().numpy()
#     else:
#         cam_np[k] = np.asarray(v)

np.savez_compressed(
    "debug_batch.npz",
    video=video_np,
    joints3d=joints3d_np,
    joints2d=joints2d_np,
    cam_K=K,
    # **{f"cam_{k}": cam_np[k] for k in cam_np}
)

print("Saved to debug_batch.npz")
