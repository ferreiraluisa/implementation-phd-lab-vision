import numpy as np
from visualize import plot_batch_sample
from visualize_2d import plot_batch_sample_2d_2dproj_3d
from visualize_features import plot_batch_sample_2d_2dproj_3d_no_video

"""
This script visualizes a batch of data previously saved in 'debug_batch.npz'.
Coded by Luisa Ferreira, 2026.
"""

data = np.load("src/debug_batch.npz", allow_pickle=True)

print(data.files)

video = data["video"]    # (B,T,3,224,224)
joints3d = data["joints3d"]  # (B,T,17,3)
joints2d = data["joints2d"]  # (B,T,17,2)
# options 1: get only intrinsic matrix K to calculate reprojection without distorsion.
K = data["cam_K"]        # (B,3,3)

# options 2: get all camera parameters to calculate reprojection with distorsion.
# f = data["cam_f"]        # (B,2)
# c = data["cam_c"]        # (B,2)
# k = data["cam_k"]        # (B,5)
# R = data["cam_rt"]        # (B,3,3)
# t = data["cam_t"]        # (B,3)
# K = data["cam_K"] # dict with intrinsics
# print("camera parameters:")
# print("f:", f.shape, f.dtype)
# print("c:", c.shape, c.dtype)
# print("k:", k.shape, k.dtype)
# print("R:", R.shape, R.dtype)
# print("t:", t.shape, t.dtype)
# print("K", K.shape, K.dtype)

plot_batch_sample_2d_2dproj_3d(video, joints3d, joints2d, K, sample_idx=1, fps=10)
# plot_batch_sample(video, joints, sample_idx=0, fps=25)
# for i in range(video.shape[0]):
#     plot_batch_sample(video, joints, sample_idx=i, fps=25)