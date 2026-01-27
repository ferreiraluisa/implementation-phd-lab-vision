import numpy as np
from visualize import plot_batch_sample
import torch

data = np.load("debug_batch.npz")

print(data.files)

video = data["video"]    # (B,T,3,224,224)
joints = data["joints"]  # (B,T,17,3)


for i in range(video.shape[0]):
    plot_batch_sample(video, joints, sample_idx=i, fps=25)