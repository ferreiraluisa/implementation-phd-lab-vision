import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Human36MPreprocessedClips

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
    pin_memory=False  # no MPS não precisa
)

video, joints = next(iter(train_dl))

print("video:", video.shape, video.dtype)
print("joints:", joints.shape, joints.dtype)

# garante CPU
video_np = video.detach().cpu().numpy()
joints_np = joints.detach().cpu().numpy()

# salva
np.savez_compressed(
    "debug_batch.npz",
    video=video_np,
    joints=joints_np
)

print("Saved to debug_batch.npz")
