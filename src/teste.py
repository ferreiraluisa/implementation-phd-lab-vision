import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_features import Human36MFeatureClips

"""
This script tests the Human36MFeatureClips dataset by loading one batch
and saving it to a .npz file for debugging purposes.

It includes:
- A custom collate_fn to avoid crashing when meta contains None, dicts, or strings.
- Proper tensor-to-numpy conversion.
- Safety checks to detect invalid samples early.

Coded by Luisa Ferreira, 2026.
"""

root = "/home/s26ldeso/Human3.6M_preprocessed_resnet_features"

train_ds = Human36MFeatureClips(
    root,
    subjects=[9],
    test_set=True,
)


def collate_keep_meta(batch):
    """
    Custom collate function:
    - Stacks only tensor fields (video, joints3d, joints2d, K)
    - Keeps meta as a list (no stacking), so it can contain dicts/strings/None safely
    """

    videos, j3ds, j2ds, Ks, metas = zip(*batch)

    # Check if any tensor field is None (this indicates a dataset bug)
    for i, (v, j3d, j2d, k) in enumerate(zip(videos, j3ds, j2ds, Ks)):
        if v is None or j3d is None or j2d is None or k is None:
            raise ValueError(
                f"[Dataset Error] Sample {i} in batch contains None in "
                f"(video/joints3d/joints2d/K). Fix dataset filtering."
            )

    video = torch.stack(videos, dim=0)
    joints3d = torch.stack(j3ds, dim=0)
    joints2d = torch.stack(j2ds, dim=0)
    K = torch.stack(Ks, dim=0)

    meta = list(metas)  # keep meta as list (no tensor stacking)
    return video, joints3d, joints2d, K, meta


train_dl = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_keep_meta,
)

# Load one batch
video, joints3d, joints2d, K, meta = next(iter(train_dl))

print("video:", video.shape, video.dtype)
print("joints3d:", joints3d.shape, joints3d.dtype)
print("joints2d:", joints2d.shape, joints2d.dtype)
print("K:", K.shape, K.dtype)
print("meta type:", type(meta), "length:", len(meta))
print("meta[0] type:", type(meta[0]))

# Convert tensors to numpy
video_np = video.detach().cpu().numpy()
joints3d_np = joints3d.detach().cpu().numpy()
joints2d_np = joints2d.detach().cpu().numpy()
K_np = K.detach().cpu().numpy()

# Save meta as object array (since it may contain dicts, strings, etc.)
meta_np = np.array(meta, dtype=object)

# Save everything to compressed NPZ file
np.savez_compressed(
    "debug_batch.npz",
    video=video_np,
    joints3d=joints3d_np,
    joints2d=joints2d_np,
    cam_K=K_np,
    meta=meta_np,
)

print("Saved to debug_batch.npz")
