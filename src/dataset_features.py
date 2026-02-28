import os
import glob
from typing import List, Optional

import torch
from torch.utils.data import Dataset

"""
This module provides a Dataset class to load precomputed ResNet50 features
for Human3.6M video clips. The preprocess was made with preprocess_resnet_features.py.

Coded by Luísa Ferreira, 2026
"""
class Human36MFeatureClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        max_clips: Optional[int] = None,
        test_set: bool = False,
    ):
        self.root = root
        self.test_set = test_set   # <-- store flag

        pattern = os.path.join(root, "S*", "*", "cam_*", "clip_*.pt")
        files = sorted(glob.glob(pattern))
        print(f"Found {len(files)} cached clips under {pattern}")
        print(f"  Example file: {files[0] if len(files) > 0 else 'None'}")

        if subjects is not None:
            keep = []
            subj_set = set(subjects)
            for p in files:
                parts = p.split(os.sep)
                s_part = [x for x in parts if x.startswith("S")]
                if len(s_part) == 0:
                    continue
                s = int(s_part[0].replace("S", ""))
                if s in subj_set:
                    keep.append(p)
            files = keep

        if max_clips is not None:
            files = files[:max_clips]

        if len(files) == 0:
            raise RuntimeError(f"No cached clips found under {root}")

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        feats = d["feats"]
        joints3d = d["joints3d"]  
        joints2d = d["joints2d"]
        K = d["K"]
        print(joints3d.shape)
        root = joints3d[:, 0:1, :]          # (T, 1, 3)
        joints3d_norm = joints3d - root     # root-relative
        joints3d_norm = joints3d_norm / 1000.0  # mm -> meters

        if self.test_set:
            meta = d.get("meta", None)
            return feats, joints3d, joints2d, K, meta

        return feats, joints3d_norm, joints2d, K
