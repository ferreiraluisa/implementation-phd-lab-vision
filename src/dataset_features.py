import os
import glob
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

""" This module provides a Dataset class to load precomputed ResNet50 features for Human3.6M video clips. The preprocess was made with preprocess_resnet_features.py.

Coded by Luísa Ferreira, 2026
"""
class Human36MFeatureClips(Dataset):
    def __init__(self, feat_root: str, subjects: Optional[List[int]] = None, max_clips: Optional[int] = None):
        self.feat_root = feat_root

        pattern = os.path.join(feat_root, "S*", "*", "cam_*", "clip_*.pt")
        files = sorted(glob.glob(pattern))
        

        if subjects is not None:
            keep = []
            subj_set = set(subjects)
            for p in files:
                # path contains .../S{n}/...
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
            raise RuntimeError(f"No cached clips found under {feat_root}")

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx], map_location="cpu")
        return d["feats"], d["joints3d"], d["joints2d"], d["K"]
