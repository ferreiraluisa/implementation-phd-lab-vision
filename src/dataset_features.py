import os
import glob
from random import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset

"""
This module provides a Dataset class to load precomputed ResNet50 features
for Human3.6M video clips. The preprocess was made with preprocess_resnet_features.py.

Coded by Luísa Ferreira, 2026
"""
import os
import glob
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset

AUGS = ["orig", "hflip", "cjitter", "trev"]
AUG_SUFFIXES = tuple(f"_{a}.pt" for a in AUGS)

class Human36MFeatureClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        max_clips: Optional[int] = None,
        test_set: bool = False,
    ):
        self.root = root
        self.test_set = test_set

        # default subjects (train)
        if subjects is None:
            subjects = [1, 6, 7, 8]
        self.subjects = subjects

        # augment only for the train subjects exactly
        self.augment = (set(subjects) == {1, 6, 7, 8}) and (not test_set)

        if self.augment:
            pattern = os.path.join(root, "S*", "*", "cam_*", "clip_*_orig.pt")
            files = sorted(glob.glob(pattern))
        else:
            files = []
            for s in subjects:
                pattern = os.path.join(root, f"S{s}", "*", "cam_*", "clip_*.pt")
                files.extend(glob.glob(pattern))
            files = sorted(files)

            # safety: exclude augmented suffixes if they ever appear here
            files = [p for p in files if not p.endswith(AUG_SUFFIXES)]

        subj_set = set(subjects)
        keep = []
        for p in files:
            parts = p.split(os.sep)
            s_part = next((x for x in parts if x.startswith("S")), None)
            if s_part is None:
                continue
            s = int(s_part.replace("S", ""))
            if s in subj_set:
                keep.append(p)
        files = keep

        if max_clips is not None:
            files = files[:max_clips]

        if len(files) == 0:
            raise RuntimeError(f"No cached clips found under {root} for subjects={subjects}")

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        base = self.files[idx]

        if self.augment:
            # base is ..._orig.pt
            aug = random.choice(AUGS)
            path = base.replace("_orig.pt", f"_{aug}.pt")
            if not os.path.isfile(path):
                path = base
        else:
            path = base

        d = torch.load(path, map_location="cpu", weights_only=True)

        feats = d["feats"]
        joints3d = d["joints3d"] / 1000.0  # mm → m
        joints2d = d["joints2d"]
        K = d["K"]

        if self.test_set:
            meta = d.get("meta", None)
            return feats, joints3d, joints2d, K, meta

        return feats, joints3d, joints2d, K
