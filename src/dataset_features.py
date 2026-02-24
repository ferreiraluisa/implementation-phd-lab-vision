import os
import glob
from typing import List, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset

"""
This module provides a Dataset class to load precomputed ResNet50 features
for Human3.6M video clips. The preprocess was made with preprocess_resnet_features.py.

Coded by Luísa Ferreira, 2026
"""

""""
SHARD LAYOUT (one file: shard_XXXXX.pt)
─────────────────────────────────────────
{
    "feats":    (N_rows, T, 2048),   # N_rows = n_clips_in_shard × n_vars
    "joints3d": (N_rows, T, 17, 3),
    "joints2d": (N_rows, T, 17, 2),
    "K":        (N_rows, 3, 3),
    "meta":     [dict × N_rows],
    "n_vars":   int,
}
"""
class Human36MFeatureClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        max_clips: Optional[int] = None,
        test_set: bool = False,
        augment: bool = False, 
        shard_cache_size: int = 2,
    ):
        self.root = Path(root)
        self.test_set = test_set  
        self.augment = augment  
        self._cache_sz = shard_cache_size

        index_path = self.root / "index.pt"
        if not index_path.exists():
            raise RuntimeError(
                f"index.pt not found in {root}. "
                "Run preprocess_resnet_features.py first."
            )
        idx_data = torch.load(index_path, map_location="cpu", weights_only=True)

        self._n_vars   = idx_data["n_variants"]
        self._aug_names = idx_data.get("aug_names", ["orig"])
        all_clips      = idx_data["clips"]   # list of dicts

        # ── Filter by subject ─────────────────────────────────────────────────
        if subjects is not None:
            subj_set  = set(subjects)
            all_clips = [c for c in all_clips if c["subject"] in subj_set]

        if max_clips is not None:
            all_clips = all_clips[:max_clips]

        if len(all_clips) == 0:
            raise RuntimeError(f"No clips found in {root} for subjects={subjects}.")

        self._clips = all_clips   # list of clip-level index dicts

        # ── Build flat item list ───────────────────────────────────────────────
        # Each item is (clip_dict, variant_offset) where variant_offset selects
        # which row inside the shard to read.
        # augment=False → only variant 0 ("orig")
        # augment=True  → one item per variant per clip
        if self.augment:
            self._items = [
                (clip, v) for clip in self._clips for v in range(self._n_vars)
            ]
        else:
            self._items = [(clip, 0) for clip in self._clips]

        # ── Shard cache: {shard_id: shard_dict} ───────────────────────────────
        self._shard_cache: dict = {}
        self._cache_order: list = []   # LRU order (oldest first)


    def __len__(self) -> int:
        return len(self._items)

    def _load_shard(self, shard_id: int) -> dict:
        if shard_id in self._shard_cache:
            # Move to most-recently-used position
            self._cache_order.remove(shard_id)
            self._cache_order.append(shard_id)
            return self._shard_cache[shard_id]

        # Evict oldest shard if cache is full
        if len(self._cache_order) >= self._cache_sz:
            oldest = self._cache_order.pop(0)
            del self._shard_cache[oldest]

        path = self.root / f"shard_{shard_id:05d}.pt"
        shard = torch.load(path, map_location="cpu", weights_only=True)
        self._shard_cache[shard_id] = shard
        self._cache_order.append(shard_id)
        return shard

    def __getitem__(self, idx: int):
        clip, var_offset = self._items[idx]

        shard  = self._load_shard(clip["shard_id"])
        row    = clip["row"] + var_offset     # exact row in the shard tensors

        feats    = shard["feats"][row]            # (T, 2048)
        joints3d = shard["joints3d"][row] / 1000.0  # mm → m, (T, 17, 3)
        joints2d = shard["joints2d"][row]         # (T, 17, 2)
        K        = shard["K"][row]               # (3, 3)

        if self.test_set:
            meta = shard["meta"][row]
            return feats, joints3d, joints2d, K, meta

        return feats, joints3d, joints2d, K