import os
import glob
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2 as T


@dataclass
class ClipIndex:
    video_path: str
    gt_path: str
    subject: str
    action: str
    cam: str
    start: int
    end: int  # exclusive


def _subject_id_from_name(subject_folder: str) -> int:
    return int(subject_folder.replace("S", "").strip())


def _load_gt_3d_as_T17x3(gt_path: str) -> torch.Tensor:
    """
    Loads gt_poses.pkl and returns joints as float32 tensor shaped (T, 17, 3).
    Assumes pickle dict contains key '3d'.
    """
    with open(gt_path, "rb") as f:
        data = pickle.load(f)

    arr = np.asarray(data["3d"])

    # Make it (T,17,3)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for data['3d'], got shape {arr.shape}")

    if arr.shape[-2:] == (17, 3):
        # (T,17,3)
        pass
    elif arr.shape[:2] == (17, 3):
        # (17,3,T) -> (T,17,3)
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[:2] == (3, 17):
        # (3,17,T) -> (T,17,3)
        arr = np.transpose(arr, (2, 1, 0))
    elif arr.shape[-2:] == (3, 17):
        # (T,3,17) -> (T,17,3)
        arr = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(
            f"Don't know how to interpret data['3d'] shape {arr.shape}. "
            "Expected (T,17,3) or (17,3,T) (or common variants)."
        )

    return torch.from_numpy(arr).float()


class Human36MPreprocessedClips(Dataset):
    """
    Dataset returning:
      video:  (T,3,224,224) normalized for ImageNet/ResNet50
      joints: (T,17,3)
    """
    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int = 40,
        stride: int = 1,
        cams: Optional[List[int]] = None,   # e.g. [0,1,2,3]
        resize: int = 224,
        center_crop: bool = True,
        return_meta: bool = True,
        max_clips: Optional[int] = None,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}

        self.root = root
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.return_meta = return_meta

        split_subjects = {
            "train": {1, 6, 7, 8},
            "val": {5},
            "test": {9, 11},
        }[split]

        tx = [T.Resize(resize)]
        if center_crop:
            tx.append(T.CenterCrop(resize))
        tx += [
            T.ToDtype(torch.float32, scale=True),  # uint8 -> float [0,1]
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ]
        self.frame_tf = T.Compose(tx)

        self.index: List[ClipIndex] = []

        subjects = sorted(
            d for d in os.listdir(root)
            if d.startswith("S") and os.path.isdir(os.path.join(root, d))
        )

        for s in subjects:
            sid = _subject_id_from_name(s)
            if sid not in split_subjects:
                continue

            subj_dir = os.path.join(root, s)
            actions = sorted(
                a for a in os.listdir(subj_dir)
                if os.path.isdir(os.path.join(subj_dir, a))
            )

            for action in actions:
                action_dir = os.path.join(subj_dir, action)
                cam_dirs = sorted(glob.glob(os.path.join(action_dir, "cam_*")))

                for cam_dir in cam_dirs:
                    cam_name = os.path.basename(cam_dir)
                    cam_id = int(cam_name.replace("cam_", ""))
                    if cams is not None and cam_id not in cams:
                        continue

                    mp4s = glob.glob(os.path.join(cam_dir, "*.mp4"))
                    gt_path = os.path.join(cam_dir, "gt_poses.pkl")
                    if not mp4s or not os.path.isfile(gt_path):
                        continue

                    video_path = mp4s[0]

                    # Use GT length as number of frames (assumes aligned)
                    joints_all = _load_gt_3d_as_T17x3(gt_path)
                    n_frames = int(joints_all.shape[0])

                    for start in range(0, n_frames - seq_len + 1, stride):
                        self.index.append(
                            ClipIndex(
                                video_path=video_path,
                                gt_path=gt_path,
                                subject=s,
                                action=action,
                                cam=cam_name,
                                start=start,
                                end=start + seq_len,
                            )
                        )
                        if max_clips is not None and len(self.index) >= max_clips:
                            break
                    if max_clips is not None and len(self.index) >= max_clips:
                        break
                if max_clips is not None and len(self.index) >= max_clips:
                    break
            if max_clips is not None and len(self.index) >= max_clips:
                break

        if len(self.index) == 0:
            raise RuntimeError(f"No clips found under root={root}. Check your folder structure and files.")

    def __len__(self) -> int:
        return len(self.index)

    def _read_video_clip(self, video_path: str, start: int, end: int) -> torch.Tensor:
        # frames: (Tv,H,W,C) uint8
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        frames = frames[start:end]
        if frames.shape[0] != (end - start):
            # Not fatal, but highlights mismatch/decoding issues
            raise RuntimeError(
                f"Frame count mismatch reading {video_path}: "
                f"got {frames.shape[0]}, expected {end-start} for slice [{start}:{end}]."
            )
        frames = frames.permute(0, 3, 1, 2)  # (T,C,H,W)
        frames = self.frame_tf(frames)       # batched transform
        return frames

    def __getitem__(self, idx: int):
        ci = self.index[idx]

        video = self._read_video_clip(ci.video_path, ci.start, ci.end)  # (T,3,224,224)
        joints_all = _load_gt_3d_as_T17x3(ci.gt_path)                   # (N,17,3)
        joints = joints_all[ci.start:ci.end]                             # (T,17,3)

        if self.return_meta:
            meta = {
                "subject": ci.subject,
                "action": ci.action,
                "cam": ci.cam,
                "video_path": ci.video_path,
                "gt_path": ci.gt_path,
                "start": ci.start,
                "end": ci.end,
            }
            return video, joints, meta

        return video, joints
