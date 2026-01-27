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


# store clip data, for each video of Human3.6M, create clips of fixed length
@dataclass
class ClipIndex:
    video_path: str
    gt_path: str
    subject: str
    action: str
    cam: str
    start: int
    end: int  # exclusive


def _load_gt(gt_path: str) -> torch.Tensor:
    with open(gt_path, "rb") as f:
        data = pickle.load(f)

    arr = np.asarray(data["3d"])
    # arr shape = (T,17,3)

    return torch.from_numpy(arr).float()

# -----------------------------
# pytorch dataset that returns fixed-length clips from Human3.6M preprocessed
# video:  (T,3,224,224) normalized for ImageNet/ResNet50
# joints: (T,17,3)
# T: number of frames in the clip
# -----------------------------
class Human36MPreprocessedClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: List[int],
        seq_len: int = 40,
        stride: int = 10,
        frame_skip: int = 2,
        cams: Optional[List[int]] = None,
        resize: int = 224, # imageNet size
        center_crop: bool = True,
        max_clips: Optional[int] = None,
    ):
        super().__init__()

        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.root = root
        self.subjects = subjects
        self.seq_len = seq_len
        self.stride = stride
        self.frame_skip = frame_skip

        tx = [T.Resize(resize)] # torchvision.transforms.v2
        # video tensor from read_video is (T,C,H,W) to match (T,3,224,224)
        if center_crop:
            tx.append(T.CenterCrop(resize))
        tx += [
            T.ToDtype(torch.float32, scale=True),  # uint8 -> float [0,1]
            T.Normalize(mean=(0.485, 0.456, 0.406), # normalize for ImageNet/ResNet50
                        std=(0.229, 0.224, 0.225)),
        ]
        self.frame_tf = T.Compose(tx)

        self.index: List[ClipIndex] = []

        for s in subjects:
            subj_dir = os.path.join(root, f"S{s}")
            # print(subj_dir)
            actions = sorted(
                a for a in os.listdir(subj_dir)
                if os.path.isdir(os.path.join(subj_dir, a))
            )

            for action in actions:
                action_dir = os.path.join(subj_dir, action)
                cam_dirs = sorted(glob.glob(os.path.join(action_dir, "cam_*")))
                # print(cam_dirs)

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

                    joints_all = _load_gt(gt_path)
                    n_frames = int(joints_all.shape[0]) # total frames from gt

                    # IMPORTANT: start/end are in the SUBSAMPLED timeline (same as video after frame_skip)
                    n_frames_sub = (n_frames + self.frame_skip - 1) // self.frame_skip

                    for start in range(0, n_frames_sub - seq_len + 1, stride):
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
                        # avoid building too large dataset
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

        # start/end are in subsampled coords, so subsample first, then slice
        frames = frames[::self.frame_skip]  # subsample
        frames = frames[start:end]

        if frames.shape[0] != (end - start):
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

        joints_all = _load_gt(ci.gt_path)  # (N,17,3)

        # Map subsampled indices back to original indices to guarantee alignment
        # (video is frames[::frame_skip][start:end] => original indices are (start..end-1)*frame_skip)
        orig_idx = torch.arange(ci.start, ci.end, dtype=torch.long) * self.frame_skip

        if int(orig_idx[-1]) >= joints_all.shape[0]:
            raise RuntimeError(
                f"Joint index out of range for {ci.gt_path}: "
                f"max orig_idx={int(orig_idx[-1])}, n_frames={joints_all.shape[0]}"
            )

        joints = joints_all[orig_idx]  # (T,17,3)

        # assert video and joints has are representing the same frames
        assert video.shape[0] == joints.shape[0], (
            f"Mismatch T: video {video.shape[0]} vs joints {joints.shape[0]}"
        )

        return video, joints
