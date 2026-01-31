import os
import glob
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as F


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

def _load_poses(gt_path):
    with open(gt_path, "rb") as f:
        data = pickle.load(f)

    j3d = torch.from_numpy(np.asarray(data["3d"])).float()
    j2d = torch.from_numpy(np.asarray(data["2d"])).float()
    return j3d, j2d


def _compute_square_crop_from_2d(joints2d, img_h, img_w, scale=1.6):
    # choose best crop to center on the subject based on the 2D joints across all frames of the sequence
    pts = joints2d.reshape(-1, 2)  # (T*J,2)

    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    w = (x_max - x_min).clamp(min=1.0)
    h = (y_max - y_min).clamp(min=1.0)

    side = scale * torch.max(w, h)

    left = cx - 0.5 * side
    top = cy - 0.5 * side

    left = left.clamp(0, img_w - side)
    top = top.clamp(0, img_h - side)

    left_i = int(left.round().item())
    top_i = int(top.round().item())
    side_i = int(side.round().item())

    side_i = max(1, min(side_i, img_w - left_i, img_h - top_i))
    return torch.tensor([top_i, left_i, side_i, side_i], dtype=torch.int64)


def _crop_and_resize_video_uint8(frames_uint8, box, out_size=224):
    top, left, hh, ww = box.tolist()

    # (T,C,H,W)
    frames = frames_uint8.permute(0, 3, 1, 2)
    frames = frames[:, :, top:top + hh, left:left + ww]  # (T,C,hh,ww)

    frames = F.resize(frames, [out_size, out_size])

    # uint8 -> float [0,1]
    frames = frames.to(torch.float32) / 255.0
    return frames


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
        resize: int = 224,        # ImageNet size
        crop_scale: float = 1.6,  # margin around subject bbox from 2D joints
        max_clips: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.subjects = subjects
        self.seq_len = seq_len
        self.stride = stride
        self.frame_skip = frame_skip
        self.resize = resize
        self.crop_scale = crop_scale

        # normalization for ImageNet/ResNet
        self.frame_tf = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        self.index: List[ClipIndex] = []

        for s in subjects:
            subj_dir = os.path.join(root, f"S{s}")
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

                    joints3d_all, _ = _load_poses(gt_path)
                    n_frames = int(joints3d_all.shape[0])

                    # start/end are in the SUBSAMPLED timeline (same as video after frame_skip)
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

    def _read_video_uint8_clip(self, video_path, start, end):
        # frames: (Tv,H,W,C) uint8
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")

        # start/end are in subsampled coords, so subsample first, then slice
        frames = frames[::self.frame_skip]
        frames = frames[start:end]

        if frames.shape[0] != (end - start):
            raise RuntimeError(
                f"Frame count mismatch reading {video_path}: "
                f"got {frames.shape[0]}, expected {end-start} for slice [{start}:{end}]."
            )
        return frames  # (T,H,W,C) uint8

    def __getitem__(self, idx):
        ci = self.index[idx]

        # read frames (uint8)
        frames_uint8 = self._read_video_uint8_clip(ci.video_path, ci.start, ci.end)  # (T,H,W,C)
        Tt, H, W, C = frames_uint8.shape
        assert C == 3

        # load joints
        joints3d_all, joints2d_all = _load_poses(ci.gt_path)  # (N,17,3), (N,17,2)

        # map clip frame indices to original frame indices
        orig_idx = torch.arange(ci.start, ci.end, dtype=torch.long) * self.frame_skip

        if int(orig_idx[-1]) >= joints3d_all.shape[0]:
            raise RuntimeError(
                f"Joint index out of range for {ci.gt_path}: "
                f"max orig_idx={int(orig_idx[-1])}, n_frames={joints3d_all.shape[0]}"
            )

        joints3d = joints3d_all[orig_idx]  # (T,17,3)
        joints2d = joints2d_all[orig_idx]  # (T,17,2)
        assert frames_uint8.shape[0] == joints3d.shape[0], (
            f"Mismatch T: video {frames_uint8.shape[0]} vs joints {joints3d.shape[0]}"
        )

        box = _compute_square_crop_from_2d(
            joints2d=joints2d,
            img_h=H,
            img_w=W,
            scale=self.crop_scale,
        )
        video = _crop_and_resize_video_uint8(frames_uint8, box, out_size=self.resize)
        # normalize frames for ImageNet/ResNet
        video = self.frame_tf(video)

        return video, joints3d
