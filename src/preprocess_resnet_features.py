# precompute_resnet_features.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import Human36MPreprocessedClips
import time 

"""
This script precomputes ResNet50 features for Human3.6M video clips and saves them to disk.
Each clip's features, along with 2D/3D joints and camera intrinsics, are stored in a structured directory format.
This allows for efficient loading during model training or evaluation. Avoiding redundant computation of ResNet features speeds up experiments significantly.

Coded by Luísa Ferreira, 2026
"""

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Precompute per-clip ResNet50 features (2048D) for Human3.6M clips")
    parser.add_argument("--root", type=str, required=True, help="H36M preprocessed root (same as train.py uses)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for cached features")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 5, 6, 7, 8, 9, 11])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ds = Human36MPreprocessedClips(
        root=args.root,
        subjects=args.subjects,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        max_clips=None,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ResNet50 like model.py => (B*T,2048,1,1) -> flatten
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

    # IMPORTANT: deterministic file naming per clip
    # DataLoader shuffles=False, so we can track global clip index.
    global_i = 0

    for batch in loader:
        video, joints3d, joints2d, K = batch  # video: (B,T,3,224,224)
        B, T, C, H, W = video.shape

        video = video.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            x = video.view(B * T, C, H, W)
            feats = backbone(x).flatten(1)          # (B*T,2048)
            feats = feats.view(B, T, -1).float()    # store float32 on disk for safety

        for b in range(B):
            clip = ds.index[global_i] 
            global_i += 1

            # Create a stable folder structure
            # out/S1/ActionName/cam_0/clip_start_end_fs2_len40.pt
            rel_dir = Path(f"S{clip.subject}") / clip.action / f"cam_{clip.cam}"
            save_dir = out_root / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"clip_{clip.start}_{clip.end}_fs{args.frame_skip}_len{args.seq_len}.pt"

            payload = {
                "feats": feats[b].cpu(),        # (T,2048)
                "joints3d": joints3d[b].cpu(),  # (T,J,3)
                "joints2d": joints2d[b].cpu(),  # (T,J,2)
                "K": K[b].cpu() if K.ndim >= 3 else K.cpu(),  # (3,3)
                "meta": {
                    "subject": clip.subject,
                    "action": clip.action,
                    "cam": clip.cam,
                    "start": clip.start,
                    "end": clip.end,
                    "seq_len": args.seq_len,
                    "frame_skip": args.frame_skip,
                }
            }
            torch.save(payload, save_path)

        if global_i % 500 == 0:
            print(f"Saved {global_i}/{len(ds)} clips...")

    print(f"Done. Saved {global_i} clips into: {out_root}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"Total preprocessing time: {end_time - start_time:.2f} seconds")
