# precompute_resnet_features.py
import os
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import Human36MPreprocessedClips

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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 5, 6, 7, 8, 9, 11])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-fp16", action="store_true", help="Store feats as float16 to save disk space")
    args = parser.parse_args()

    # choose device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # helps performance for fixed input sizes (224x224)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"number gpus:{torch.cuda.device_count()}, device: {device}")

    # dataset yields: video, joints3d, joints2d, K
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
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    # ResNet50 like model.py => (B*T,2048,1,1) -> flatten
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # -----------------------------
    # Multi-GPU: DataParallel
    # -----------------------------
    use_dp = device.startswith("cuda") and (torch.cuda.device_count() > 1)
    if use_dp:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    backbone = backbone.to(device).eval()
    if use_dp:
        backbone = nn.DataParallel(backbone)

    # IMPORTANT: deterministic file naming per clip
    # DataLoader shuffles=False, so we can track global clip index.
    global_i = 0

    t_all = time.time()
    t_last = time.time()

    for it, batch in enumerate(loader):
        # batch: video: (B,T,3,224,224)
        video, joints3d, joints2d, K = batch
        B, T, C, H, W = video.shape

        # move video to GPU (non_blocking helps when pin_memory=True)
        video = video.to(device, non_blocking=True)

        # -----------------------------
        # ResNet forward (multi-GPU splits on dim 0 automatically)
        # -----------------------------
        # We flatten (B,T,3,224,224) -> (B*T,3,224,224) to run ResNet per frame.
        # DataParallel will split the (B*T) batch across GPUs.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.startswith("cuda"),
        ):
            x = video.view(B * T, C, H, W)          # (B*T,3,224,224)
            feats = backbone(x).flatten(1)          # (B*T,2048)
            feats = feats.view(B, T, -1)            # (B,T,2048)

        # choose dtype on disk
        feats_to_save = feats.half() if args.save_fp16 else feats.float()

        # save each clip in batch
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
                "feats": feats_to_save[b].cpu(),      # (T,2048)
                "joints3d": joints3d[b].cpu(),        # (T,J,3)
                "joints2d": joints2d[b].cpu(),        # (T,J,2)
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

        # light progress + timing
        if global_i % 500 == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(f"Saved {global_i}/{len(ds)} clips... (+{dt:.1f}s since last report)")

    print(f"Done. Saved {global_i} clips into: {out_root}")
    print(f"Total preprocessing time: {time.time() - t_all:.2f} seconds")


if __name__ == "__main__":
    main()