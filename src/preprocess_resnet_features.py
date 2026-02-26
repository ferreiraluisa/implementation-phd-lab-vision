import os
import argparse
from pathlib import Path
import time
from queue import Queue
from threading import Thread

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

Used Claude to help optimize and refactor the code for better performance and readability.
(older version was taking too long(5 hours to process only 6k/200k clips), this version processes all ~20k clips in ~30 minutes on a single GPU).
"""

AUG_NAMES = ["", "hflip", "temp_rev"]  # must match the order of variants produced in dataset.py when augment=True

class AsyncFileWriter:
    # asynchronous file writer using a background thread and a queue, to speed up saving features to disk without blocking the main processing loop
    # this helped a lot!!!!!!!!
    def __init__(self, max_queue_size=100):
        self.queue = Queue(maxsize=max_queue_size)
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.count = 0
    
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:  
                break
            payload, save_path = item
            torch.save(payload, save_path, _use_new_zipfile_serialization=False)
            self.queue.task_done()
    
    def save(self, payload, save_path):
        self.queue.put((payload, save_path))
        self.count += 1
    
    def wait(self):
        self.queue.join()
    
    def stop(self):
        self.queue.put(None)
        self.thread.join()

def _meta_at(meta_batch, b: int):
    # meta_batch is a dict produced by default_collate: {key: list/tensor of length B}
    out = {}
    for k, v in meta_batch.items():
        if torch.is_tensor(v):
            # tensor with shape (B, ...) -> select b
            vb = v[b]
            out[k] = vb.item() if vb.numel() == 1 else vb
        else:
            # list/tuple -> select b
            out[k] = v[b]
    return out

def augment_collate_fn(batch):
    # when augment = True, each sample in the batch is a list of augmented variants of the same clip, so we need to collate them separately and return a list of collated batches, one for each variant.
    n_variants = len(batch[0])
    collated = []
    for v in range(n_variants):
        videos   = torch.stack([sample[v][0] for sample in batch])
        joints3d = torch.stack([sample[v][1] for sample in batch])
        joints2d = torch.stack([sample[v][2] for sample in batch])
        K        = torch.stack([sample[v][3] for sample in batch])
        collated.append((videos, joints3d, joints2d, K))
    return collated


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("OPTIMIZED: Precompute per-clip ResNet50 features for H36M")
    parser.add_argument("--root", type=str, required=True, help="H36M preprocessed root")
    parser.add_argument("--out", type=str, required=True, help="Output directory for cached features")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 5, 6, 7, 8, 9, 11])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-fp16", action="store_true", help="Store feats as float16")
    parser.add_argument("--augment", action="store_true", help="enable online data augmentation")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # to help performance, enable cudnn benchmark and allow TF32 on Ampere+ GPUs, which can speed up ResNet inference significantly with minimal impact on feature quality. Also set pin_memory=True in DataLoader for faster host to GPU transfers.
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPUs available: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ds = Human36MPreprocessedClips(
        root=args.root,
        subjects=args.subjects,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        max_clips=None,
        augment=args.augment,  
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2,  
        collate_fn=augment_collate_fn if args.augment else None,
        # persistent_workers=True,  # keep workers alive for the entire epoch, faster than respawning each time
    )

    # ResNet50 backbone
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = backbone.to(device).eval()

    # -------------------------
    # multi-gpu with nn.DataParallel
    # -------------------------
    use_dp = device.startswith("cuda") and (torch.cuda.device_count() > 1)
    if use_dp:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        backbone = nn.DataParallel(backbone)

    # Compile for massive speedup (PyTorch 2.0+)
    use_compile = (not use_dp) and device.startswith("cuda")
    if use_compile:
        try:
            backbone = torch.compile(backbone, mode="max-autotune")
            print("Using torch.compile(backbone)")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    # async writer
    writer = AsyncFileWriter() 

    global_i = 0
    t_all = time.time()
    t_last = time.time()
    
    print("Warming up compiled model...")
    warmup_batch = next(iter(loader))
    if args.augment:
        warmup_video = warmup_batch[0][0]  # first variant's videos tensor
    else:
        warmup_video = warmup_batch[0]  
    warmup_video = warmup_video.to(device, non_blocking=True)
    B, T, C, H, W = warmup_video.shape
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", 
                        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
        _ = backbone(warmup_video.view(B * T, C, H, W).contiguous())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("✓ Warmup complete")

    n_clips  = len(ds)
    n_vars   = len(AUG_NAMES) if args.augment else 1

    print(f"Processing {n_clips} clips  ×  {n_vars} variant(s)  =  "
          f"{n_clips * n_vars} files total …")
    print("-" * 60)

    for it, batch in enumerate(loader):

        # ------------------------------------------------------------------
        # Normalise batch format.
        # Plain   → batch = (video, j3d, j2d, K, box)   — wrap in list
        # Augment → batch = [(video, j3d, j2d, K), ...]  — already a list
        # ------------------------------------------------------------------
        if args.augment:
            variants_batch = batch          # list of N variant-batches
            box_batch      = None           # box not available for augmented variants
        else:
            video, joints3d, joints2d, K, box = batch
            variants_batch = [(video, joints3d, joints2d, K)]
            box_batch      = box

        # Use the first variant to know the batch size
        B = variants_batch[0][0].shape[0]

        # ------------------------------------------------------------------
        # Run ResNet on every variant
        # ------------------------------------------------------------------
        all_feats = []   # shape per variant: (B, T, 2048)
        for v_idx, (v_video, v_j3d, v_j2d, v_K) in enumerate(variants_batch):
            v_video = v_video.to(device, non_blocking=True)
            Bv, T, C, H, W = v_video.shape

            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                enabled=device.startswith("cuda"),
            ):
                x     = v_video.view(Bv * T, C, H, W).contiguous()
                feats = backbone(x).flatten(1).view(Bv, T, -1)

            feats_cpu = (feats.half() if args.save_fp16 else feats.float()).cpu()
            all_feats.append(feats_cpu)

        # ------------------------------------------------------------------
        # Save one .pt file per (clip × variant)
        # ------------------------------------------------------------------
        for b in range(B):
            clip = ds.index[global_i]
            global_i += 1

            rel_dir  = Path(f"S{clip.subject}") / clip.action / f"{clip.cam}"
            save_dir = out_root / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            for v_idx, (v_video, v_j3d, v_j2d, v_K) in enumerate(variants_batch):
                aug_tag = f"_{AUG_NAMES[v_idx]}" if args.augment else ""
                fname   = (f"clip_{clip.start}_{clip.end}"
                           f"_fs{args.frame_skip}_len{args.seq_len}{aug_tag}.pt")
                save_path = save_dir / fname

                payload = {
                    "feats":    all_feats[v_idx][b],
                    "joints3d": v_j3d[b].cpu(),
                    "joints2d": v_j2d[b].cpu(),
                    "K":        v_K[b].cpu() if v_K.ndim >= 3 else v_K.cpu(),
                    "meta": {
                        "subject":    clip.subject,
                        "action":     clip.action,
                        "cam":        clip.cam,
                        "start":      clip.start,
                        "end":        clip.end,
                        "aug":        AUG_NAMES[v_idx] if args.augment else "orig",
                        "box":        box_batch[b].cpu() if box_batch is not None else None,
                        "seq_len":    args.seq_len,
                        "frame_skip": args.frame_skip,
                    },
                }
                writer.save(payload, save_path)

        # Progress
        if global_i % 100 == 0 or global_i == n_clips:
            dt            = time.time() - t_last
            clips_per_sec = 100 / dt if dt > 0 else 0
            t_last        = time.time()
            progress      = 100 * global_i / n_clips
            eta           = (n_clips - global_i) / clips_per_sec if clips_per_sec > 0 else 0
            print(f"[{progress:5.1f}%] {global_i:5d}/{n_clips} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA: {eta:6.1f}s")

    if writer:
        print("\nWaiting for file writes to complete...")
        writer.wait()
        writer.stop()

    total_time = time.time() - t_all
    print("-" * 60)
    print(f"✓ Done! Saved {global_i} clips to: {out_root}")
    print(f"✓ Total time: {total_time:.1f}s ({len(ds)/total_time:.1f} clips/s)")
    print(f"✓ Average time per clip: {1000*total_time/len(ds):.1f}ms")


if __name__ == "__main__":
    main()