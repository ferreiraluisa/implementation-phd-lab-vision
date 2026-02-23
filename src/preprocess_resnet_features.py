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
AUG_NAMES = ["orig", "cjitter", "hflip", "trev"]

class AsyncFileWriter:
    # asynchronous file writer using a background thread and a queue, to speed up saving features to disk without blocking the main processing loop
    # this helped a lot!!!!!!!!
    # with augmentation data, solution was to save clips in shards of 100 clips each, to reduce the number of individual file writes and improve throughput.
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
            shard_dict, save_path = item
            torch.save(shard_dict, save_path, _use_new_zipfile_serialization=False)
            self.queue.task_done()
    
    def save(self, shard_dict, save_path):
        self.queue.put((shard_dict, save_path))
        self.count += 1
    
    def wait(self):
        self.queue.join()
    
    def stop(self):
        self.queue.put(None)
        self.thread.join()

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

def empty_shard_buffer():
    return {
        "feats": [],
        "joints3d": [],
        "joints2d": [],
        "K": [],
        "meta": []
    }

def flush_shard(buf, sid, n_vars, out_root, writer):
    shard_dict = {
        "feats":    torch.stack(buf["feats"]),     # (rows, T, 2048)
        "joints3d": torch.stack(buf["joints3d"]),  # (rows, T, 17, 3)
        "joints2d": torch.stack(buf["joints2d"]),  # (rows, T, 17, 2)
        "K":        torch.stack(buf["K"]),         # (rows, 3, 3)
        "meta":     buf["meta"],                   # list of dicts
        "n_vars":   n_vars,
    }
    path = out_root / f"shard_{sid:05d}.pt"
    writer.write(shard_dict, path)
    return empty_shard_buffer()


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
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation (random horizontal flip)")
    parser.add_argument("--shard-size", type=int, default=512, help="Number of clips per shard file")

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

    n_vars = len(AUG_NAMES) if args.augment else 1

    print(f"Device     : {device}")
    if device.startswith("cuda"):
        print(f"GPUs       : {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
    print(f"Augment    : {args.augment}  "
          f"({'4 variants/clip → ' + ', '.join(AUG_NAMES) if args.augment else 'none'})")
    print(f"Shard size : {args.shard_size} clips  ({args.shard_size * n_vars} variant entries/shard)")

    ds = Human36MPreprocessedClips(
        root=args.root,
        subjects=args.subjects,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        augment=args.augment,
        max_clips=None,
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
    warmup_video = (warmup_batch[0][0] if args.augment else warmup_batch[0]).to(device, non_blocking=True)
    B, T, C, H, W = warmup_video.shape
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                        enabled=device.startswith("cuda")):
        _ = backbone(warmup_video.view(B * T, C, H, W).contiguous())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("✓ Warmup complete\n")

    print(f"\nProcessing {len(ds)} clips...")
    print("-" * 60)

    # shard buffers 
    # accumulate clips in memory and save to disk in larger shard files to reduce overhead of many small file writes, especially important when augmenting (4x more entries).
    shard_buf = empty_shard_buffer()
    shard_id = 0    
    shard_clips = 0
    clip_index = []

    n_clips = len(ds)

    global_clip_i = 0
    t_all = time.time()
    t_last = time.time()

    print(f"Processing {n_clips} clips × {n_vars} variant(s) = {n_clips * n_vars} entries …")
    print(f"Writing shards of {args.shard_size} clips each → {out_root}")
    print("-" * 60)

    for batch in loader:
        if args.augment:
            variants_batch = batch
            box_batch = None
        else:
            video, joints3d, joints2d, K, box = batch
            variants_batch = [(video, joints3d, joints2d, K)]
            box_batch = box

        B = variants_batch[0][0].shape[0]  # batch size

        all_feats = []
        feat_dtype = torch.float16 if args.save_fp16 else torch.float32
        # run resnet on every variant
        for v_video, _j3d, _j2d, _K in variants_batch:
            v_video = v_video.to(device, non_blocking=True)
            Bv, T, C, H, W = v_video.shape
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                enabled=device.startswith("cuda"),
            ):
                x     = v_video.view(Bv * T, C, H, W).contiguous()
                feats = backbone(x).flatten(1).view(Bv, T, -1)
            all_feats.append(feats.to(feat_dtype).cpu())

        for b in range(B):
            clip = ds.index[global_clip_i]

            row_in_shard = shard_clips * n_vars
            clip_index.append({
                "shard_id": shard_id,
                "row": row_in_shard,
                "subject": clip.subject,
                "action": clip.action,
                "cam": clip.cam,
                "start": clip.start,
                "end": clip.end,
            })

            for v_idx, (v_video, v_j3d, v_j2d, v_K) in enumerate(variants_batch):
                shard_buf["feats"].append(all_feats[v_idx][b])
                shard_buf["joints3d"].append(v_j3d[b].cpu())
                shard_buf["joints2d"].append(v_j2d[b].cpu())
                shard_buf["K"].append(v_K[b].cpu() if v_K.ndim >= 3 else v_K.cpu())
                shard_buf["meta"].append({
                    "subject":    clip.subject,
                    "action":     clip.action,
                    "cam":        clip.cam,
                    "start":      clip.start,
                    "end":        clip.end,
                    "aug":        AUG_NAMES[v_idx] if args.augment else "orig",
                    "box":        box_batch[b].cpu() if box_batch is not None else None,
                })

            shard_clips  += 1
            global_clip_i += 1

            # Flush when shard is full
            if shard_clips >= args.shard_size:
                shard_buf = flush_shard(shard_buf, shard_id)
                shard_id   += 1
                shard_clips = 0

        # Progress
        if global_clip_i % 200 == 0 or global_clip_i == n_clips:
            dt            = time.time() - t_last
            clips_per_sec = 200 / dt if dt > 0 else 0
            t_last        = time.time()
            progress      = 100 * global_clip_i / n_clips
            eta           = (n_clips - global_clip_i) / clips_per_sec if clips_per_sec > 0 else 0
            print(f"[{progress:5.1f}%] {global_clip_i:6d}/{n_clips} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA {eta:6.1f}s | "
                  f"shard {shard_id} ({shard_clips} clips buffered)")

    # Flush any remaining clips
    if shard_clips > 0:
        flush_shard(shard_buf, shard_id)
        shard_id += 1
            
    # save global index
    print("\nWaiting for all shards to be written to disk...")
    writer.wait()
    writer.stop()

    index_path = out_root / "index.pt"
    torch.save({
        "clips": clip_index,
        "n_shards": shard_id,
        "n_clips": n_clips,
        "n_variants": n_vars,
        "aug_names": AUG_NAMES if args.augment else ["orig"],
        "seq_len": args.seq_len,
        "frame_skip": args.frame_skip,
        "feat_dtype": "float16" if args.save_fp16 else "float32",
    }, index_path)

    print("✓ All shards written and index saved.")

    total_time  = time.time() - t_all
    files_saved = shard_id  # number of shard files
    print("-" * 60)
    print(f"✓ Done!  {n_clips} clips × {n_vars} variant(s) packed into {files_saved} shard(s)")
    print(f"✓ Total time        : {total_time:.1f}s")
    print(f"✓ Throughput        : {n_clips / total_time:.1f} clips/s  "
          f"({n_clips * n_vars / total_time:.1f} variant entries/s)")
    print(f"✓ Avg time per clip : {1000 * total_time / n_clips:.1f} ms")


if __name__ == "__main__":
    main()