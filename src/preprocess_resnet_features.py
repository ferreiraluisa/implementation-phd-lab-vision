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

import random

"""
This script precomputes ResNet50 features for Human3.6M video clips and saves them to disk.
Each clip's features, along with 2D/3D joints and camera intrinsics, are stored in a structured directory format.
This allows for efficient loading during model training or evaluation. Avoiding redundant computation of ResNet features speeds up experiments significantly.

Coded by Luísa Ferreira, 2026

Used Claude to help optimize and refactor the code for better performance and readability.
(older version was taking too long(5 hours to process only 6k/200k clips), this version processes all ~20k clips in ~30 minutes on a single GPU).

Shard writing is fully randomized: all processed entries (clips × variants) are buffered into a
large shuffle pool, then randomly permuted before being packed into shards. This ensures that:
  - Clips from the same subject/action/camera are scattered across different shards.
  - Variants (orig, cjitter, hflip, trev) of the same clip land in different shards.
  - Your within-shard sampler sees maximally decorrelated clips at training time.

The shuffle pool size (--shuffle-pool) controls the trade-off between RAM usage and randomness
quality. Default is 8192 clips, which at fp16 + seq_len=40 costs ~5 GB RAM. Set lower if needed.
All entries that don't fit in a full shard at pool-flush time are carried over to the next pool.
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


def empty_entry():
    return {
        "feat":     None,   # (T, 2048)
        "joints3d": None,   # (T, 17, 3)
        "joints2d": None,   # (T, 17, 2)
        "K":        None,   # (3, 3)
        "meta":     None,   # dict
    }


def flush_pool_to_shards(pool, shard_id, n_vars, out_root, writer, shard_size, clip_index, carry_over):
    """
    Shuffle `pool` (list of entry dicts) in-place, then pack entries into full shards.
    Any leftover entries that don't fill a complete shard are returned as `carry_over`
    and will be prepended to the next pool.

    Each shard contains exactly `shard_size` entries (variant rows).
    The clip_index is updated with the correct shard_id and row for every entry.
    Returns (shard_id, carry_over).
    """
    # Combine carry-over from the previous pool flush with the new pool entries.
    combined = carry_over + pool

    # Fully randomize order — this breaks subject/action/variant locality.
    random.shuffle(combined)

    n_full_shards = len(combined) // shard_size
    for s in range(n_full_shards):
        entries = combined[s * shard_size : (s + 1) * shard_size]
        shard_dict = {
            "feats":    torch.stack([e["feat"]     for e in entries]),  # (shard_size, T, 2048)
            "joints3d": torch.stack([e["joints3d"] for e in entries]),  # (shard_size, T, 17, 3)
            "joints2d": torch.stack([e["joints2d"] for e in entries]),  # (shard_size, T, 17, 2)
            "K":        torch.stack([e["K"]        for e in entries]),  # (shard_size, 3, 3)
            "meta":     [e["meta"] for e in entries],
            "n_vars":   n_vars,
        }
        path = out_root / f"shard_{shard_id:05d}.pt"
        writer.save(shard_dict, path)

        # Record where each entry landed so the index stays accurate.
        for row, e in enumerate(entries):
            e["meta"]["_shard_id"] = shard_id
            e["meta"]["_row"]      = row
            clip_index.append({
                "shard_id": shard_id,
                "row":      row,
                "subject":  e["meta"]["subject"],
                "action":   e["meta"]["action"],
                "cam":      e["meta"]["cam"],
                "start":    e["meta"]["start"],
                "end":      e["meta"]["end"],
            })

        shard_id += 1

    # Entries that didn't fill a complete shard become carry-over for the next pool.
    new_carry_over = combined[n_full_shards * shard_size:]
    return shard_id, new_carry_over


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
    parser.add_argument("--shard-size", type=int, default=512, help="Number of entries per shard file")
    parser.add_argument("--shuffle-pool", type=int, default=8192,
                        help="Number of clips to accumulate before shuffling and writing shards. "
                             "Larger = better randomness, more RAM. "
                             "At fp16 + seq_len=40: ~5 GB for 8192 clips × 4 variants.")

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

    n_vars = len(AUG_NAMES) if args.augment else 1

    print(f"Device     : {device}")
    if device.startswith("cuda"):
        print(f"GPUs       : {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
    print(f"Augment    : {args.augment}  "
          f"({'4 variants/clip → ' + ', '.join(AUG_NAMES) if args.augment else 'none'})")
    print(f"Shard size : {args.shard_size} entries")
    print(f"Shuffle pool: {args.shuffle_pool} clips "
          f"({args.shuffle_pool * n_vars} variant entries) — "
          f"variants of the same clip are shuffled independently across shards")

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

    use_dp = device.startswith("cuda") and (torch.cuda.device_count() > 1)
    if use_dp:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        backbone = nn.DataParallel(backbone)

    use_compile = (not use_dp) and device.startswith("cuda")
    if use_compile:
        try:
            backbone = torch.compile(backbone, mode="max-autotune")
            print("Using torch.compile(backbone)")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    # async writer
    writer = AsyncFileWriter()

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

    n_clips = len(ds)
    print(f"Processing {n_clips} clips × {n_vars} variant(s) = {n_clips * n_vars} entries …")
    print(f"Writing shards of {args.shard_size} entries each → {out_root}")
    print("-" * 60)

    feat_dtype = torch.float16 if args.save_fp16 else torch.float32

    # --- Shuffle pool & shard state ---
    # `shuffle_pool` accumulates variant entries (one per clip × aug variant).
    # When it reaches `shuffle_pool_size` clips, we shuffle all entries and
    # flush as many complete shards as possible. The remainder carries over.
    shuffle_pool_size = args.shuffle_pool * n_vars   # in entries (not clips)
    shuffle_pool: list = []          # list of entry dicts
    carry_over:   list = []          # entries leftover from last pool flush
    clip_index:   list = []
    shard_id:     int  = 0
    pool_clips:   int  = 0           # clips accumulated in current pool

    global_clip_i = 0
    t_all  = time.time()
    t_last = time.time()

    for batch in loader:
        if args.augment:
            variants_batch = batch
            box_batch = None
        else:
            video, joints3d, joints2d, K, box = batch
            variants_batch = [(video, joints3d, joints2d, K)]
            box_batch = box

        B = variants_batch[0][0].shape[0]

        # Run ResNet on every variant.
        all_feats = []
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

        # Add each clip's variants as independent entries in the shuffle pool.
        # Variants are NOT grouped — they're added as separate entries so the
        # shuffle can scatter them to completely different shards.
        for b in range(B):
            clip = ds.index[global_clip_i]

            for v_idx, (v_video, v_j3d, v_j2d, v_K) in enumerate(variants_batch):
                entry = {
                    "feat":     all_feats[v_idx][b],
                    "joints3d": v_j3d[b].cpu(),
                    "joints2d": v_j2d[b].cpu(),
                    "K":        v_K[b].cpu() if v_K.ndim >= 3 else v_K.cpu(),
                    "meta": {
                        "subject": clip.subject,
                        "action":  clip.action,
                        "cam":     clip.cam,
                        "start":   clip.start,
                        "end":     clip.end,
                        "aug":     AUG_NAMES[v_idx] if args.augment else "orig",
                        "box":     box_batch[b].cpu() if box_batch is not None else None,
                    },
                }
                shuffle_pool.append(entry)

            pool_clips    += 1
            global_clip_i += 1

            # When the pool is full, shuffle and flush complete shards.
            if len(shuffle_pool) >= shuffle_pool_size:
                shard_id, carry_over = flush_pool_to_shards(
                    shuffle_pool, shard_id, n_vars, out_root, writer,
                    args.shard_size, clip_index, carry_over
                )
                shuffle_pool = []
                pool_clips   = 0

        # Progress
        if global_clip_i % 200 == 0 or global_clip_i == n_clips:
            dt            = time.time() - t_last
            clips_per_sec = 200 / dt if dt > 0 else 0
            t_last        = time.time()
            progress      = 100 * global_clip_i / n_clips
            eta           = (n_clips - global_clip_i) / clips_per_sec if clips_per_sec > 0 else 0
            print(f"[{progress:5.1f}%] {global_clip_i:6d}/{n_clips} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA {eta:6.1f}s | "
                  f"shards written: {shard_id} | pool: {len(shuffle_pool)} entries buffered")

    # Final flush: shuffle whatever remains (pool + carry_over) and write all shards,
    # including a potentially partial last shard.
    final_pool = carry_over + shuffle_pool
    random.shuffle(final_pool)

    # Write all complete shards from the final pool.
    n_full = len(final_pool) // args.shard_size
    for s in range(n_full):
        entries = final_pool[s * args.shard_size : (s + 1) * args.shard_size]
        shard_dict = {
            "feats":    torch.stack([e["feat"]     for e in entries]),
            "joints3d": torch.stack([e["joints3d"] for e in entries]),
            "joints2d": torch.stack([e["joints2d"] for e in entries]),
            "K":        torch.stack([e["K"]        for e in entries]),
            "meta":     [e["meta"] for e in entries],
            "n_vars":   n_vars,
        }
        path = out_root / f"shard_{shard_id:05d}.pt"
        writer.save(shard_dict, path)
        for row, e in enumerate(entries):
            clip_index.append({
                "shard_id": shard_id,
                "row":      row,
                "subject":  e["meta"]["subject"],
                "action":   e["meta"]["action"],
                "cam":      e["meta"]["cam"],
                "start":    e["meta"]["start"],
                "end":      e["meta"]["end"],
            })
        shard_id += 1

    # Write the partial last shard (if any leftover entries).
    leftover = final_pool[n_full * args.shard_size:]
    if leftover:
        shard_dict = {
            "feats":    torch.stack([e["feat"]     for e in leftover]),
            "joints3d": torch.stack([e["joints3d"] for e in leftover]),
            "joints2d": torch.stack([e["joints2d"] for e in leftover]),
            "K":        torch.stack([e["K"]        for e in leftover]),
            "meta":     [e["meta"] for e in leftover],
            "n_vars":   n_vars,
        }
        path = out_root / f"shard_{shard_id:05d}.pt"
        writer.save(shard_dict, path)
        for row, e in enumerate(leftover):
            clip_index.append({
                "shard_id": shard_id,
                "row":      row,
                "subject":  e["meta"]["subject"],
                "action":   e["meta"]["action"],
                "cam":      e["meta"]["cam"],
                "start":    e["meta"]["start"],
                "end":      e["meta"]["end"],
            })
        shard_id += 1

    # Save global index.
    print("\nWaiting for all shards to be written to disk...")
    writer.wait()
    writer.stop()

    index_path = out_root / "index.pt"
    torch.save({
        "clips":       clip_index,
        "n_shards":    shard_id,
        "n_clips":     n_clips,
        "n_variants":  n_vars,
        "aug_names":   AUG_NAMES if args.augment else ["orig"],
        "seq_len":     args.seq_len,
        "frame_skip":  args.frame_skip,
        "feat_dtype":  "float16" if args.save_fp16 else "float32",
        "randomized":  True,  # shards are fully shuffled; variants of the same clip may be in different shards
    }, index_path)

    print("✓ All shards written and index saved.")

    total_time = time.time() - t_all
    print("-" * 60)
    print(f"✓ Done!  {n_clips} clips × {n_vars} variant(s) packed into {shard_id} shard(s)")
    print(f"✓ Total time        : {total_time:.1f}s")
    print(f"✓ Throughput        : {n_clips / total_time:.1f} clips/s  "
          f"({n_clips * n_vars / total_time:.1f} variant entries/s)")
    print(f"✓ Avg time per clip : {1000 * total_time / n_clips:.1f} ms")
    print(f"✓ Shards are fully randomized — no subject/action/variant locality within shards.")


if __name__ == "__main__":
    main()