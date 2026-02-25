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

Shard writing is randomized at the CLIP level: clips are buffered into a large shuffle pool,
randomly permuted, then packed into shards. This ensures that:
  - Clips from the same subject/action/camera are scattered across different shards.
  - All n_vars variants of a clip stay as consecutive rows within the same shard,
    which is required by the dataset class (row = clip["row"] + var_offset).
  - Your within-shard sampler sees maximally decorrelated clips at training time.

The shuffle pool size (--shuffle-pool) controls the trade-off between RAM usage and randomness
quality. Default is 8192 clips, which at fp16 + seq_len=40 costs ~5 GB RAM. Set lower if needed.
Clips that don't fill a complete shard at pool-flush time carry over to the next pool.
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


def flush_pool_to_shards(pool, shard_id, n_vars, out_root, writer, shard_size, clip_index, carry_over):
    """
    Shuffle `pool` (list of clip dicts) at the CLIP level, then pack into shards.

    Each clip occupies exactly n_vars consecutive rows:
        row 0 → orig, row 1 → cjitter, row 2 → hflip, row 3 → trev  (or just row 0 if no augment)
    This matches the dataset access pattern: shard["feats"][clip["row"] + var_offset].

    shard_size must be a multiple of n_vars so clips are never split across shards.
    Clips that don't fill a complete shard carry over to the next pool flush.
    Returns (shard_id, carry_over).
    """
    assert shard_size % n_vars == 0, \
        f"--shard-size ({shard_size}) must be a multiple of n_vars ({n_vars})"

    clips_per_shard = shard_size // n_vars
    combined = carry_over + pool

    # Shuffle at clip level — subjects/actions get mixed, variants stay together.
    random.shuffle(combined)

    n_full_shards = len(combined) // clips_per_shard
    for s in range(n_full_shards):
        clips = combined[s * clips_per_shard : (s + 1) * clips_per_shard]

        # Flatten variants: for each clip write all n_vars rows consecutively.
        rows_feat, rows_j3d, rows_j2d, rows_K, rows_meta = [], [], [], [], []
        for clip_i, c in enumerate(clips):
            base_row = clip_i * n_vars
            for v_idx in range(n_vars):
                rows_feat.append(c["variants"][v_idx]["feat"])
                rows_j3d.append(c["variants"][v_idx]["joints3d"])
                rows_j2d.append(c["variants"][v_idx]["joints2d"])
                rows_K.append(c["variants"][v_idx]["K"])
                rows_meta.append(c["variants"][v_idx]["meta"])

            # clip_index points to the FIRST row (orig); dataset adds var_offset.
            clip_index.append({
                "shard_id": shard_id,
                "row":      base_row,
                "subject":  c["subject"],
                "action":   c["action"],
                "cam":      c["cam"],
                "start":    c["start"],
                "end":      c["end"],
            })

        shard_dict = {
            "feats":    torch.stack(rows_feat),    # (shard_size, T, 2048)
            "joints3d": torch.stack(rows_j3d),     # (shard_size, T, 17, 3)
            "joints2d": torch.stack(rows_j2d),     # (shard_size, T, 17, 2)
            "K":        torch.stack(rows_K),        # (shard_size, 3, 3)
            "meta":     rows_meta,
            "n_vars":   n_vars,
        }
        path = out_root / f"shard_{shard_id:05d}.pt"
        writer.save(shard_dict, path)
        shard_id += 1

    new_carry_over = combined[n_full_shards * clips_per_shard:]
    return shard_id, new_carry_over


def write_final_shard(clips, shard_id, n_vars, out_root, writer, clip_index):
    """Write a partial (last) shard from whatever clips remain."""
    rows_feat, rows_j3d, rows_j2d, rows_K, rows_meta = [], [], [], [], []
    for clip_i, c in enumerate(clips):
        base_row = clip_i * n_vars
        for v_idx in range(n_vars):
            rows_feat.append(c["variants"][v_idx]["feat"])
            rows_j3d.append(c["variants"][v_idx]["joints3d"])
            rows_j2d.append(c["variants"][v_idx]["joints2d"])
            rows_K.append(c["variants"][v_idx]["K"])
            rows_meta.append(c["variants"][v_idx]["meta"])
        clip_index.append({
            "shard_id": shard_id,
            "row":      base_row,
            "subject":  c["subject"],
            "action":   c["action"],
            "cam":      c["cam"],
            "start":    c["start"],
            "end":      c["end"],
        })
    shard_dict = {
        "feats":    torch.stack(rows_feat),
        "joints3d": torch.stack(rows_j3d),
        "joints2d": torch.stack(rows_j2d),
        "K":        torch.stack(rows_K),
        "meta":     rows_meta,
        "n_vars":   n_vars,
    }
    path = out_root / f"shard_{shard_id:05d}.pt"
    writer.save(shard_dict, path)


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
    clips_per_shard = args.shard_size // n_vars
    assert args.shard_size % n_vars == 0, \
        f"--shard-size ({args.shard_size}) must be divisible by n_vars ({n_vars})"
    print(f"Shard size : {args.shard_size} rows = {clips_per_shard} clips × {n_vars} variant(s)")
    print(f"Shuffle pool: {args.shuffle_pool} clips — "
          f"clips shuffled at clip level, variants stay as consecutive rows within the same shard")

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
    # `shuffle_pool` is a list of CLIP dicts. Each clip dict holds all its variant
    # data nested inside, so the shuffle operates at clip granularity — variants
    # always travel together and land as consecutive rows in the same shard.
    shuffle_pool: list = []
    carry_over:   list = []
    clip_index:   list = []
    shard_id:     int  = 0

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

        # Build one clip-level dict per sample in the batch.
        # All variants are nested inside so they always stay together through shuffling.
        for b in range(B):
            clip = ds.index[global_clip_i]

            clip_dict = {
                "subject": clip.subject,
                "action":  clip.action,
                "cam":     clip.cam,
                "start":   clip.start,
                "end":     clip.end,
                "variants": [
                    {
                        "feat":     all_feats[v_idx][b],
                        "joints3d": variants_batch[v_idx][1][b].cpu(),
                        "joints2d": variants_batch[v_idx][2][b].cpu(),
                        "K":        variants_batch[v_idx][3][b].cpu()
                                    if variants_batch[v_idx][3].ndim >= 3
                                    else variants_batch[v_idx][3].cpu(),
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
                    for v_idx in range(n_vars)
                ],
            }
            shuffle_pool.append(clip_dict)
            global_clip_i += 1

            # Flush when the pool reaches the target size.
            if len(shuffle_pool) >= args.shuffle_pool:
                shard_id, carry_over = flush_pool_to_shards(
                    shuffle_pool, shard_id, n_vars, out_root, writer,
                    args.shard_size, clip_index, carry_over
                )
                shuffle_pool = []

        # Progress
        if global_clip_i % 200 == 0 or global_clip_i == n_clips:
            dt            = time.time() - t_last
            clips_per_sec = 200 / dt if dt > 0 else 0
            t_last        = time.time()
            progress      = 100 * global_clip_i / n_clips
            eta           = (n_clips - global_clip_i) / clips_per_sec if clips_per_sec > 0 else 0
            print(f"[{progress:5.1f}%] {global_clip_i:6d}/{n_clips} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA {eta:6.1f}s | "
                  f"shards written: {shard_id} | pool: {len(shuffle_pool)} clips buffered")

    # Final flush: shuffle whatever remains (pool + carry_over) and write all shards,
    # including a potentially partial last shard.
    final_pool = carry_over + shuffle_pool
    random.shuffle(final_pool)

    clips_per_shard = args.shard_size // n_vars
    n_full = len(final_pool) // clips_per_shard

    # Write all complete shards.
    for s in range(n_full):
        clips = final_pool[s * clips_per_shard : (s + 1) * clips_per_shard]
        rows_feat, rows_j3d, rows_j2d, rows_K, rows_meta = [], [], [], [], []
        for clip_i, c in enumerate(clips):
            base_row = clip_i * n_vars
            for v_idx in range(n_vars):
                rows_feat.append(c["variants"][v_idx]["feat"])
                rows_j3d.append(c["variants"][v_idx]["joints3d"])
                rows_j2d.append(c["variants"][v_idx]["joints2d"])
                rows_K.append(c["variants"][v_idx]["K"])
                rows_meta.append(c["variants"][v_idx]["meta"])
            clip_index.append({
                "shard_id": shard_id,
                "row":      base_row,
                "subject":  c["subject"],
                "action":   c["action"],
                "cam":      c["cam"],
                "start":    c["start"],
                "end":      c["end"],
            })
        writer.save({
            "feats":    torch.stack(rows_feat),
            "joints3d": torch.stack(rows_j3d),
            "joints2d": torch.stack(rows_j2d),
            "K":        torch.stack(rows_K),
            "meta":     rows_meta,
            "n_vars":   n_vars,
        }, out_root / f"shard_{shard_id:05d}.pt")
        shard_id += 1

    # Write the partial last shard (if any leftover clips remain).
    leftover = final_pool[n_full * clips_per_shard:]
    if leftover:
        write_final_shard(leftover, shard_id, n_vars, out_root, writer, clip_index)
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
        "randomized":  True,  # clips shuffled across shards; variants of each clip are consecutive rows in the same shard
    }, index_path)

    print("✓ All shards written and index saved.")

    total_time = time.time() - t_all
    print("-" * 60)
    print(f"✓ Done!  {n_clips} clips × {n_vars} variant(s) packed into {shard_id} shard(s)")
    print(f"✓ Total time        : {total_time:.1f}s")
    print(f"✓ Throughput        : {n_clips / total_time:.1f} clips/s  "
          f"({n_clips * n_vars / total_time:.1f} variant entries/s)")
    print(f"✓ Avg time per clip : {1000 * total_time / n_clips:.1f} ms")
    print(f"✓ Shards are randomized at clip level — variants stay together as consecutive rows.")


if __name__ == "__main__":
    main()