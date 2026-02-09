import os
import glob
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision

from config import DEVICE, SEQ_LEN, JOINTS_NUM
from dataset_features import Human36MFeatureClips
from model import PHDFor3DJoints as PHD
from train import evaluate  # uses your existing evaluate() (metrics print)


"""
Use the trained model to generate results on the test set (Subject 9),
and save one example containing:
  - video clip (cropped/resized or raw; here we load the raw mp4 frames slice)
  - 3D joints GT
  - 3D joints predicted
  - meta

Coded by Luisa Ferreira, 2026.
"""


def _find_video_path(preprocessed_root: str, meta: dict) -> str:
    """
    meta is expected to contain:
      meta["subject"] (int or str), meta["action"] (str), meta["cam"] (str like "cam_0")
    and your folder layout:
      preprocessed_root / S{subject} / {action} / {cam} / *.mp4
    """
    subject = int(meta["subject"])
    action = meta["action"]
    cam = meta["cam"]

    cam_dir = os.path.join(preprocessed_root, f"S{subject}", action, cam)
    mp4s = sorted(glob.glob(os.path.join(cam_dir, "*.mp4")))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 found under {cam_dir}")
    return mp4s[0]

def collate_with_meta(batch):
    # batch: list of samples returned by Dataset.__getitem__
    # each sample is either (feats,j3d,j2d,K) or (feats,j3d,j2d,K,meta)
    if len(batch[0]) == 5:
        feats, j3d, j2d, K, meta = zip(*batch)
        return (
            torch.stack(feats, 0),
            torch.stack(j3d, 0),
            torch.stack(j2d, 0),
            torch.stack(K, 0),
            list(meta),  # keep python dicts as-is
        )
    else:
        feats, j3d, j2d, K = zip(*batch)
        return (
            torch.stack(feats, 0),
            torch.stack(j3d, 0),
            torch.stack(j2d, 0),
            torch.stack(K, 0),
        )



def _load_video_clip_from_meta(preprocessed_root: str, meta: dict) -> np.ndarray:
    """
    Loads the raw frames from the mp4 corresponding to this clip.

    meta is expected to contain:
      - start, end (clip indices)
      - frame_skip (to map clip indices to actual video frames)
    Returns:
      video_np: (T,H,W,3) uint8
    """
    video_path = _find_video_path(preprocessed_root, meta)

    start = int(meta["start"])
    end = int(meta["end"])
    frame_skip = int(meta.get("frame_skip", 1))

    frames, _, _info = torchvision.io.read_video(video_path, pts_unit="sec")  # (N,H,W,3) uint8
    frames = frames[::frame_skip]         # match your preprocessing
    frames = frames[start:end]            # take the clip range

    if frames.numel() == 0:
        raise RuntimeError(f"Loaded 0 frames from {video_path} with start={start}, end={end}, frame_skip={frame_skip}")

    return frames.numpy().astype(np.uint8)


@torch.no_grad()
def _predict_one_batch(model: torch.nn.Module, batch, device: torch.device):
    """
    Assumes dataset_features returns either:
      (feats, j3d, j2d, K)            for train/val
    or:
      (feats, j3d, j2d, K, meta)      for test
    """
    if len(batch) == 5:
        feats, joints3d_gt, joints2d, K, meta = batch
    else:
        feats, joints3d_gt, joints2d, K = batch
        meta = None

    feats = feats.to(device).float()  # (B,T,2048)
    # model forward from your training code:
    # forward(...) returns (j3d_past, j3d_future, j3d_pred_full)
    j3d_pred = model.forward(feats, predict_future=False)[2]  # (B,T,J,3)

    return joints3d_gt, j3d_pred.cpu(), meta


def main():
    parser = argparse.ArgumentParser("Test Subject 9 + dump one example to NPZ")
    parser.add_argument("--features_root", type=str, required=True,
                        help="Root of cached ResNet feature clips (contains S9/.../clip_*.pt)")
    parser.add_argument("--preprocessed_root", type=str, required=True,
                        help="Root of preprocessed mp4 videos (contains S9/.../cam_*/.mp4)")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to checkpoint saved by train.py (state_dict)")
    parser.add_argument("--out", type=str, default="outputs/batch_result_S9.npz")

    args = parser.parse_args()

    device = torch.device(DEVICE if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # --------- Load TEST set: Subject 9 ----------
    test_set = Human36MFeatureClips(
        root=args.features_root,
        subjects=[9],
        test_set=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,  # to handle meta in test set
    )

    # --------- Load model ----------
    model = PHD(joints_num=JOINTS_NUM).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # --------- Evaluate metrics on all test ----------
    avg_loss, avg_mpjpe, avg_l3d, avg_l2d = evaluate(model, test_loader, device=device, test_set=True)

    print(
        f"Test metrics | loss: {avg_loss:.6f} | mpjpe (m): {avg_mpjpe:.6f} "
        f"| mpjpe (mm): {avg_mpjpe*1000.0:.2f} | l3d: {avg_l3d:.6f} | l2d: {avg_l2d:.6f}"
    )

     # --------- Dump one BATCH ----------
    batch = next(iter(test_loader))
    joints3d_gt, joints3d_pred, meta = _predict_one_batch(model, batch, device)

    if meta is None:
        raise RuntimeError(
            "Your dataset_features did not return meta. "
            "Make sure test_set=True returns (feats, j3d, j2d, K, meta)."
        )

    B = joints3d_gt.shape[0]

    # Convert whole batch to numpy
    joints3d_gt_np = joints3d_gt.detach().cpu().numpy()      # (B,T,J,3)
    joints3d_pred_np = joints3d_pred.detach().cpu().numpy()  # (B,T,J,3)

    # Load videos for each sample in batch
    videos = []
    metas_payload = []
    for b in range(B):
        meta_b = meta[b]
        if not isinstance(meta_b, dict):
            raise RuntimeError(f"Expected meta[{b}] to be dict, got {type(meta_b)}")

        video_b = _load_video_clip_from_meta(args.preprocessed_root, meta_b)  # (T,H,W,3) uint8
        videos.append(video_b)

        # keep meta as python object inside npz
        metas_payload.append(meta_b)

    # Stack videos: requires all videos have same shape (they should: same seq_len + same resolution)
    videos_np = np.stack(videos, axis=0)  # (B,T,H,W,3)

    # Ensure output directory exists
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(
        out_path,
        video=videos_np,                       # (B,T,H,W,3) uint8
        joints3d=joints3d_gt_np,               # (B,T,J,3)
        predicted3djoints=joints3d_pred_np,    # (B,T,J,3)
        meta=np.array(metas_payload, dtype=object),
        test_metrics=np.array([avg_loss, avg_mpjpe, avg_l3d, avg_l2d], dtype=object),
    )

    print(f"[OK] Saved batch to: {out_path}")
    print(f"video shape: {videos_np.shape} | joints3d: {joints3d_gt_np.shape} | pred: {joints3d_pred_np.shape}")


if __name__ == "__main__":
    main()
