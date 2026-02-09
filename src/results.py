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
from train import evaluate


"""
Use the trained model to generate results on the test set (Subject 9),
and save ONE BATCH containing:
  - video clips (raw mp4 slice)
  - 3D joints GT
  - 3D joints predicted
  - meta

Coded by Luisa Ferreira, 2026.
"""


def _find_video_path(preprocessed_root: str, meta: dict) -> str:
    subject = int(meta["subject"])
    action = str(meta["action"])
    cam = str(meta["cam"])
    if not cam.startswith("cam_"):
        cam = f"cam_{cam}"

    cam_dir = os.path.join(preprocessed_root, f"S{subject}", action, cam)
    mp4s = sorted(glob.glob(os.path.join(cam_dir, "*.mp4")))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 found under {cam_dir}")
    return mp4s[0]


def collate_with_meta(batch):
    # each sample is either (feats,j3d,j2d,K) or (feats,j3d,j2d,K,meta)
    if len(batch[0]) == 5:
        feats, j3d, j2d, K, meta = zip(*batch)
        return (
            torch.stack(feats, 0),
            torch.stack(j3d, 0),
            torch.stack(j2d, 0),
            torch.stack(K, 0),
            list(meta),  # keep python dicts
        )
    else:
        feats, j3d, j2d, K = zip(*batch)
        return (
            torch.stack(feats, 0),
            torch.stack(j3d, 0),
            torch.stack(j2d, 0),
            torch.stack(K, 0),
        )


def _pad_or_trim_video(video: np.ndarray, target_T: int) -> np.ndarray:
    """
    video: (T,H,W,3) uint8
    Ensures T == target_T by trimming or padding last frame.
    """
    T = video.shape[0]
    if T == target_T:
        return video
    if T > target_T:
        return video[:target_T]
    # pad with last frame
    pad_n = target_T - T
    last = video[-1:]
    pad = np.repeat(last, pad_n, axis=0)
    return np.concatenate([video, pad], axis=0)


def _load_video_clip_from_meta(preprocessed_root: str, meta: dict, seq_len: int) -> np.ndarray:
    video_path = _find_video_path(preprocessed_root, meta)

    start = int(meta["start"])
    end = int(meta["end"])
    frame_skip = int(meta.get("frame_skip", 1))

    frames, _, _info = torchvision.io.read_video(video_path, pts_unit="sec")  # (N,H,W,3) uint8
    frames = frames[::frame_skip]
    frames = frames[start:end]

    if frames.numel() == 0:
        raise RuntimeError(
            f"Loaded 0 frames from {video_path} with start={start}, end={end}, frame_skip={frame_skip}"
        )

    video_np = frames.numpy().astype(np.uint8)  # (T,H,W,3)
    video_np = _pad_or_trim_video(video_np, seq_len)
    return video_np


@torch.no_grad()
def _predict_one_batch(model: torch.nn.Module, batch, device: torch.device):
    if len(batch) == 5:
        feats, joints3d_gt, joints2d, K, meta = batch
    else:
        feats, joints3d_gt, joints2d, K = batch
        meta = None

    feats = feats.to(device).float()
    j3d_pred = model.forward(feats, predict_future=False)[2]  # (B,T,J,3)
    return joints3d_gt, j3d_pred.detach().cpu(), meta


def _safe_device(prefer: str) -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser("Test Subject 9 + dump ONE BATCH to NPZ")
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--preprocessed_root", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/batch_result_S9.npz")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--save-n", type=int, default=16, help="How many samples from the batch to save")
    args = parser.parse_args()

    device = _safe_device(args.device)

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
        collate_fn=collate_with_meta,
    )

    # --------- Load model (with ECC fallback) ----------
    try:
        model = PHD(joints_num=JOINTS_NUM).to(device)
    except RuntimeError as e:
        if "ECC" in str(e) or "uncorrectable" in str(e):
            print("[WARN] CUDA ECC error. Falling back to CPU.")
            device = torch.device("cpu")
            model = PHD(joints_num=JOINTS_NUM).to(device)
        else:
            raise

    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # --------- Evaluate metrics ----------
    avg_loss, avg_mpjpe, avg_l3d, avg_l2d = evaluate(model, test_loader, device=device, test_set=True)
    print(
        f"Test metrics | loss: {avg_loss:.6f} | mpjpe (m): {avg_mpjpe:.6f} "
        f"| mpjpe (mm): {avg_mpjpe*1000.0:.2f} | l3d: {avg_l3d:.6f} | l2d: {avg_l2d:.6f}"
    )

    # --------- Dump ONE BATCH ----------
    batch = next(iter(test_loader))
    joints3d_gt, joints3d_pred, meta = _predict_one_batch(model, batch, device)

    if meta is None:
        raise RuntimeError("Meta is None. Ensure dataset_features returns meta when test_set=True.")

    B = joints3d_gt.shape[0]
    B = min(B, args.save_n)

    joints3d_gt_np = joints3d_gt[:B].detach().cpu().numpy()      # (B,T,J,3)
    joints3d_pred_np = joints3d_pred[:B].detach().cpu().numpy()  # (B,T,J,3)

    videos = []
    metas_payload = []
    for b in range(B):
        meta_b = meta[b]
        if not isinstance(meta_b, dict):
            raise RuntimeError(f"Expected meta[{b}] to be dict, got {type(meta_b)}")

        video_b = _load_video_clip_from_meta(args.preprocessed_root, meta_b, seq_len=args.seq_len)
        videos.append(video_b)
        metas_payload.append(meta_b)

    videos_np = np.stack(videos, axis=0)  # (B,T,H,W,3)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(
        out_path,
        video=videos_np,
        joints3d=joints3d_gt_np,
        predicted3djoints=joints3d_pred_np,
        meta=np.array(metas_payload, dtype=object),
        test_metrics=np.array([avg_loss, avg_mpjpe, avg_l3d, avg_l2d], dtype=np.float32),
    )

    print(f"[OK] Saved batch to: {out_path}")
    print(f"video shape: {videos_np.shape} | joints3d: {joints3d_gt_np.shape} | pred: {joints3d_pred_np.shape}")


if __name__ == "__main__":
    main()
