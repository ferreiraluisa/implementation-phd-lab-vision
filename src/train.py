# train.py
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    H36M_ROOT,
    SEQ_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
    JOINTS_NUM,
    FRAME_SKIP
)
from dataset import Human36MPreprocessedClips  
from model import PHDFor3DJoints as PHD


# -----------------------------
# Metrics / Losses
# -----------------------------

# Mean Per Joint Position Error (same units as your joints3d; often mm in H36M pipelines)
# pred, gt: (B,T,J,3)
@torch.no_grad()
def mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> float:
    err = torch.norm(pred - gt, dim=-1)  # (B,T,J)
    return float(err.mean().item())


def save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer,
                    epoch: int, best_val: float, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
        },
        path,
    )


# OPTION 1: project with intrinsic matrix K only (no distortion)
# K = [[fx, 0, cx],
#      [0, fy, cy],
#      [0,  0,  1]]
# K * P_cam = [u, v, w]  =>  uv = [u/w, v/w](2d pixels)
def project_with_K_torch(P_cam, K, eps):
    P_col = P_cam.unsqueeze(-1)                           # (...,3,1)
    P_h = torch.matmul(K, P_col).squeeze(-1)              # (...,3)

    z = P_h[..., 2:3].clamp(min=eps)
    uv = P_h[..., 0:2] / z                                # (...,2)
    return uv


def train(model, loader, optim, scaler, device, lambda_2d: float = 1.0, log_every: int = 100):
    model.train()
    t0 = time.time()

    running_loss = 0.0
    running_l3d = 0.0
    running_l2d = 0.0
    running_mpjpe = 0.0
    n_batches = 0

    for it, batch in enumerate(loader):
        # NEW: dataset returns (video, joints3d, joints2d, K)
        video, joints3d, joints2d, K = batch

        video = video.to(device)         # (B,T,3,H,W)
        joints3d = joints3d.to(device)   # (B,T,J,3)
        joints2d = joints2d.to(device)   # (B,T,J,2) pixel coords in the *same* cropped/resized image space as video
        K = K.to(device)                 # (3,3) or (B,3,3) or (B,T,3,3)

        optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.startswith("cuda"))):
            _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)
            # joints_pred: (B,T,J,3)  assumed camera coordinates that match K

            # 3D loss
            l3d = (joints_pred - joints3d).pow(2).mean()

            # 2D reprojection loss
            proj2d = project_with_K_torch(joints_pred, K, eps=1e-6)  # (B,T,J,2)
            l2d = (proj2d - joints2d).pow(2).mean()

            loss = l3d + (lambda_2d * l2d)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        running_loss += float(loss.item())
        running_l3d += float(l3d.item())
        running_l2d += float(l2d.item())
        running_mpjpe += mpjpe_mm(joints_pred.detach(), joints3d.detach())
        n_batches += 1

        if log_every > 0 and (it + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"  iter {it+1:05d}/{len(loader):05d} | "
                f"loss {running_loss/n_batches:.6f} (3d {running_l3d/n_batches:.6f} + "
                f"{lambda_2d:.3g}*2d {running_l2d/n_batches:.6f}) | "
                f"mpjpe {running_mpjpe/n_batches:.3f} | "
                f"time {dt:.1f}s"
            )

    return running_loss / max(n_batches, 1), running_mpjpe / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, lambda_2d: float = 1.0):
    model.eval()

    total_loss = 0.0
    total_l3d = 0.0
    total_l2d = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    for batch in loader:
        video, joints3d, joints2d, K = batch

        video = video.to(device)
        joints3d = joints3d.to(device)
        joints2d = joints2d.to(device)
        K = K.to(device)

        _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)

        l3d = (joints_pred - joints3d).pow(2).mean()
        proj2d = project_with_K_torch(joints_pred, K)
        l2d = (proj2d - joints2d).pow(2).mean()

        loss = l3d + (lambda_2d * l2d)

        total_loss += float(loss.item())
        total_l3d += float(l3d.item())
        total_l2d += float(l2d.item())
        total_mpjpe += mpjpe_mm(joints_pred, joints3d)
        n_batches += 1

    # (optionally you can also print l3d/l2d outside)
    return (
        total_loss / max(n_batches, 1),
        total_mpjpe / max(n_batches, 1),
        total_l3d / max(n_batches, 1),
        total_l2d / max(n_batches, 1),
    )


def main():
    parser = argparse.ArgumentParser("Phase-1 training: freeze ResNet, train f_movie + f_3D (3D joints + 2D reprojection)")
    parser.add_argument("--root", type=str, default=H36M_ROOT)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--frame_skip", type=int, default=FRAME_SKIP, help="frame subsampling rate during video loading")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5, help="clip sampling stride in frames (dataset indexing)")
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--max-val-clips", type=int, default=None)
    parser.add_argument("--lambda-2d", type=float, default=1.0, help="2D reprojection loss weight")
    parser.add_argument("--outdir", type=str, default="./runs/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = DEVICE
    os.makedirs(args.outdir, exist_ok=True)

    # load data
    train_set = Human36MPreprocessedClips(
        root=args.root,
        subjects=[1, 6, 7, 8],
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        max_clips=args.max_train_clips,
    )
    val_set = Human36MPreprocessedClips(
        root=args.root,
        subjects=[5],
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=max(1, args.stride),
        max_clips=args.max_val_clips,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    model = PHD(latent_dim=2048, joints_num=JOINTS_NUM, freeze_backbone=True).to(device)

    # ----------------------------------
    # TRAINING PHASE 1 : freeze ResNet, train f_movie + f_3D
    # ----------------------------------
    for p in model.f_AR.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found. Did you accidentally freeze everything?")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_val={best_val:.4f})")

    print("===== Phase-1 training =====")
    print(f"Device: {device}")
    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")
    print(f"Seq len: {args.seq_len} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"Lambda 2D: {args.lambda_2d}")
    print("============================")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        tr_loss, tr_mpjpe = train(
            model, train_loader, optim, scaler, device,
            lambda_2d=args.lambda_2d,
            log_every=args.log_every
        )
        va_loss, va_mpjpe, va_l3d, va_l2d = evaluate(
            model, val_loader, device,
            lambda_2d=args.lambda_2d
        )

        print(f"Train: loss={tr_loss:.6f} | mpjpe={tr_mpjpe:.3f}")
        print(f"Val:   loss={va_loss:.6f} (3d {va_l3d:.6f} + {args.lambda_2d:.3g}*2d {va_l2d:.6f}) | mpjpe={va_mpjpe:.3f}")

        save_checkpoint(
            os.path.join(args.outdir, "last.pt"),
            model, optim, epoch, best_val, args
        )

        if va_mpjpe < best_val:
            best_val = va_mpjpe
            save_checkpoint(
                os.path.join(args.outdir, "best.pt"),
                model, optim, epoch, best_val, args
            )
            print(f"New best val MPJPE: {best_val:.3f} (saved best.pt)")

    print("\nDone.")
    print(f"Best val MPJPE: {best_val:.3f}")


if __name__ == "__main__":
    main()
