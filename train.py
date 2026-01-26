# train.py
import os
import time
import math
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    H36M_ROOT,
    SEQ_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
    JOINTS_NUM,
)
from dataset import Human36MPreprocessedClips
from model import PHDFor3DJoints


# -----------------------------
# Utils
# -----------------------------
def seed_all(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster
    torch.backends.cudnn.deterministic = False


@torch.no_grad()
def mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Mean Per Joint Position Error.
    pred, gt: (B,T,J,3)
    Returns mm (assumes your GT is in mm; if it's meters, multiply by 1000 outside).
    """
    err = torch.norm(pred - gt, dim=-1)  # (B,T,J)
    return float(err.mean().item())


def velocity_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Optional smoothness: match finite-difference velocities.
    pred, gt: (B,T,J,3)
    """
    pred_v = pred[:, 1:] - pred[:, :-1]
    gt_v = gt[:, 1:] - gt[:, :-1]
    return (pred_v - gt_v).pow(2).mean()


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


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optim, scaler, device, lambda_vel: float = 0.0, log_every: int = 50):
    model.train()
    t0 = time.time()

    running_loss = 0.0
    running_mpjpe = 0.0
    n_batches = 0

    for it, batch in enumerate(loader):
        if len(batch) == 3:
            video, joints, _meta = batch
        else:
            video, joints = batch

        video = video.to(device)   # (B,T,3,H,W)
        joints = joints.to(device) # (B,T,17,3)

        optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.startswith("cuda"))):
            _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)
            # joints_pred: (B,T,J,3)
            loss_recon = (joints_pred - joints).pow(2).mean()
            loss = loss_recon
            if lambda_vel > 0:
                loss = loss + lambda_vel * velocity_loss(joints_pred, joints)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        running_loss += float(loss.item())
        running_mpjpe += mpjpe_mm(joints_pred.detach(), joints.detach())
        n_batches += 1

        if log_every > 0 and (it + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"  iter {it+1:05d}/{len(loader):05d} | "
                f"loss {running_loss/n_batches:.6f} | mpjpe {running_mpjpe/n_batches:.3f} | "
                f"time {dt:.1f}s"
            )

    return running_loss / max(n_batches, 1), running_mpjpe / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, lambda_vel: float = 0.0):
    model.eval()

    total_loss = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 3:
            video, joints, _meta = batch
        else:
            video, joints = batch

        video = video.to(device)
        joints = joints.to(device)

        _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)

        loss_recon = (joints_pred - joints).pow(2).mean()
        loss = loss_recon
        if lambda_vel > 0:
            loss = loss + lambda_vel * velocity_loss(joints_pred, joints)

        total_loss += float(loss.item())
        total_mpjpe += mpjpe_mm(joints_pred, joints)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mpjpe / max(n_batches, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Phase-1 training: freeze ResNet, train f_movie + f_3D (3D joints)")
    parser.add_argument("--root", type=str, default=H36M_ROOT)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--stride", type=int, default=5, help="clip sampling stride in frames (dataset indexing)")
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--max-val-clips", type=int, default=None)
    parser.add_argument("--lambda-vel", type=float, default=0.0, help="optional velocity loss weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./runs/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    seed_all(args.seed)
    device = DEVICE
    os.makedirs(args.outdir, exist_ok=True)

    # Datasets / Loaders
    train_set = Human36MPreprocessedClips(
        root=args.root,
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
        return_meta=True,
        max_clips=args.max_train_clips,
    )
    val_set = Human36MPreprocessedClips(
        root=args.root,
        split="val",
        seq_len=args.seq_len,
        stride=max(1, args.stride),
        return_meta=True,
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

    # Model
    model = PHDFor3DJoints(latent_dim=2048, joints_num=JOINTS_NUM, freeze_backbone=True).to(device)

    # Phase-1: freeze f_AR explicitly (paper trains f_movie + f_3D first)
    for p in model.f_AR.parameters():
        p.requires_grad = False

    # Optimizer on trainable params only
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

    # Train
    print("===== Phase-1 training =====")
    print(f"Device: {device}")
    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")
    print(f"Seq len: {args.seq_len} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"Lambda vel: {args.lambda_vel}")
    print("============================")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        tr_loss, tr_mpjpe = train_one_epoch(
            model, train_loader, optim, scaler, device,
            lambda_vel=args.lambda_vel,
            log_every=args.log_every,
        )
        va_loss, va_mpjpe = evaluate(model, val_loader, device, lambda_vel=args.lambda_vel)

        print(f"Train: loss={tr_loss:.6f} | mpjpe={tr_mpjpe:.3f}")
        print(f"Val:   loss={va_loss:.6f} | mpjpe={va_mpjpe:.3f}")

        # Save last
        save_checkpoint(
            os.path.join(args.outdir, "last.pt"),
            model, optim, epoch, best_val, args
        )

        # Save best (by val mpjpe)
        if va_mpjpe < best_val:
            best_val = va_mpjpe
            save_checkpoint(
                os.path.join(args.outdir, "best.pt"),
                model, optim, epoch, best_val, args
            )
            print(f"  ✅ New best val MPJPE: {best_val:.3f} (saved best.pt)")

    print("\nDone.")
    print(f"Best val MPJPE: {best_val:.3f}")


if __name__ == "__main__":
    main()
