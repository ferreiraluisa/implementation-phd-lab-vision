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


@torch.no_grad()
# -----------------------------
# Metrics / Losses
# -----------------------------

# Mean Per Joint Position Error in mm
# pred, gt: (Batch Size, T fixed length, Joints Num, 3)
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


def train(model, loader, optim, scaler, device, log_every: int = 100):
    model.train()
    t = time.time()

    running_loss = 0.0
    running_mpjpe = 0.0
    n_batches = 0

    for it, batch in enumerate(loader):
        video, joints = batch

        video = video.to(device)   # (B,T,3,H,W)
        joints = joints.to(device) # (B,T,17,3)

        # zero grad before forward
        optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.startswith("cuda"))):
            _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)
            # joints_pred: (B,T,J,3)
            loss = (joints_pred - joints).pow(2).mean() # mean squared error

        # backward + optimize
        # scaler is to help with mixed precision training, meaning float16 + float32
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
            dt = time.time() - t
            print(
                f"  iter {it+1:05d}/{len(loader):05d} | "
                f"loss {running_loss/n_batches:.6f} | mpjpe {running_mpjpe/n_batches:.3f} | "
                f"time {dt:.1f}s"
            )

    return running_loss / max(n_batches, 1), running_mpjpe / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    for batch in loader:
        video, joints = batch

        video = video.to(device)
        joints = joints.to(device)

        _phi, _phi_hat, joints_pred, _joints_hat = model(video, predict_future=False)

        loss = (joints_pred - joints).pow(2).mean()

        total_loss += float(loss.item())
        total_mpjpe += mpjpe_mm(joints_pred, joints)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mpjpe / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser("Phase-1 training: freeze ResNet, train f_movie + f_3D (3D joints)")
    parser.add_argument("--root", type=str, default=H36M_ROOT)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5, help="clip sampling stride in frames (dataset indexing)")
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--max-val-clips", type=int, default=None)
    parser.add_argument("--lambda-vel", type=float, default=0.0, help="optional velocity loss weight")
    parser.add_argument("--outdir", type=str, default="./runs/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = DEVICE
    os.makedirs(args.outdir, exist_ok=True)

    # load data
    train_set = Human36MPreprocessedClips(
        root=args.root,
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
        max_clips=args.max_train_clips,
    )
    val_set = Human36MPreprocessedClips(
        root=args.root,
        split="val",
        seq_len=args.seq_len,
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


    model = PHDFor3DJoints(latent_dim=2048, joints_num=JOINTS_NUM, freeze_backbone=True).to(device)

    # ----------------------------------
    # TRAINING PHASE 1 : freeze ResNet, train f_movie + f_3D
    # ----------------------------------
    # froze the weights of the pre-trained ResNet from [26] and trained the temporal encoder fmovie and the 3D regressor f3D jointly on the task of estimating 3D human meshes from video
    for p in model.f_AR.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found. Did you accidentally freeze everything?")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    start_epoch = 0
    best_val = float("inf")

    # optionally resume from a checkpoint if needed(in case of time limit on bender)
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
    print(f"Lambda vel: {args.lambda_vel}")
    print("============================")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        tr_loss, tr_mpjpe = train(model, train_loader, optim, scaler, device, log_every=args.log_every)
        va_loss, va_mpjpe = evaluate(model, val_loader, device)

        print(f"Train: loss={tr_loss:.6f} | mpjpe={tr_mpjpe:.3f}")
        print(f"Val:   loss={va_loss:.6f} | mpjpe={va_mpjpe:.3f}")

        # save last
        save_checkpoint(
            os.path.join(args.outdir, "last.pt"),
            model, optim, epoch, best_val, args
        )

        # save best by val MPJPE
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
