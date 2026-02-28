import os
import time
import argparse
from collections import defaultdict

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
from dataset_features import Human36MFeatureClips
from model import PHDFor3DJoints as PHD

# -----------------------------
# Metrics / Losses
# -----------------------------

H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

# Pre-build edge index tensors once at module load time (moved to device inside each loss)
_EDGE_SRC = torch.tensor([e[0] for e in H36M_EDGES], dtype=torch.long)  # (E,)
_EDGE_DST = torch.tensor([e[1] for e in H36M_EDGES], dtype=torch.long)  # (E,)

# Mean Per Joint Position Error (same units as your joints3d; often mm in H36M pipelines)
# pred, gt: (B,T,J,3)
@torch.no_grad()
def mpjpe_m(pred: torch.Tensor, gt: torch.Tensor) -> float:
    err = torch.norm(pred - gt, dim=-1)  # (B,T,J)
    return float(err.mean().item())

# Bone length loss: encourage predicted bone lengths to match GT at every timestep.
# pred, gt: (B,T,J,3)
def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    src = _EDGE_SRC.to(pred.device)
    dst = _EDGE_DST.to(pred.device)
    pred_bones = pred[:, :, dst] - pred[:, :, src]   # (B,T,E,3)
    gt_bones   = gt[:, :, dst]   - gt[:, :, src]     # (B,T,E,3)
    pred_len = torch.norm(pred_bones, dim=-1)          # (B,T,E)
    gt_len   = torch.norm(gt_bones,   dim=-1)          # (B,T,E)
    return F.mse_loss(pred_len, gt_len)

def velocity_loss(pred, gt):
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel   = gt[:, 1:] - gt[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)



def save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer,
                    epoch: int, best_val: float, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # handle nn.DataParallel case for saving only the underlying model state_dict
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model": model_state,
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
def project_with_K_torch(P_cam, K, eps=1e-6):
    """
    P_cam: (B,T,J,3)  or (...,3)
    K:     (3,3) or (B,3,3) or (B,T,3,3)
    returns: (B,T,J,2) or (...,2)
    """
    # Ensure K is broadcastable to P_cam (...,3,1)
    # P_col has shape (...,3,1)
    P_col = P_cam.unsqueeze(-1)

    if K.dim() == 2:
        # (3,3) -> (1,1,1,3,3) (works for B,T,J)
        while K.dim() < P_col.dim():
            K = K.unsqueeze(0)
    elif K.dim() == 3:
        # (B,3,3) -> (B,1,1,3,3)
        K = K[:, None, None, :, :]
    elif K.dim() == 4:
        # (B,T,3,3) -> (B,T,1,3,3)
        K = K[:, :, None, :, :]
    else:
        raise ValueError(f"Unexpected K shape: {tuple(K.shape)}")

    P_h = torch.matmul(K, P_col).squeeze(-1)  # (...,3)
    z = P_h[..., 2:3].clamp(min=eps)
    uv = P_h[..., 0:2] / z
    return uv


# ============================================================
# OPTIMISER HELPER  (call this instead of optim.AdamW(model.parameters()))
# ============================================================
# [REGULARISATION] applying L2 weight decay to GroupNorm/LayerNorm scale parameters and bias
# terms is harmful — those are not weight matrices and shouldn't be penalised. This helper
# splits parameters into two groups and sets weight_decay=0 for norms and biases.
# use this instead of: AdamW(model.parameters(), weight_decay=...)
def build_optimizer(model, lr=1e-4, weight_decay=1e-3):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_norm = any(nd in name for nd in ("gn1.", "gn2.", "norm", "LayerNorm"))
        is_bias = name.endswith(".bias")
        if is_norm or is_bias:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )

def train(model, loader, optim, scheduler, scaler, device, lambda_vel: float = 1, lambda_bone: float = 1, log_every: int = 500):
    model.train()
    epoch_start = time.time()

    running_loss = 0.0
    running_l3d = 0.0
    running_lvel = 0.0
    running_lbone = 0.0
    running_mpjpe = 0.0
    n_batches = 0

    # timers for each thing (data / forward / backward / total)
    timers = defaultdict(float)

    # this measures the time between iterations (i.e., DataLoader + host work)
    end_data = time.time()

    for it, batch in enumerate(loader):
        t_iter_start = time.time()

        # --------------------
        # Data loading time
        # --------------------
        timers["data"] += (t_iter_start - end_data)

        # NEW: dataset returns (feats, joints3d, joints2d, K)
        feats, joints3d, joints2d, K = batch

        feats = feats.to(device, non_blocking=True)         # (B,T,2048)
        joints3d = joints3d.to(device, non_blocking=True)   # (B,T,J,3)
        joints2d = joints2d.to(device, non_blocking=True)   # (B,T,J,2) 
        K = K.to(device, non_blocking=True)              # (3,3) or (B,3,3) or (B,T,3,3)

        optim.zero_grad(set_to_none=True)

        # --------------------
        # Forward + Loss time
        # --------------------
        t_fwd = time.time()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            # IMPORTANT: I precomputed ResNet features, so we must NOT pass video into the model
            # We use a dedicated path that assumes feats = ResNet output (2048D)
            _phi, _phi_hat, joints_pred, _joints_hat = model.forward(feats, predict_future=False)
            # joints_pred: (B,T,J,3) assumed camera coordinates that match K

            # 3D loss
            l3d = (joints_pred - joints3d).pow(2).mean()
            # Bone length loss: encourage consistent limb lengths across time
            lbone = bone_length_loss(joints_pred, joints3d)
            # Velocity loss: encourage consistent motion dynamics
            lvel = velocity_loss(joints_pred, joints3d)
            loss = l3d + 0.5 * lbone + 1.0 * lvel

        timers["forward+loss"] += (time.time() - t_fwd)

        # --------------------
        # Backward + Optim time
        # --------------------
        t_bwd = time.time()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        scheduler.step()

        timers["backward"] += (time.time() - t_bwd)

        # --------------------
        # Logging / metrics
        # --------------------
        running_loss += float(loss.item())
        running_l3d += float(l3d.item())
        running_lvel += float(lvel.item())
        running_lbone += float(lbone.item())
        running_mpjpe += mpjpe_m(joints_pred.detach(), joints3d.detach())
        n_batches += 1

        t_iter_end = time.time()
        timers["iter"] += (t_iter_end - t_iter_start)
        end_data = t_iter_end

        if log_every > 0 and (it + 1) % log_every == 0:
            dt_epoch = time.time() - epoch_start
            print(
                f"[3D + bone + velocity]  iter {it+1:05d}/{len(loader):05d} | "
                f"loss {running_loss/n_batches:.6f} (3d {running_l3d/n_batches:.6f}, vel {running_lvel/n_batches:.6f}, bone {running_lbone/n_batches:.6f}) | "
                f"mpjpe {running_mpjpe/n_batches:.3f} | "
                f"time/iter {timers['iter']/n_batches:.4f}s | "
                f"epoch {dt_epoch:.1f}s"
            )

    # print time to each thing (per epoch)
    epoch_time = time.time() - epoch_start
    print("\n[Train timing]")
    print(f"  data loading:    {timers['data']:.2f}s")
    print(f"  forward+loss:    {timers['forward+loss']:.2f}s")
    print(f"  backward+optim:  {timers['backward']:.2f}s")
    print(f"  total iter time: {timers['iter']:.2f}s")
    print(f"  total epoch:     {epoch_time:.2f}s")
    print(f"  avg iter time:   {timers['iter']/max(n_batches,1):.4f}s\n")

    return running_loss / max(n_batches, 1), running_mpjpe / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, lambda_vel: float = 1.0, lambda_bone: float = 1.0, test_set: bool = False):
    model.eval()

    total_loss = 0.0
    total_l3d = 0.0
    total_lbone = 0.0
    total_lvel = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    # Disable 2D loss during warmup
    # use_2d_loss = (epoch >= warmup_epochs)
    # training only with 3d loss

    # timing for evaluation
    t_eval_start = time.time()
    timers = defaultdict(float)
    end_data = time.time()

    for batch in loader:
        t_iter_start = time.time()
        timers["data"] += (t_iter_start - end_data)
        if test_set:
            feats, joints3d, joints2d, K, meta = batch
        else:
            feats, joints3d, joints2d, K = batch

        feats = feats.to(device, non_blocking=True)         # (B,T,2048)
        joints3d = joints3d.to(device, non_blocking=True)   # (B,T,J,3)
        joints2d = joints2d.to(device, non_blocking=True)   # (B,T,J,2)
        K = K.to(device, non_blocking=True)            # (B,3,3) 

        t_fwd = time.time()
        _phi, _phi_hat, joints_pred, _joints_hat = model.forward(feats, predict_future=False)
        timers["forward"] += (time.time() - t_fwd)

        l3d = (joints_pred - joints3d).pow(2).mean()
        lbone = bone_length_loss(joints_pred, joints3d)
        lvel = velocity_loss(joints_pred, joints3d)
        loss = l3d + 0.5 * lbone + 1.0 * lvel

        total_loss += float(loss.item())
        total_l3d += float(l3d.item())
        total_lbone += float(lbone.item())
        total_lvel += float(lvel.item())
        total_mpjpe += mpjpe_m(joints_pred, joints3d)
        n_batches += 1

        t_iter_end = time.time()
        timers["iter"] += (t_iter_end - t_iter_start)
        end_data = t_iter_end

    eval_time = time.time() - t_eval_start
    print("[Val timing]")
    print(f"  data loading:  {timers['data']:.2f}s")
    print(f"  forward:       {timers['forward']:.2f}s")
    print(f"  total:         {eval_time:.2f}s")
    print(f"  avg iter time: {timers['iter']/max(n_batches,1):.4f}s\n")

    # (optionally you can also print l3d/l2d outside)
    return (
        total_loss / max(n_batches, 1),
        total_mpjpe / max(n_batches, 1),
        total_l3d / max(n_batches, 1),
        0.0,  # total_l2d / max(n_batches, 1), since we're not using 2D loss in this version
    )


def main():
    parser = argparse.ArgumentParser("Phase-1 training: freeze ResNet, train f_movie + f_3D (3D joints + 2D reprojection)")
    parser.add_argument("--root", type=str, default=H36M_ROOT)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lambda-2d", type=float, default=1e-6, help="2D reprojection loss weight")
    parser.add_argument("--outdir", type=str, default="./runs/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=10,
                    help="Stop if val MPJPE doesn't improve for this many epochs (0 disables).")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="Minimum MPJPE improvement to reset patience.")
    args = parser.parse_args()

    # set device and multi-GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        device = torch.device("cpu")
        num_gpus = 0
        gpu_ids = []
    
    use_multi_gpu = num_gpus > 1 

    os.makedirs(args.outdir, exist_ok=True)

    effective_batch_size = args.batch_size
    if use_multi_gpu:
        # each GPU gets batch_size / num_gpus samples
        per_gpu_batch_size = args.batch_size // num_gpus
        effective_batch_size = per_gpu_batch_size * num_gpus
        print(f"Multi-GPU training: {num_gpus} GPUs (IDs: {gpu_ids})")
        print(f"Effective batch size: {effective_batch_size} ({per_gpu_batch_size} per GPU)")

    # load data
    # load data
    train_set = Human36MFeatureClips(
        root=args.root,
        subjects=[1, 6, 7, 8],
    )
    val_set = Human36MFeatureClips(
        root=args.root,
        subjects=[5],
    )

    # DataLoader optimizations
    # loader_kwargs = {
    #     'pin_memory': True if torch.cuda.is_available() else False,
    #     'prefetch_factor': args.prefetch_factor if args.num_workers > 0 else None,
    #     'persistent_workers': args.persistent_workers if args.num_workers > 0 else False,
    # }

    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    model = PHD(joints_num=JOINTS_NUM)

    # ----------------------------------
    # TRAINING PHASE 1 : freeze ResNet, train f_movie + f_3D
    # ----------------------------------
    for p in model.f_AR.parameters():
        p.requires_grad = False

    # move to device before wrapping in DataParallel (if using)
    model = model.to(device)
    
    # multi-GPU with nn.DataParallel
    if use_multi_gpu:
        print(f"Wrapping model in DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found. Did you accidentally freeze everything?")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    optim = build_optimizer(raw_model, lr=args.lr, weight_decay=5e-3)

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=args.lr,
    steps_per_epoch=len(train_loader),
    epochs=args.epochs,
    pct_start=0.1,          # 10% warmup
    anneal_strategy='cos',
    )

    start_epoch = 0
    best_val = float("inf")
    no_improve_epochs = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        # handle nn.DataParallel case for loading state_dict
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model"], strict=True)
        else:
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
        t_epoch = time.time()

        tr_loss, tr_mpjpe = train(
            model, train_loader, optim, scheduler, scaler, device,
            log_every=args.log_every
        )
        va_loss, va_mpjpe, va_l3d, va_l2d = evaluate(
            model, val_loader, device
        )


        print(f"Train: loss={tr_loss:.6f} | mpjpe={tr_mpjpe:.3f}")
        print(f"Val:   loss={va_loss:.6f} (3d {va_l3d:.6f} + {args.lambda_2d:.3g}*2d {va_l2d:.6f}) | mpjpe={va_mpjpe:.3f}")
        print(f"Epoch time: {time.time() - t_epoch:.2f}s")

        save_checkpoint(
            os.path.join(args.outdir, "last.pt"),
            model, optim, epoch, best_val, args
        )

        improved = (best_val - va_mpjpe) > args.early_stop_min_delta

        if improved:
            best_val = va_mpjpe
            no_improve_epochs = 0
            save_checkpoint(
                os.path.join(args.outdir, "best.pt"),
                model, optim, epoch, best_val, args
            )
            print(f"New best val MPJPE: {best_val:.3f} (saved best.pt)")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{args.early_stop_patience} epochs "
                f"(best {best_val:.3f}, current {va_mpjpe:.3f})")

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val MPJPE: {best_val:.3f}")
            break

    print("\nDone.")
    print(f"Best val MPJPE: {best_val:.3f}")


if __name__ == "__main__":
    main()