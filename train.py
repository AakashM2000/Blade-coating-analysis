# SPDX-License-Identifier: MIT
"""
Train a UNet to segment background / substrate / coating / tool in blade-coating frames.

Structure follows image_segmentation/segmentation_training_tracked.py (CE + Dice loss, AMP,
gradient clipping, ReduceLROnPlateau, MLflow tracking, IoU + confusion matrix), with changes for
this use case:
  * train/val split per VIDEO (see dataset.split_by_video),
  * random crops for training, full frames for validation,
  * optional class weights for the CE loss (the tool and a thin wet film are small classes),
  * model selection on validation mIoU, and checkpoints that store class names / channels so
    inference does not need to repeat them,
  * --resume to continue an interrupted run,
  * MLflow is optional (falls back to console logging if it is not installed).

Usage (from the repo root):
    python -m blade_coating_segmentation.train ^
        --data "datasets/cvat_exports/task_1" "datasets/cvat_exports/task_2" ^
        --epochs 80 --batch-size 4 --crop 512 512 --lr 2e-4 --class-weights auto
"""
import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from blade_coating_segmentation import CLASS_NAMES, CLASS_PALETTE
from blade_coating_segmentation.dataset import (
    BladeCoatingDataset, SubsetWithTransform, split_by_video, get_train_transforms, get_val_transforms,
)
from image_segmentation.utils.UNet import UNet, UNetS, UNetXS
from image_segmentation.utils.evaluate import DiceLoss
import image_segmentation.utils.tracking_utils as tracking

ARCHITECTURES = {"UNet": UNet, "UNetS": UNetS, "UNetXS": UNetXS}


# ---------------------------------------------------------
# Optional MLflow
# ---------------------------------------------------------

class _ConsoleTracker:
    """Minimal stand-in with the MLflow calls used below."""
    def set_experiment(self, *a, **k): pass
    def start_run(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()
    def log_params(self, *a, **k): pass
    def set_tags(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass


def get_tracker(use_mlflow: bool):
    if use_mlflow:
        try:
            import mlflow
            return mlflow
        except ImportError:
            print("mlflow not installed -> logging to console only (pip install mlflow)")
    return _ConsoleTracker()


# ---------------------------------------------------------
# Loss / train / validate
# ---------------------------------------------------------

def compute_losses(logits, targets, ce_loss_fn, dice_loss_fn, ce_weight=1.0, dice_weight=1.0):
    ce_loss = ce_loss_fn(logits, targets)
    dice_loss = dice_loss_fn(logits, targets)
    return ce_weight * ce_loss + dice_weight * dice_loss, ce_loss, dice_loss


def compute_class_weights(freqs: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    """Median-frequency balancing, clipped so very rare classes do not dominate."""
    freqs = np.where(freqs > 0, freqs, np.nan)
    w = np.nanmedian(freqs) / freqs
    w = np.nan_to_num(w, nan=1.0)
    return torch.tensor(np.clip(w, 0.1, max_weight), dtype=torch.float32)


def train_one_epoch(model, loader, device, optimizer, scaler, ce_loss_fn, dice_loss_fn, amp, tracker, global_step):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc="Training", unit="batch")
    for batch in pbar:
        images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss, ce_loss, dice_loss = compute_losses(logits, targets, ce_loss_fn, dice_loss_fn)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        if global_step % 10 == 0:
            tracker.log_metrics({"train_loss_iter": loss.item(), "train_ce_loss_iter": ce_loss.item(),
                                 "train_dice_loss_iter": dice_loss.item()}, step=global_step)
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return epoch_loss / max(1, len(loader)), global_step


@torch.no_grad()
def validate(model, loader, device, ce_loss_fn, dice_loss_fn, n_classes, ignore_index, amp):
    model.eval()
    val_loss = val_dice = 0.0
    running_cm = torch.zeros((n_classes, n_classes), dtype=torch.int64, device=device)
    for batch in tqdm(loader, desc="Validation", unit="batch", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        targets = batch["mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
        logits = logits.float()
        _, ce_loss, dice_loss = compute_losses(logits, targets, ce_loss_fn, dice_loss_fn)
        val_loss += ce_loss.item()
        val_dice += 1.0 - dice_loss.item()
        running_cm += tracking.compute_confusion_matrix(logits.argmax(dim=1), targets,
                                                        num_classes=n_classes, ignore_index=ignore_index)
    n = max(1, len(loader))
    per_class_iou = tracking.per_class_iou_from_cm(running_cm)
    return val_loss / n, val_dice / n, per_class_iou, float(np.nanmean(per_class_iou)), running_cm


@torch.no_grad()
def log_examples(model, loader, device, epoch, out_dir: Path, tracker, n_samples=3):
    model.eval()
    logged = 0
    for batch in loader:
        imgs = batch["image"].to(device, dtype=torch.float32)
        preds = model(imgs).argmax(dim=1)
        for b in range(imgs.size(0)):
            img = imgs[b].cpu()
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            overlay = tracking.overlay_prediction(img, preds[b].cpu(), batch["mask"][b], CLASS_PALETTE)
            path = out_dir / f"overlay_e{epoch + 1:03d}_s{logged}.png"
            plt.imsave(path, overlay)
            tracker.log_artifact(str(path), artifact_path="overlays")
            logged += 1
            if logged >= n_samples:
                return


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def train_model(
    data_dirs,
    arch: str = "UNetS",
    n_channels: int = 3,
    epochs: int = 80,
    batch_size: int = 4,
    crop_size=(512, 512),
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    val_percent: float = 0.15,
    ignore_index: int = 255,
    class_weights: str = "auto",
    amp: bool = True,
    num_workers: Optional[int] = None,
    out_dir: str = "artifacts/blade_coating",
    init_weights: Optional[str] = None,
    resume: Optional[str] = None,
    use_mlflow: bool = True,
    experiment_name: str = "BladeCoating-Segmentation",
    run_name: Optional[str] = None,
    log_images_every_n_epochs: int = 5,
    early_stop_patience: int = 15,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    full = BladeCoatingDataset(data_dirs, ignore_index=ignore_index, grayscale=(n_channels == 1))
    train_sub, val_sub = split_by_video(full, val_percent=val_percent, seed=seed)
    train_set = SubsetWithTransform(train_sub, get_train_transforms(tuple(crop_size), ignore_index))
    val_set = SubsetWithTransform(val_sub, get_val_transforms())

    if num_workers is None:
        num_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    loader_args = dict(num_workers=num_workers, pin_memory=device.type == "cuda",
                       persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **loader_args)
    # Full frames can differ in size between setups -> batch_size=1 for validation
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_args)

    # ---- Model ----
    model = ARCHITECTURES[arch](n_channels=n_channels, n_classes=len(CLASS_NAMES))
    if init_weights:
        # e.g. start from the spin-coater weights: matching layers are reused, the rest are new
        state = torch.load(init_weights, map_location="cpu")
        state = state.get("model", state)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        print(f"Initialised {len(compatible)}/{len(own)} tensors from {init_weights}")
    model.to(device)

    # ---- Loss ----
    weight = None
    if class_weights == "auto":
        freqs = full.class_pixel_frequencies(indices=train_sub.indices)
        weight = compute_class_weights(freqs)
        print("Class pixel frequencies:", dict(zip(CLASS_NAMES, np.round(freqs, 4))))
        print("CE class weights:       ", dict(zip(CLASS_NAMES, np.round(weight.numpy(), 3))))
        weight = weight.to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    dice_loss_fn = DiceLoss(multiclass=True, ignore_index=ignore_index)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    start_epoch, best_miou, global_step = 0, -1.0, 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_miou, global_step = ckpt["epoch"] + 1, ckpt["best_miou"], ckpt["global_step"]
        print(f"Resumed from {resume} at epoch {start_epoch}")

    meta = {"arch": arch, "n_channels": n_channels, "class_names": CLASS_NAMES, "class_palette": CLASS_PALETTE}

    tracker = get_tracker(use_mlflow)
    tracker.set_experiment(experiment_name)
    with tracker.start_run(run_name=run_name):
        tracker.log_params({
            "arch": arch, "n_channels": n_channels, "epochs": epochs, "batch_size": batch_size,
            "crop_size": f"{crop_size[0]}x{crop_size[1]}", "learning_rate": learning_rate,
            "weight_decay": weight_decay, "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau",
            "amp": amp, "class_weights": class_weights, "n_train": len(train_set), "n_val": len(val_set),
            "data_dirs": ";".join(map(str, data_dirs)), "init_weights": init_weights or "",
        })
        tracker.set_tags({
            "device": str(device), "torch_version": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        })

        epochs_without_improvement = 0
        for epoch in range(start_epoch, epochs):
            train_loss, global_step = train_one_epoch(model, train_loader, device, optimizer, scaler,
                                                      ce_loss_fn, dice_loss_fn, amp, tracker, global_step)
            val_loss, val_dice, per_class_iou, miou, cm = validate(model, val_loader, device, ce_loss_fn,
                                                                  dice_loss_fn, len(CLASS_NAMES), ignore_index, amp)
            scheduler.step(0.0 if math.isnan(miou) else miou)

            metrics = {"train_loss_epoch": train_loss, "val_loss_epoch": val_loss, "val_dice": val_dice,
                       "val_mIoU": miou, "lr": optimizer.param_groups[0]["lr"]}
            metrics.update({f"val_IoU_{n}": float(v) for n, v in zip(CLASS_NAMES, per_class_iou) if not np.isnan(v)})
            tracker.log_metrics(metrics, step=epoch)

            iou_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, per_class_iou))
            print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_dice={val_dice:.4f} val_mIoU={miou:.4f} | {iou_str}")

            if (epoch + 1) % log_images_every_n_epochs == 0:
                log_examples(model, val_loader, device, epoch, out_dir, tracker)
                cm_np = cm.cpu().numpy().astype(np.int64)
                cm_path = out_dir / f"confusion_matrix_e{epoch + 1:03d}.png"
                tracking.plot_confusion_matrix(cm_np, CLASS_NAMES, str(cm_path))
                tracker.log_artifact(str(cm_path), artifact_path="diagnostics")

            if not math.isnan(miou) and miou > best_miou:
                best_miou = miou
                epochs_without_improvement = 0
                best_path = out_dir / f"best_{arch}_bladecoating.pt"
                torch.save({"model": model.state_dict(), **meta, "epoch": epoch, "val_mIoU": miou}, best_path)
                tracker.log_artifact(str(best_path), artifact_path="checkpoints")
                print(f"  -> Saved new best model (mIoU={miou:.4f}): {best_path}")
            else:
                epochs_without_improvement += 1

            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best_miou": best_miou, "global_step": global_step, **meta},
                       out_dir / "last.pt")

            if epochs_without_improvement >= early_stop_patience:
                print(f"No mIoU improvement for {early_stop_patience} epochs -> early stop.")
                break

    with open(out_dir / "training_summary.json", "w") as f:
        json.dump({"best_val_mIoU": best_miou, **meta}, f, indent=2)
    print("Training complete. Best val mIoU:", best_miou)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", help="One or more CVAT 'Segmentation mask 1.1' export folders")
    p.add_argument("--arch", choices=list(ARCHITECTURES), default="UNetS")
    p.add_argument("--channels", type=int, choices=[1, 3], default=3, help="3 = RGB (recommended: film colour helps), 1 = grayscale")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-percent", type=float, default=0.15)
    p.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="artifacts/blade_coating")
    p.add_argument("--init-weights", default=None, help="Optional checkpoint to warm-start from")
    p.add_argument("--resume", default=None, help="Path to last.pt to continue training")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--experiment", default="BladeCoating-Segmentation")
    p.add_argument("--run-name", default=None)
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument("--summary", action="store_true", help="Print a torchinfo model summary and exit")
    a = p.parse_args()

    if a.summary:
        import sys
        from torchinfo import summary
        sys.stdout.reconfigure(encoding="utf-8")  # torchinfo prints box characters the Windows console can't encode
        summary(ARCHITECTURES[a.arch](a.channels, len(CLASS_NAMES)), input_size=(a.batch_size, a.channels, *a.crop))
        return
    if not a.data:
        p.error("--data is required for training")

    train_model(a.data, arch=a.arch, n_channels=a.channels, epochs=a.epochs, batch_size=a.batch_size,
                crop_size=a.crop, learning_rate=a.lr, weight_decay=a.weight_decay, val_percent=a.val_percent,
                class_weights=a.class_weights, amp=not a.no_amp, num_workers=a.workers, out_dir=a.out,
                init_weights=a.init_weights, resume=a.resume, use_mlflow=not a.no_mlflow,
                experiment_name=a.experiment, run_name=a.run_name, early_stop_patience=a.early_stop)


if __name__ == "__main__":
    main()
