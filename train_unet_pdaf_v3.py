"""
PDAF Training Script v3 — DPE dual-mode training enabled.

Changes from v2:
1. DPE dual_mode: both conditional denoising AND pure Gaussian generation
2. sc_loss weight reduced to 0.1 (was 0.3, too high for fine-grained segmentation)
3. dcm_modulation_scale increased to 0.5 (was 0.1, too weak)
4. teacher epochs increased to 25 (was 15)
5. kl_loss weight increased to 0.1 (was 0.05)
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, GaussianBlur

# Direct import to avoid visdom dependency in models/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location('unet_pdaf', str(Path(__file__).parent / 'models/unet_pdaf.py'))
unet_pdaf_module = importlib.util.module_from_spec(spec)
sys.modules['models.unet_pdaf'] = unet_pdaf_module
spec.loader.exec_module(unet_pdaf_module)
PDAFUNet = unet_pdaf_module.PDAFUNet
UNetBackbone = unet_pdaf_module.UNetBackbone


# ---------------------------------------------------------------------------
# Normalization — per-image spatial mean/std (matching baseline U-Net).
# ---------------------------------------------------------------------------


def normalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
    """Per-image spatial mean/std normalization. Input: (B, C, H, W)."""
    mean = images.mean(dim=(2, 3), keepdim=True)
    std = images.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (images - mean) / std


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return (mask > 127).astype(np.float32)


def _drive_id(stem: str) -> str:
    return stem.split("_")[0]


def discover_pairs(root: Path, split: str) -> list[tuple[Path, Path, Path | None]]:
    input_dir = root / split / "input"
    label_dir = root / split / "label"
    if not input_dir.exists():
        input_dir = root / "images" / split
    if not label_dir.exists():
        label_dir = root / "labels" / split

    if not input_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split directories under {root}")

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    pairs: list[tuple[Path, Path, Path | None]] = []
    for image_path in image_paths:
        stem = image_path.stem
        candidates = [
            label_dir / f"{stem}.png",
            label_dir / f"{stem}_manual1.png",
            label_dir / f"{stem}_1stHO.png",
        ]
        label_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if label_path is None:
            continue

        fov = None
        for suffix in [f"{_drive_id(stem)}_training_mask.gif", f"{stem}_training_mask.gif"]:
            candidate = root / "mask" / suffix
            if candidate.exists():
                fov = candidate
                break
        pairs.append((image_path, label_path, fov))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {root}/{split}")
    return pairs


class TrainPatchDataset(Dataset):
    def __init__(self, root: Path, patch_size: int, samples_per_epoch: int) -> None:
        self.pairs = discover_pairs(root, "train")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path, _ = random.choice(self.pairs)
        image = read_rgb(image_path)
        mask = read_binary_mask(mask_path)

        h, w = mask.shape
        patch = self.patch_size
        if h < patch or w < patch:
            pad_h = max(0, patch - h)
            pad_w = max(0, patch - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = mask.shape

        top = random.randint(0, h - patch)
        left = random.randint(0, w - patch)
        image = image[top : top + patch, left : left + patch]
        mask = mask[top : top + patch, left : left + patch]

        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return (
            torch.from_numpy(image.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask[None]).float(),
        )


class EvalDataset(Dataset):
    def __init__(self, root: Path, split: str = "val") -> None:
        self.root = root
        self.pairs = discover_pairs(root, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_path, mask_path, fov_path = self.pairs[index]
        image = read_rgb(image_path)
        mask = read_binary_mask(mask_path)

        if fov_path is not None:
            fov = read_binary_mask(fov_path)
        else:
            fov = (image.mean(axis=2) > 0.03).astype(np.float32)

        return (
            torch.from_numpy(image.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask[None]).float(),
            torch.from_numpy(fov[None]).float(),
            image_path.stem,
        )


class PhotometricAugmentor(nn.Module):
    """Build pseudo-target images with label-preserving photometric changes."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.color_jitter = ColorJitter(
            brightness=cfg["brightness"],
            contrast=cfg["contrast"],
            saturation=cfg["saturation"],
            hue=cfg["hue"],
        )
        self.blur = GaussianBlur(kernel_size=cfg["blur_kernel"], sigma=cfg["blur_sigma"])
        self.noise_std = cfg["noise_std"]
        self.gamma_range = cfg["gamma_range"]
        self.blur_prob = cfg["blur_prob"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for image in images:
            aug = image.clamp(0.0, 1.0)
            aug = self.color_jitter(aug)
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            aug = aug.clamp(0.0, 1.0).pow(gamma)
            if random.random() < self.blur_prob:
                aug = self.blur(aug)
            if self.noise_std > 0:
                aug = aug + torch.randn_like(aug) * self.noise_std
            outputs.append(aug.clamp(0.0, 1.0))
        return torch.stack(outputs, dim=0)


class RetinalDomainAugmentor(nn.Module):
    """Pseudo-target builder tailored for retinal domain shift."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.photometric = PhotometricAugmentor(cfg)
        self.vignette_strength = cfg.get("vignette_strength", [0.15, 0.45])
        self.channel_shift = cfg.get("channel_shift", 0.08)
        self.low_freq_bias = cfg.get("low_freq_bias", 0.12)

    def _vignette_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        rr = torch.sqrt(xx * xx + yy * yy).clamp(max=1.0)
        strength = random.uniform(self.vignette_strength[0], self.vignette_strength[1])
        return 1.0 - strength * (rr ** 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for image in images:
            aug = self.photometric(image.unsqueeze(0))[0]
            h, w = aug.shape[-2:]
            mask = self._vignette_mask(h, w, aug.device, aug.dtype)
            aug = aug * mask.unsqueeze(0)
            shift = torch.empty((3, 1, 1), device=aug.device, dtype=aug.dtype).uniform_(
                -self.channel_shift, self.channel_shift
            )
            aug = aug + shift
            pooled = F.avg_pool2d(aug.unsqueeze(0), kernel_size=31, stride=1, padding=15)[0]
            aug = aug * (1.0 - self.low_freq_bias) + pooled * self.low_freq_bias
            outputs.append(aug.clamp(0.0, 1.0))
        return torch.stack(outputs, dim=0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between LPE's variational posterior q(z) and N(0,1)."""
    logvar_clamped = logvar.clamp(min=-10.0, max=2.0)
    kl = -0.5 * torch.sum(1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1)
    return kl.mean()


def semantic_consistency_loss(
    teacher_logits: torch.Tensor,
    logits_hat: torch.Tensor,
) -> torch.Tensor:
    """Semantic consistency — only supervise the inference branch (logits_hat)."""
    teacher_prob = torch.sigmoid(teacher_logits).detach()
    return F.mse_loss(torch.sigmoid(logits_hat), teacher_prob)


def prior_constraint_loss(z_hat: torch.Tensor, z_tilde: torch.Tensor, lpe_grad: bool = True) -> torch.Tensor:
    """Prior constraint term. When lpe_grad=True, gradient flows to both DPE and LPE."""
    target = z_tilde if lpe_grad else z_tilde.detach()
    return F.smooth_l1_loss(torch.tanh(z_hat), torch.tanh(target))


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Keep the deployable DPE branch close to the stronger z_tilde branch."""
    return F.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits).detach())


def feature_alignment_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    align_layers: list[str],
) -> torch.Tensor:
    """Feature alignment loss: align student features to teacher features at intermediate layers."""
    loss = 0.0
    count = 0
    for name in align_layers:
        if name in student_features and name in teacher_features:
            s_feat = student_features[name]
            t_feat = teacher_features[name].detach()
            # Interpolate to match spatial dimensions if needed
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            loss += F.mse_loss(s_feat, t_feat)
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def compute_metrics(prob: np.ndarray, target: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    valid = fov > 0.5
    y_true = target[valid].astype(np.uint8)
    y_prob = prob[valid]
    y_pred = (y_prob >= 0.5).astype(np.uint8)

    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    tn = np.logical_and(y_pred == 0, y_true == 0).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    eps = 1e-8

    soft_dice = (2.0 * (prob * target)[valid].sum() + eps) / (
        (prob[valid] ** 2 + target[valid] ** 2).sum() + eps
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "soft_dice": float(soft_dice),
        "iou": tp / (tp + fp + fn + eps),
        "acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "sen": tp / (tp + fn + eps),
        "spe": tn / (tn + fp + eps),
        "auc": auc,
    }


def save_prob_map(path: Path, prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prob * 255).clip(0, 255).astype(np.uint8)).save(path)


@torch.no_grad()
def evaluate_teacher(model: UNetBackbone, loader: DataLoader, device: torch.device, pred_dir: Path | None = None) -> dict[str, float]:
    model.eval()
    rows = []
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for raw_image, target, fov, name in loader:
        image = normalize_image_tensor(raw_image.to(device))
        logits = model(image)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        target_np = target.numpy()[0, 0]
        fov_np = fov.numpy()[0, 0]
        rows.append(compute_metrics(prob, target_np, fov_np))
        if pred_dir is not None:
            save_prob_map(pred_dir / f"{name[0]}_prob.png", prob)
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def evaluate_pdaf(model: PDAFUNet, loader: DataLoader, device: torch.device, pred_dir: Path | None = None) -> dict[str, float]:
    model.eval()
    rows = []
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for raw_image, target, fov, name in loader:
        image = normalize_image_tensor(raw_image.to(device))
        logits = model.infer(image)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        target_np = target.numpy()[0, 0]
        fov_np = fov.numpy()[0, 0]
        rows.append(compute_metrics(prob, target_np, fov_np))
        if pred_dir is not None:
            save_prob_map(pred_dir / f"{name[0]}_prob.png", prob)
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


def save_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_teacher(
    model: UNetBackbone,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Stage 1: pretrain the segmentation teacher before PDAF starts."""
    stage_cfg = cfg["teacher"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=stage_cfg["lr"],
        betas=tuple(stage_cfg["betas"]),
        weight_decay=stage_cfg["weight_decay"],
    )
    checkpoint_path = output_dir / "teacher_best.pt"
    history: list[dict[str, float]] = []
    best_dice = -1.0

    for epoch in range(1, stage_cfg["epochs"] + 1):
        model.train()
        losses = []
        for raw_image, target in train_loader:
            image = normalize_image_tensor(raw_image.to(device))
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = segmentation_loss(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate_teacher(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        save_history(output_dir / "teacher_metrics.csv", history)
        logging.info(
            "teacher epoch=%03d loss=%.4f dice=%.4f iou=%.4f auc=%.4f",
            epoch,
            row["loss"],
            metrics["dice"],
            metrics["iou"],
            metrics["auc"],
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "config": cfg,
                },
                checkpoint_path,
            )

    return checkpoint_path


def train_pdaf(
    model: PDAFUNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Stage 2: train student + LPE + DCM + DPE with the frozen teacher.

    DPE dual-mode training:
      Mode 1 (conditional denoising): DPE(student_cond, target_prior=z_tilde) → z_hat
      Mode 2 (pure Gaussian):         DPE(student_cond, target_prior=None)    → z_hat
    Both modes predict z_tilde as the target.
    """
    stage_cfg = cfg["pdaf"]
    augmentor_name = stage_cfg.get("augmentor", "retinal")
    if augmentor_name == "retinal":
        augmentor = RetinalDomainAugmentor(cfg["augmentation"])
    else:
        augmentor = PhotometricAugmentor(cfg["augmentation"])

    dpe_dual_mode = stage_cfg.get("dpe_dual_mode", False)
    dpe_pure_gaussian_prob = stage_cfg.get("dpe_pure_gaussian_prob", 0.3)

    params = list(model.student.parameters()) + list(model.lpe.parameters()) + list(model.dcm.parameters()) + list(model.dpe.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=stage_cfg["lr"],
        betas=tuple(stage_cfg["betas"]),
        weight_decay=stage_cfg["weight_decay"],
    )
    checkpoint_path = output_dir / "pdaf_best.pt"
    history: list[dict[str, float]] = []
    best_dice = -1.0
    grad_clip = stage_cfg.get("grad_clip_norm", None)
    prior_warmup_epochs = max(1, stage_cfg.get("prior_warmup_epochs", 1))
    distill_weight_base = stage_cfg.get("lambda_distill", 0.0)
    prior_mode_lpe_grad = stage_cfg.get("prior_mode_lpe_grad", False)
    dpe_start_epoch = stage_cfg.get("dpe_start_epoch", 10)
    patience = stage_cfg.get("patience", 15)
    epochs_without_improvement = 0

    for epoch in range(1, stage_cfg["epochs"] + 1):
        model.train()
        model.freeze_teacher()
        losses = []
        task_losses = []
        sc_losses = []
        kl_losses = []
        prior_losses = []
        distill_losses = []
        gaussian_losses = []

        for raw_source, target in train_loader:
            raw_pseudo_target = augmentor(raw_source).to(device)
            raw_source = raw_source.to(device)
            target = target.to(device)

            source = normalize_image_tensor(raw_source)
            pseudo_target = normalize_image_tensor(raw_pseudo_target)

            optimizer.zero_grad(set_to_none=True)
            use_dpe = epoch >= dpe_start_epoch

            # Determine DPE mode for this batch
            use_gaussian = dpe_dual_mode and use_dpe and (random.random() < dpe_pure_gaussian_prob)

            outputs = model.forward_train(source, pseudo_target, use_dpe=use_dpe, use_gaussian=use_gaussian)

            # Task loss — supervise both branches.
            if use_dpe:
                task = 0.5 * (
                    segmentation_loss(outputs["logits_tilde"], target) +
                    segmentation_loss(outputs["logits_hat"], target)
                )
            else:
                task = segmentation_loss(outputs["logits_tilde"], target)

            # Semantic consistency — only for inference branch
            if use_dpe:
                sc = semantic_consistency_loss(
                    outputs["teacher_source_logits"],
                    outputs["logits_hat"],
                )
            else:
                sc = torch.zeros((), device=device)

            # KL divergence: regularizes LPE's latent space
            kl = kl_divergence_loss(outputs["mu"], outputs["logvar"])

            # Prior constraint + distillation
            if use_dpe:
                warmup_ratio = min(1.0, epoch / prior_warmup_epochs)
                prior_weight = stage_cfg["lambda_prior"] * warmup_ratio
                distill_weight = distill_weight_base * warmup_ratio

                if use_gaussian:
                    # Pure Gaussian mode: z_hat generated from pure noise
                    # prior_loss computed against z_tilde
                    prior = prior_constraint_loss(outputs["z_hat"], outputs["z_tilde"], lpe_grad=True)
                    distill = distillation_loss(outputs["logits_hat"], outputs["logits_tilde"])
                    gaussian_losses.append(1.0)
                else:
                    # Conditional denoising mode: z_hat from z_tilde + noise
                    prior = prior_constraint_loss(outputs["z_hat"], outputs["z_tilde"], lpe_grad=True)
                    distill = distillation_loss(outputs["logits_hat"], outputs["logits_tilde"])
                    gaussian_losses.append(0.0)
            else:
                prior = torch.zeros((), device=device)
                distill = torch.zeros((), device=device)
                prior_weight = 0.0
                distill_weight = 0.0
                gaussian_losses.append(0.0)

            # Feature alignment: align student features to teacher features
            lambda_feat = stage_cfg.get("lambda_feat", 0.0)
            if lambda_feat > 0 and "raw_student_features" in outputs:
                feat_loss = feature_alignment_loss(
                    outputs["raw_student_features"],
                    outputs["teacher_target_features"],
                    align_layers=["layer3", "layer4"],
                )
            else:
                feat_loss = torch.zeros((), device=device)

            total = (
                stage_cfg["lambda_task"] * task +
                stage_cfg["lambda_sc"] * sc +
                stage_cfg["lambda_kl"] * kl +
                prior_weight * prior +
                distill_weight * distill +
                lambda_feat * feat_loss
            )

            if not torch.isfinite(total):
                logging.warning(
                    "skip batch at epoch %03d: task=%.4f sc=%.4f prior=%.4f",
                    epoch, task.item(), sc.item(), prior.item(),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            losses.append(total.item())
            task_losses.append(task.item())
            sc_losses.append(sc.item())
            kl_losses.append(kl.item())
            prior_losses.append((prior_weight * prior).item())
            distill_losses.append((distill_weight * distill).item())

        metrics = evaluate_pdaf(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "task_loss": float(np.mean(task_losses)),
            "sc_loss": float(np.mean(sc_losses)),
            "kl_loss": float(np.mean(kl_losses)),
            "prior_loss": float(np.mean(prior_losses)),
            "distill_loss": float(np.mean(distill_losses)),
            "gaussian_ratio": float(np.mean(gaussian_losses)) if gaussian_losses else 0.0,
            **metrics,
        }
        history.append(row)
        save_history(output_dir / "pdaf_metrics.csv", history)
        logging.info(
            "pdaf epoch=%03d loss=%.4f task=%.4f sc=%.4f kl=%.4f prior=%.4f distill=%.4f "
            "gaussian_ratio=%.2f dice=%.4f soft_dice=%.4f iou=%.4f auc=%.4f",
            epoch,
            row["loss"],
            row["task_loss"],
            row["sc_loss"],
            row["kl_loss"],
            row["prior_loss"],
            row["distill_loss"],
            row["gaussian_ratio"],
            metrics["dice"],
            metrics["soft_dice"],
            metrics["iou"],
            metrics["auc"],
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "teacher": model.teacher.state_dict(),
                    "student": model.student.state_dict(),
                    "lpe": model.lpe.state_dict(),
                    "dcm": model.dcm.state_dict(),
                    "dpe": model.dpe.state_dict(),
                    "metrics": metrics,
                    "config": cfg,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(
                    "early stopping at epoch %03d: no improvement for %d epochs (best dice=%.4f)",
                    epoch, patience, best_dice,
                )
                break

    return checkpoint_path


def load_pdaf_checkpoint(model: PDAFUNet, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.teacher.load_state_dict(checkpoint["teacher"])
    model.student.load_state_dict(checkpoint["student"])
    model.lpe.load_state_dict(checkpoint["lpe"])
    model.dcm.load_state_dict(checkpoint["dcm"])
    model.dpe.load_state_dict(checkpoint["dpe"])
    model.freeze_teacher()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PDAF v3 — DPE dual-mode, improved hyperparameters.")
    parser.add_argument("--config", type=Path, default=Path("hyps/unet_v3.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-teacher-pretrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["data"]["root"] = str(args.data_root)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        cfg["seed"] = args.seed

    set_seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train_v3.log", encoding="utf-8"),
        ],
    )

    device = torch.device(args.device)
    train_loader = DataLoader(
        TrainPatchDataset(
            root=Path(cfg["data"]["root"]),
            patch_size=cfg["data"]["patch_size"],
            samples_per_epoch=cfg["data"]["steps_per_epoch"] * cfg["data"]["batch_size"],
        ),
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EvalDataset(Path(cfg["data"]["root"]), split=cfg["data"]["val_split"]),
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    pdaf_model = PDAFUNet(cfg).to(device)

    if args.eval_only:
        checkpoint_path = args.checkpoint or (output_dir / "pdaf_best.pt")
        load_pdaf_checkpoint(pdaf_model, checkpoint_path, device)
        metrics = evaluate_pdaf(pdaf_model, val_loader, device, pred_dir=output_dir / "predictions")
        logging.info("eval metrics: %s", metrics)
        return

    teacher_path = args.teacher_checkpoint or (output_dir / "teacher_best.pt")
    if not teacher_path.exists():
        if args.skip_teacher_pretrain:
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        logging.info("stage 1/2: pretraining frozen teacher backbone")
        teacher_model = pdaf_model.teacher
        teacher_path = train_teacher(teacher_model, train_loader, val_loader, device, cfg, output_dir)
    else:
        logging.info("using existing teacher checkpoint: %s", teacher_path)

    pdaf_model.load_teacher(str(teacher_path))
    logging.info("stage 2/2: training PDAF modules and student network (teacher-init student)")
    checkpoint_path = train_pdaf(pdaf_model, train_loader, val_loader, device, cfg, output_dir)
    load_pdaf_checkpoint(pdaf_model, checkpoint_path, device)
    metrics = evaluate_pdaf(pdaf_model, val_loader, device, pred_dir=output_dir / "predictions")
    logging.info("best validation metrics: %s", metrics)


if __name__ == "__main__":
    main()