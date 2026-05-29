from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))


class FakeVisdom:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


sys.modules["visdom"] = FakeVisdom()

from datasets.preprocessing import RetinalPreprocessor
from models.unet_pdaf import PDAFUNet, UNetBackbone
from train_unet_pdaf import (
    compute_metrics,
    load_pdaf_checkpoint,
    normalize_batch,
    read_binary_mask,
    read_rgb,
    set_seed,
)


def infer_dataset_name(root: Path) -> str:
    name = root.name.upper()
    if "CHASE" in name:
        return "CHASE_DB1"
    if "DRIVE" in name:
        return "DRIVE"
    if "FIVES" in name:
        return "FIVES"
    if "HRF" in name:
        return "HRF"
    if "STARE" in name:
        return "STARE"
    raise ValueError(f"Unsupported dataset root: {root}")


def estimate_fov(image: np.ndarray) -> np.ndarray:
    return (image.mean(axis=2) > 0.03).astype(np.float32)


def discover_eval_pairs(root: Path, split: str) -> list[tuple[Path, Path, Path | None]]:
    dataset = infer_dataset_name(root)

    if dataset in {"DRIVE", "CHASE_DB1"}:
        input_dir = root / split / "input"
        label_dir = root / split / "label"
        if not input_dir.exists():
            input_dir = root / "images" / split
        if not label_dir.exists():
            label_dir = root / "labels" / split
        if not input_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing split directories under {root}")

        image_paths = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        pairs: list[tuple[Path, Path, Path | None]] = []
        for image_path in image_paths:
            stem = image_path.stem
            if dataset == "DRIVE":
                label_candidates = [
                    label_dir / f"{stem}.png",
                    label_dir / f"{stem}_manual1.png",
                ]
                fov = None
                drive_id = stem.split("_")[0]
                for suffix in [f"{drive_id}_training_mask.gif", f"{stem}_training_mask.gif"]:
                    candidate = root / "mask" / suffix
                    if candidate.exists():
                        fov = candidate
                        break
            else:
                label_candidates = [label_dir / f"{stem}_1stHO.png", label_dir / f"{stem}.png"]
                fov = None

            label_path = next((candidate for candidate in label_candidates if candidate.exists()), None)
            if label_path is not None:
                pairs.append((image_path, label_path, fov))
        if pairs:
            return pairs

    if dataset == "FIVES":
        split_dir = root / split
        image_dir = split_dir / "Original"
        label_dir = split_dir / "Ground truth"
        pairs = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() != ".png":
                continue
            label_path = label_dir / image_path.name
            if label_path.exists():
                pairs.append((image_path, label_path, None))
        if pairs:
            return pairs

    if dataset == "HRF":
        image_dir = root / "images"
        label_dir = root / "manual1"
        fov_dir = root / "mask"
        pairs = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            stem = image_path.stem
            label_path = label_dir / f"{stem}.tif"
            fov_path = fov_dir / f"{stem}_mask.tif"
            if label_path.exists():
                pairs.append((image_path, label_path, fov_path if fov_path.exists() else None))
        if pairs:
            return pairs

    if dataset == "STARE":
        image_paths = sorted(
            p for p in root.iterdir() if p.suffix.lower() == ".ppm" and not p.stem.endswith((".ah", ".vk"))
        )
        pairs = []
        for image_path in image_paths:
            stem = image_path.stem
            # Use the first observer as the default supervision target.
            label_path = root / f"{stem}.ah.ppm"
            if label_path.exists():
                pairs.append((image_path, label_path, None))
        if pairs:
            return pairs

    raise RuntimeError(f"No evaluation pairs found for {root} split={split}")


class GenericEvalDataset(Dataset):
    """Whole-image evaluation with the same normalization path used in training."""

    def __init__(self, root: Path, split: str = "val", use_retinal_preprocessing: bool = False) -> None:
        self.pairs = discover_eval_pairs(root, split)
        self.use_retinal_preprocessing = use_retinal_preprocessing
        self.preprocessor = None
        if use_retinal_preprocessing:
            self.preprocessor = RetinalPreprocessor(
                clip_limit=2.0,
                tile_size=(8, 8),
                apply_clahe=True,
                apply_fov_crop=True,
                normalize_method="gaussian",
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_path, mask_path, fov_path = self.pairs[index]
        rgb = read_rgb(image_path)
        mask = read_binary_mask(mask_path)
        if fov_path is not None:
            fov = read_binary_mask(fov_path)
        else:
            fov = estimate_fov(rgb)

        if self.preprocessor is None:
            image = rgb
        else:
            processed = self.preprocessor.preprocess_image(
                str(image_path),
                str(fov_path) if fov_path is not None else None,
            ).astype(np.float32)
            image = np.repeat(processed[..., None], 3, axis=2)

        return (
            torch.from_numpy(image.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask[None]).float(),
            torch.from_numpy(fov[None]).float(),
            image_path.stem,
        )


def run_model(
    model: PDAFUNet | UNetBackbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    rows = []
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    fovs: list[np.ndarray] = []

    with torch.no_grad():
        for raw_image, target, fov, name in loader:
            image = normalize_batch(raw_image.to(device))
            if isinstance(model, PDAFUNet):
                logits = model.infer(image)
            else:
                logits = model(image)

            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            target_np = target.numpy()[0, 0]
            fov_np = fov.numpy()[0, 0]

            rows.append(compute_metrics(prob, target_np, fov_np))
            probs.append(prob)
            targets.append(target_np)
            fovs.append(fov_np)
            print(f"  {name[0]}: dice={rows[-1]['dice']:.4f}, auc={rows[-1]['auc']:.4f}")

    metrics = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}
    return metrics, probs, targets, fovs


def best_threshold_metrics(probs: list[np.ndarray], targets: list[np.ndarray], fovs: list[np.ndarray]) -> tuple[float, float]:
    best_dice = -1.0
    best_thresh = 0.5
    for thresh in np.arange(0.02, 0.81, 0.02):
        tp = fp = fn = 0
        for prob, target, fov in zip(probs, targets, fovs):
            valid = fov > 0.5
            pred = (prob >= thresh).astype(np.float32)
            tp += np.logical_and(pred == 1, target == 1)[valid].sum()
            fp += np.logical_and(pred == 1, target == 0)[valid].sum()
            fn += np.logical_and(pred == 0, target == 1)[valid].sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        if dice > best_dice:
            best_dice = float(dice)
            best_thresh = float(thresh)
    return best_thresh, best_dice


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  IoU:  {metrics['iou']:.4f}")
    print(f"  Acc:  {metrics['acc']:.4f}")
    print(f"  Sen:  {metrics['sen']:.4f}")
    print(f"  Spe:  {metrics['spe']:.4f}")
    print(f"  AUC:  {metrics['auc']:.4f}")


def evaluate_mode(
    model: PDAFUNet | UNetBackbone,
    dataset_root: Path,
    split: str,
    device: torch.device,
    model_name: str,
    mode_name: str,
    use_retinal_preprocessing: bool,
) -> dict[str, float]:
    print(f"\n=== {dataset_root.name} eval: {model_name} | {mode_name} ===")
    dataset = GenericEvalDataset(dataset_root, split=split, use_retinal_preprocessing=use_retinal_preprocessing)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics, probs, targets, fovs = run_model(model, loader, device)
    print_metrics(f"{model_name} | {mode_name} metrics", metrics)
    best_thresh, best_dice = best_threshold_metrics(probs, targets, fovs)
    print(f"  Best-threshold Dice: {best_dice:.4f} @ threshold={best_thresh:.2f}")
    return {
        "target_dataset": dataset_root.name,
        "model": model_name,
        "mode": mode_name,
        **metrics,
        "best_threshold": best_thresh,
        "best_threshold_dice": best_dice,
    }


def save_results(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_dataset",
                "model",
                "mode",
                "dice",
                "iou",
                "acc",
                "sen",
                "spe",
                "auc",
                "best_threshold",
                "best_threshold_dice",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain retinal evaluation for CHASE_DB1-trained teacher/PDAF.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda:1")
    parser.add_argument("--config", default="hyps/unet.yaml", help="Config used to build the model.")
    parser.add_argument("--target-root", default="data/DRIVE", help="Target dataset root for cross-domain evaluation.")
    parser.add_argument("--target-split", default="val", help="Target split or subset name.")
    parser.add_argument("--checkpoint", default="outputs/unet_pdaf_retinal_rescue/pdaf_best.pt", help="PDAF checkpoint.")
    parser.add_argument(
        "--teacher-checkpoint",
        default="outputs/unet_pdaf_teacher_pretrain/teacher_best.pt",
        help="Teacher checkpoint trained on the same source dataset.",
    )
    parser.add_argument(
        "--output",
        default="outputs/unet_pdaf_retinal_rescue/cross_domain_comparison.csv",
        help="CSV summary path.",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Run only the official main evaluation and skip supplementary analyses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    teacher_model = UNetBackbone(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        out_channels=cfg["model"]["out_channels"],
    ).to(device)
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher_model.load_state_dict(teacher_ckpt["model"])
    teacher_model.eval()

    pdaf_model = PDAFUNet(cfg).to(device)
    load_pdaf_checkpoint(pdaf_model, args.checkpoint, device)
    print("Models loaded")

    target_root = Path(args.target_root)
    print("\nOfficial cross-domain result = whole-image inference + training-consistent normalization + FOV metrics + threshold 0.5")
    rows = []
    rows.append(evaluate_mode(teacher_model, target_root, args.target_split, device, "teacher", "official_main_eval", False))
    rows.append(evaluate_mode(pdaf_model, target_root, args.target_split, device, "pdaf", "official_main_eval", False))

    if not args.official_only:
        print("\nSupplementary analyses below; these are not the main reported result.")
        rows.append(evaluate_mode(teacher_model, target_root, args.target_split, device, "teacher", "oracle_threshold_reference", False))
        rows.append(evaluate_mode(pdaf_model, target_root, args.target_split, device, "pdaf", "oracle_threshold_reference", False))
        rows.append(evaluate_mode(teacher_model, target_root, args.target_split, device, "teacher", "retinal_preprocessed_reference", True))
        rows.append(evaluate_mode(pdaf_model, target_root, args.target_split, device, "pdaf", "retinal_preprocessed_reference", True))

    save_results(rows, Path(args.output))
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
