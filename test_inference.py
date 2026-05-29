"""
Test inference with pretrained PDAF weights on DRIVE dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Patch visdom BEFORE any imports
class FakeVisdom:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
sys.modules['visdom'] = FakeVisdom()

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import yaml


def read_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_mask(path):
    mask = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return (mask > 127).astype(np.float32)


def normalize_image(image):
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    return (image - mean) / (std + 1e-6)


def build_pairs(root, split):
    input_dir = root / split / "input"
    label_dir = root / split / "label"
    pairs = []
    for image_path in sorted(input_dir.glob("*.tif")):
        stem = image_path.stem
        candidates = [label_dir / f"{stem}.png", label_dir / f"{stem}_manual1.png"]
        label_path = next((p for p in candidates if p.exists()), None)
        if label_path is None:
            raise FileNotFoundError(f"Missing label for {image_path}")
        pairs.append((image_path, label_path))
    return pairs


class DriveTestDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.pairs = build_pairs(root, "test")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        raw_image = read_image(image_path)
        image = normalize_image(raw_image)
        mask = read_mask(label_path)
        fov_path = self.root / "mask" / f"{image_path.stem.split('_')[0]}_training_mask.gif"
        if fov_path.exists():
            fov = read_mask(fov_path)
        else:
            fov = (raw_image.mean(axis=2) > 0.03).astype(np.float32)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask[None]).float()
        fov_t = torch.from_numpy(fov[None]).float()
        return image_t, mask_t, fov_t, image_path.stem


class Args:
    def __init__(self):
        self.lpe_weight = "checkpoints/best-lpe-bdd100k.pt"
        self.dcm_weight = "checkpoints/best-dcm-bdd100k.pt"
        self.dpe_weight = "checkpoints/best-dpe-bdd100k.pt"
        self.scales = [1.0]
        self.vhflip = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/DRIVE"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_inference"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args_obj = Args()
    args_obj.device = args.device

    print(f"Device: {device}")
    print("Testing PDAF inference on DRIVE dataset")

    # Load config
    cfg_path = Path("checkpoints/deeplab_res50.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Task Model: {cfg['Task Model']}")

    # Build model - use PDAF which wraps DCM, DPE, LPE
    from models.pdaf import build_PDAF

    model = build_PDAF(args_obj, cfg)
    model = model.to(device)
    model.eval()
    print("Model built successfully")

    # Load data
    test_dataset = DriveTestDataset(args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test dataset: {len(test_dataset)} images")

    # Evaluate
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks, fovs, names in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            fovs = fovs.to(device)

            # TTA inference - returns numpy array of class indices
            preds = model.tta_forward(images, scales=args_obj.scales, hflip=args_obj.vhflip)

            # Binarize - preds are already 0/1 class indices
            preds_binary = torch.from_numpy(preds).float().to(device)
            masks_binary = (masks > 0).float()
            fovs_binary = (fovs > 0)

            tp += ((preds_binary == 1) & (masks_binary == 1) & fovs_binary).sum().item()
            fp += ((preds_binary == 1) & (masks_binary == 0) & fovs_binary).sum().item()
            tn += ((preds_binary == 0) & (masks_binary == 0) & fovs_binary).sum().item()
            fn += ((preds_binary == 0) & (masks_binary == 1) & fovs_binary).sum().item()

            print(f"{names[0]}: tp={tp}, fp={fp}, tn={tn}, fn={fn}")

    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    acc = (tp + tn) / (tp + fp + tn + fn)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n=== Results ===")
    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")
    print(f"Acc:  {acc:.4f}")
    print(f"Sen:  {sen:.4f}")
    print(f"Spe:  {spe:.4f}")


if __name__ == "__main__":
    main()