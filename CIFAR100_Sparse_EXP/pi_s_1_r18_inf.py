#!/usr/bin/env python3
"""
Evaluate TEMP / π² ResNet-18 on CIFAR-100 using Lightning Fabric.

Features:
- Supports TEMP / TEMP-s / baseline ResNet-18
- Correct DDP-safe evaluation
- Clean separation of concerns
- Easy to extend with noise / quantization / distillation

This file is **evaluation-only** (no training).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt

from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import argparse

# ───────────── Argument Parser ─────────────
parser = argparse.ArgumentParser(description="Evaluate pi^2_sparse networks")
parser.add_argument("--current_dir", type=str, default= "/home/madhu/TEMP_FINAL_CODES", help="Path to dataset directory")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student model checkpoint (.pt)")
args = parser.parse_args()
# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
CURRENT_DIR = args.current_dir
ARCH_DIR = os.path.join(CURRENT_DIR, "network_arch")
DATA_DIR = os.path.join(CURRENT_DIR, "Datasets")
# DATA_DIR = "/home/madhu/.local/TEMP_CODES_FINAL/Datasets"
sys.path.append(ARCH_DIR)

from resnet_18 import  ResNet18_TEMP


# ─────────────────────────────────────────────
# Cutout augmentation (used only in training;
# kept here for completeness / reuse)
# ─────────────────────────────────────────────
class Cutout:
    """Randomly masks square regions in an image."""
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


# ─────────────────────────────────────────────
# Evaluation function (DDP-safe)
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(fabric: Fabric, model: torch.nn.Module, dataloader: DataLoader):
    """
    Evaluate a model using Fabric.

    Notes:
    - Accuracy is reduced correctly across GPUs
    - Loss is explicitly reduced across ranks
    """
    model.eval()

    accuracy = Accuracy(task="multiclass", num_classes=100).to(fabric.device)
    total_loss = torch.tensor(0.0, device=fabric.device)

    for images, labels in dataloader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction="sum")

        total_loss += loss
        accuracy.update(outputs, labels)

    # Reduce loss across all ranks
    total_loss = fabric.all_reduce(total_loss, reduce_op="sum")

    avg_loss = total_loss.item() / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100

    fabric.print(f"[Eval] Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return avg_loss, acc


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def main():
    # Fabric setup (DDP is sufficient for evaluation)
    fabric = Fabric(
        accelerator="cuda",
        devices=4,
        strategy="ddp",
        precision="16-mixed",  # switch to "32-true" for final numbers
    )
    fabric.launch()
    seed_everything(42)

    # Dataset transforms
    stats = ((0.5071, 0.4867, 0.4408),
             (0.2675, 0.2565, 0.2761))

    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats),
    ])

    test_dataset = CIFAR100(
        DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = fabric.setup_dataloaders(test_loader)

    # ─────────────────────────────
    # Model configuration
    # ─────────────────────────────
    K_act = [16, 60, 110, 150, 250, 50]   # π² activation K

    model = ResNet18_TEMP(
        num_classes=100,
        K_act=K_act,
        temp1=True,
        temp2=True,
        temp3=True,
        temp4=True,
        temp5=True,
        temp6=True,
    )

    # Load pretrained checkpoint safely
    # ckpt_path = "/data/madhu/CIFAR100/ResNet18/ResNet18_cifar100_k_1_p4.pt"
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    model = fabric.setup(model)

    # Parameter count (rank-0 only)
    if fabric.global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # ─────────────────────────────
    # Evaluation
    # ─────────────────────────────
    start = time.time()
    evaluate(fabric, model, test_loader)
    elapsed = (time.time() - start) / 60.0

    fabric.print(f"Evaluation completed in {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
