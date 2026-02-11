#!/usr/bin/env python3
"""
Train a TEMP / π² ResNet-18 on CIFAR-10 with knowledge distillation.

Pipeline:
- Dense ResNet-18 teacher
- TEMP/π² reference model (model1) for initialization
- TEMP/π² student model (model) for training
- Weight copying: model1 → model (kept exactly as provided)
- Knowledge distillation (KL + CE)
- Multi-GPU training via Lightning Fabric

"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as tt

from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Train TEMP / π² ResNet-18 on CIFAR-10 with distillation"
    )

    parser.add_argument("--current_dir", type=str, required=True,
                        help="Project root directory (contains network_arch/, Datasets/)")
    parser.add_argument("--teacher_ckpt", type=str,default = "/home/madhu/.local/Processing-in-Interconnect-Codes/Trained_models/CIFAR10_sparse/ResNet18_cifar10_cutout_mlp_43.pt",help="Path to teacher ResNet-18 checkpoint")
    parser.add_argument("--init_ckpt", type=str,default = "/home/madhu/.local/Processing-in-Interconnect-Codes/Trained_models/CIFAR10_sparse/ResNet18_cifar10_k_new43_2.pt",
                        help="Checkpoint for model1 (TEMP init model)")
    parser.add_argument("--save_ckpt", type=str, default="/home/madhu/.local/Processing-in-Interconnect-Codes/Trained_models/CIFAR10_sparse/ResNet18_cifar10_k_new43_31.pt",
                        help="Path to save best student checkpoint")

    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-8)

    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss")

    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["16-mixed", "32-true"])

    return parser.parse_args()


# ─────────────────────────────────────────────
# Cutout augmentation
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
# Distillation loss
# ─────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)

    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss


# ─────────────────────────────────────────────
# Evaluation (DDP-safe)
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(fabric, model, dataloader):
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    total_loss = torch.tensor(0.0, device=fabric.device)

    for images, labels in dataloader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction="sum")
        total_loss += loss
        accuracy.update(outputs, labels)

    total_loss = fabric.all_reduce(total_loss, reduce_op="sum")
    avg_loss = total_loss.item() / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100

    fabric.print(f"[Eval] Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return avg_loss, acc

# ─────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────
def main():
    args = get_args()
    seed_everything(42)

    # ── Path setup ──
    arch_dir = os.path.join(args.current_dir, "network_arch")
    data_dir = os.path.join(args.current_dir, "Datasets")
    # data_dir = "/home/madhu/.local/TEMP_CODES_FINAL/Datasets"
    sys.path.append(arch_dir)

    from resnet_18 import ResNet18, ResNet18_TEMP

    # ── Fabric setup ──
    fabric = Fabric(
        accelerator="cuda",
        devices=4,
        strategy="ddp",
        precision=args.precision,
    )
    fabric.launch()
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         Cutout(n_holes=1, length=16),
                         tt.Normalize(*stats,inplace=True)])
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats),
    ])

    train_set = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    gamma = [16,50,100,100,200,50]
     # Final TEMP / π² student
    model = ResNet18_TEMP(
        num_classes=10,
        K_act=gamma,
        temp1=True, temp2=True, temp3=True,
        temp4=True, temp5=True, temp6=True,
    )

    # TEMP reference model (initialization only)
    model1 = ResNet18_TEMP(
        num_classes=10,
        K_act=gamma,
        temp1=True, temp2=True, temp3=True,
        temp4=True, temp5=True,
    )

    # Dense teacher
    teacher = ResNet18(num_classes=10)

    # ─────────────────────────────────────────
    # Load checkpoints
    # ─────────────────────────────────────────
    model1.load_state_dict(torch.load(args.init_ckpt))
    teacher.load_state_dict(torch.load(args.teacher_ckpt))
    teacher.eval()

    model1 = model1.to(fabric.device)
    teacher = teacher.to(fabric.device)

    # ─────────────────────────────────────────
    # Weight copying: model1 → model (UNCHANGED)
    # ─────────────────────────────────────────
    for p_model, p_model_T in zip(model.parameters(), model1.parameters()):
        if (p_model.size() == p_model_T.size()):
            p_model.data = p_model_T.data.contiguous().clone()
        else:
            reshaped = (
                p_model_T.data
                .clone()
                .contiguous()
                .view(p_model_T.size()[0], -1)
                .T
            )
            p_model.data = reshaped.contiguous()

    # ── Optimizer & scheduler ──
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
    )

    model, optimizer = fabric.setup(model, optimizer)

    # ─────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        fabric.print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()

        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            student_logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                args.temperature, args.alpha
            )

            fabric.backward(loss)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        fabric.print(f"[Train] Loss: {avg_train_loss:.4f}")

        _, acc = evaluate(fabric, model, test_loader)

        if acc > best_acc and fabric.global_rank == 0:
            best_acc = acc
            torch.save(model.state_dict(), args.save_ckpt)
            fabric.print(f"✔ Saved new best model ({acc:.2f}%)")

    elapsed = (time.time() - start) / 60
    fabric.print(f"Training completed in {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()

