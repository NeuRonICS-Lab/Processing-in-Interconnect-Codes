#!/usr/bin/env python3
"""
Train a π² LeNet-5 variant on Fashion-MNIST with Lightning Fabric.

- Uses LeNet5_K (π² TEMP version) as student.
- Loads a teacher LeNet5_K checkpoint and copies weights into the student.
- Tracks average training loss + gradient norm per epoch and saves plots.
- Fabric handles multi-GPU/DDP/AMP.

Usage:
------
python train_fmnist_lenet5_pi2.py \
    --current_dir /home/madhu/.local/TEMP_CODES_FINAL
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt


# ───────────── Argument Parser ─────────────
# Path setup
parser = argparse.ArgumentParser(description="Train PI2-NN on FMNIST with Fabric")
parser.add_argument(
    "--current_dir",
    type=str,
    default="/home/madhu/.local/Processing-in-Interconnect-Codes",
    help="Base project directory (containing network_arch and Datasets)",
)
args = parser.parse_args()


# ───────────── Path setup ─────────────
current_dir = args.current_dir
arch_dir = os.path.join(current_dir, "network_arch")
result_dir = os.path.join(current_dir, "Trained_models")
sys.path.append(arch_dir)

print("Project Directory:", current_dir)
print("Architecture Directory:", arch_dir)
print("Model Save Directory:", result_dir)

# Import model definitions
from LeNet_mnist import LeNet5_K


# ───────────── Evaluation functions ─────────────
@torch.no_grad()
def evaluate(fabric, model, dataloader):
    """Evaluate model with Fabric (multi-device aware)."""
    model.eval()
    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    for data, target in dataloader:
        output = model(data)
        total_loss += F.cross_entropy(output, target, reduction="sum").item()
        accuracy.update(output, target)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    fabric.print(f"[Eval] loss={avg_loss:.4f}, acc={acc:.2f}%")
    return avg_loss, acc


@torch.no_grad()
def evaluate_single_device(model, dataloader, device):
    """Evaluate model outside Fabric (single-device)."""
    model.eval()
    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += F.cross_entropy(output, target, reduction="sum").item()
        accuracy.update(output, target)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"[Eval] loss={avg_loss:.4f}, acc={acc:.2f}%")
    return avg_loss, acc


# ───────────── Main Training ─────────────
def main():
    # Fabric setup
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Dataset & transforms
    transform = tt.ToTensor()
    data_dir = os.path.join(current_dir, "Datasets")
    fmnist_train = FashionMNIST(data_dir, train=True, download=True, transform=transform)
    fmnist_test = FashionMNIST(data_dir, train=False, download=True, transform=transform)

    # Split into train / validation
    val_size = int(0.1 * len(fmnist_train))
    train_size = len(fmnist_train) - val_size
    train_data, val_data = random_split(fmnist_train, [train_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    # ── Student network ──
    K = [12, 50, 85, 25, 75]  # TEMP constraints
    student = LeNet5_K(gamma=K, temp1=True, temp2=True, temp3=True, temp4=True, temp5=True)

    # ── Teacher network ──
    teacher = LeNet5_K()
    teacher_ckpt = os.path.join(result_dir, "FMNIST/fmnist_lenet5.pt")
    teacher.load_state_dict(torch.load(teacher_ckpt))

    # Copy weights teacher → student
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        if ps.size() == pt.size():
            ps.data.copy_(pt.data)
        else:  # reshape conv vs linear if needed
            ps.data.copy_(pt.data.view(pt.size()[0], -1).T)

    # Quick eval before training
    # evaluate_single_device(student, test_loader, "cuda")
    # evaluate_single_device(teacher, test_loader, "cuda")

    # ── Training setup ──
    epochs = 20
    lr = 5e-3
    optimizer = optim.Adamax(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    student, optimizer = fabric.setup(student, optimizer)
    
    #inital accuracy of student
    evaluate(fabric, student, test_loader)
    fabric.print("Starting training...")
    start = time.time()

    for epoch in range(1, epochs + 1):
        fabric.print(f"\nEpoch {epoch}/{epochs}")
        student.train()
        epoch_loss, num_batches = 0.0, 0

        for data, target in train_loader:
            output = student(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        fabric.print(f"[Train] loss={avg_loss:.4f}")

        # Validation + test eval
        fabric.print("[Val]")
        evaluate(fabric, student, val_loader)
        fabric.print("[Test]")
        evaluate(fabric, student, test_loader)

        scheduler.step()

    elapsed = (time.time() - start) / 60
    fabric.print(f"Training finished in {elapsed:.2f} min")

    # ── Final test evaluation ──
    fabric.print("[Final Test]")
    evaluate(fabric, student, test_loader)

    # ── Save trained model (rank-0 only) ──
    if fabric.global_rank == 0:
        save_dir = os.path.join(result_dir, "FMNIST")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "fmnist_pi2_student.pt")
        torch.save(student.state_dict(), save_path)
        fabric.print(f"Model saved to {save_path}")

      


if __name__ == "__main__":
    main()
