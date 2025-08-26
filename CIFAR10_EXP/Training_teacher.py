import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as tt
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import numpy as np

# ───────────── Argument Parser ─────────────
parser = argparse.ArgumentParser(description="Train ResNet9 on CIFAR-10")
parser.add_argument("--current_dir", type=str, default= "/home/madhu/.local/Processing-in-Interconnect-Codes", help="Path to dataset directory")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="CUDA device IDs")
args = parser.parse_args()

# ───────────── Dynamic Import Setup ─────────────
current_dir = args.current_dir
arch_dir = os.path.join(current_dir, "network_arch")
sys.path.append(arch_dir)
data_dir = os.path.join(current_dir, 'Datasets')
sys.path.append(data_dir)
results_dir = os.path.join(current_dir, 'Trained_models')
sys.path.append(results_dir)
from resnet_9 import ResNet9_100_temp1

# ───────────── Evaluation Function ─────────────
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
def evaluate(fabric, model, dataloader):
    model.eval()
    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy.update(output, target)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Avg loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ───────────── Main ─────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    # Transforms
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         Cutout(n_holes=1, length=16),
                         tt.Normalize(*stats,inplace=True)])
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    # Dataset
    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Model
    model = ResNet9_100_temp1(in_channels=3, out=10)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        three_phase=True,
        final_div_factor=10.0,
        pct_start=0.3
    )

    model, optimizer = fabric.setup(model, optimizer)

    print("Training started...")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        for data, target in train_loader:
            output = model(data)
            loss = F.cross_entropy(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        evaluate(fabric, model, test_loader)

    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes")
    torch.save(model.state_dict(), os.path.join(results_dir, "CIFAR10", "ResNet9_cifar10.pt"))

if __name__ == "__main__":
    main()
