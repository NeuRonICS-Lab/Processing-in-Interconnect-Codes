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

# ───────────────────────── Argument Parser ─────────────────────────────
parser = argparse.ArgumentParser(description="Distillation with Quantized Student on CIFAR-10")
parser.add_argument("--current_dir", type=str, default= "/home/madhu/.local/Processing-in-Interconnect-Codes", help="Path to dataset directory")
parser.add_argument("--n_bits", type=int, default=3, help="Number of bits for quantization")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for fine-tuning")
parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="CUDA device IDs")
args = parser.parse_args()

# ───────────────────────── Setup ─────────────────────────────
# ───────────── Setup Import Path ─────────────
current_dir = args.current_dir
print(current_dir)
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)
data_dir = os.path.join(current_dir, 'Datasets')
sys.path.append(data_dir)
results_dir = os.path.join(current_dir, 'Trained_models')
sys.path.append(results_dir)

from resnet_9 import ResNet9_100_temp1

torch.manual_seed(3407)

# ───────────────────────── Cutout ─────────────────────────────
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.
        return img * torch.from_numpy(mask).expand_as(img)

# ───────────────────────── Symmetric Quantization ─────────────────────────────
def symmetric_quantize(tensor, n_bits=4):
    qmax = 2**(n_bits - 1) - 1
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    q_tensor = (tensor / scale).round().clamp(-qmax, qmax)
    return q_tensor * scale

# ───────────────────────── Evaluation ─────────────────────────────
def evaluate(fabric, model, dataloader):
    model.eval()
    loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()
            accuracy.update(output, target)
    avg_loss = loss / len(dataloader.dataset)
    acc = accuracy.compute().item() * 100
    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return avg_loss, acc

# ───────────────────────── Distillation Loss ─────────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    distill = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    cls = F.cross_entropy(student_logits, labels)
    return alpha * distill + (1 - alpha) * cls

# ───────────────────────── Main ─────────────────────────────
def main():
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp", precision="16-mixed")
    fabric.launch()
    seed_everything(42)

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = tt.Compose([
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        Cutout(n_holes=1, length=16),
        tt.Normalize(*stats, inplace=True)
    ])
    transform_test = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    gamma = [16, 50, 35, 35, 60, 70, 80, 200, 50]
    student = ResNet9_100_temp1(3, out=10, gamma=gamma,
                          temp0=True, temp1=True, temp2=True, temp3=True,
                          temp4=True, temp5=True, temp6=True, temp7=True, temp8=True).to(fabric.device)
    teacher = ResNet9_100_temp1(3, out=10).to(fabric.device)

    student.load_state_dict(torch.load(os.path.join(results_dir, "CIFAR10", "ResNet9_cifar10_k_new.pt")))
    teacher.load_state_dict(torch.load(os.path.join(results_dir, "CIFAR10", "ResNet9_cifar10.pt")))

    # Initial quantization
    with torch.no_grad():
        for _, param in student.named_parameters():
            param.data.copy_(symmetric_quantize(param.data, n_bits=args.n_bits))

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), three_phase=True,
        final_div_factor=10.0, pct_start=0.4
    )
    student, optimizer = fabric.setup(student, optimizer)

    print("Starting Training...")
    accmax = 0
    temperature = 5.0
    alpha = 0.5
    start = time.time()

    evaluate(fabric, student, test_loader)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        student.train()
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        for data, target in train_loader:
            with torch.no_grad():
                teacher_output = teacher(data)
            student_output = student(data)
            loss = distillation_loss(student_output, teacher_output, target, temperature, alpha)

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Apply symmetric quantization
            with torch.no_grad():
                for _, param in student.named_parameters():
                    param.data.copy_(symmetric_quantize(param.data, n_bits=args.n_bits))

        _, acc = evaluate(fabric, student, test_loader)
        if acc > accmax:
            accmax = acc
            torch.save(student.state_dict(), os.path.join(results_dir, "CIFAR10", "ResNet9_cifar10_k_qat.pt"))


    elapsed = (time.time() - start) / 60
    print(f"Training completed in {elapsed:.2f} minutes")

if __name__ == "__main__":
    main()
