################ RESULTS ##########################
# Train acc: 99.37%
# Val acc: 79.00%
# Best val acc: 79.34%  
# Train time per epoch <2mins
###################################################



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR10
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import os
import sys
import time

from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
#from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *
from fastai.callback.mixup import *
from tqdm import tqdm

current_dir = "/home/madhu/.local/TEMP_CODES_FINAL"
print(current_dir)
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)
print(arch_dir)
result_dir = os.path.join(current_dir, 'Trained_models')
print(result_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

from resnet19_custom import my_resnet19

def evaluate(fabric, model, test_loader):
    model.eval()
    test_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=100).to(fabric.device)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.as_subclass(torch.Tensor).to(fabric.device)
            target=target.as_subclass(torch.Tensor).to(fabric.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            accuracy(output, target)

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

def get_dataloaders(
    bs=64,
    item_tfms=[RandomResizedCrop(size=224, min_scale=0.35), FlipItem(0.5)],
    batch_tfms=RandomErasing(p=0.9, max_count=3),
    IMAGENET100_PATH = "/home/madhu/.local/TEMP_CODES_FINAL/Datasets/ImageNet-100/"
):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(train_name="train", valid_name="val"),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    return dblock.dataloaders(IMAGENET100_PATH,path=IMAGENET100_PATH, bs=bs, num_workers=8)

def main():
    fabric = Fabric(accelerator="cuda", devices=4, strategy="deepspeed_stage_2", precision="bf16-mixed")
    fabric.launch()

    seed_everything(42)
    batch_size = 128
    dls = get_dataloaders(bs=batch_size,
                      #batch_tfms=[RandomErasing(p=0.4, max_count=1)],
                      batch_tfms=[],
                      item_tfms=[Resize(size=224), FlipItem(0.5)]
                      #item_tfms=[RandomResizedCrop(224, min_scale=0.35), FlipItem(0.5)]
                      )
    train_loader, test_loader = dls.train, dls.valid
    #train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = my_resnet19(num_classes=100)
    epochs = 50
    lr = 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader),
                                                pct_start=0.2,              # % of cycle spent increasing LR
                                                anneal_strategy='cos',      # cosine annealing
                                                div_factor=25.0,            # initial_lr = max_lr/div_factor
                                                final_div_factor=1e4        # min_lr = max_lr/final_div_factor
                                                    )
    model, optimizer = fabric.setup(model, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    start = time.time()
    print('timing started')
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        running_loss, correct, total = 0, 0, 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data=data.as_subclass(torch.Tensor).to(fabric.device)
            target=target.as_subclass(torch.Tensor).to(fabric.device)
            output = model(data)
            loss=criterion(output, target)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            running_loss += loss.item() * data.size(0)
            _, preds = output.max(1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        print(f" Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        evaluate(fabric, model, test_loader)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")
    torch.save(model.state_dict(),'/data/madhu/img100/ResNet19_IN100.pt') 

if __name__ == "__main__":
    main()
    
