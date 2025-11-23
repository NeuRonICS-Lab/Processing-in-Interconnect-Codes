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

from resnet19_custom import resnet19_TEMPs

# Function to register hooks and store gradients
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soft targets from teacher and student
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    # Distillation loss (soft cross-entropy)
    distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature**2)
    # Classification loss (hard cross-entropy)
    classification_loss = F.cross_entropy(student_logits, labels)
    # Combined loss
    return alpha * distillation_loss + (1 - alpha) * classification_loss

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

import random
from fastai.vision.all import *


def get_dataloaders(
    bs=64,
    item_tfms=[RandomResizedCrop(size=75, min_scale=0.35), FlipItem(0.5)],
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

    return dblock.dataloaders(IMAGENET100_PATH,path=IMAGENET100_PATH, bs=bs, num_workers=8,pin_memory=True)

def main():
    fabric = Fabric(accelerator="cuda", devices=4, strategy="deepspeed_stage_2", precision="bf16-mixed")
    fabric.launch()

    seed_everything(42)
    batch_size = 128
    dls = get_dataloaders(bs=batch_size,
                      batch_tfms=[],
                      item_tfms=[Resize(size=75), FlipItem(0.5)]
                      )
    train_loader, test_loader = dls.train, dls.valid
    xb, yb = dls.valid.one_batch()
    print(xb.shape)
    xb, yb = dls.train.one_batch()
    print(xb.shape)
    #train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    K_in = [0.5,0.3,0.3,0.3,0.3,0.3]  #0.3 - 17.5, 0.2 - 16
    K_act = [10,16,16,16,16,16] #50-92.77,200 - 92.85
    model = resnet19_TEMPs(num_classes=100,temp1=True,K_act=K_act,K_in=K_in)
    
    #TEACHER MODEL
    model_T = resnet19_TEMPs(num_classes=100)
    model_T.load_state_dict(torch.load("/data/madhu/img100/ResNet19_IN100_75.pt"))
    for p_model, p_model_T in zip(model.parameters(), model_T.parameters()):
        if(p_model.size()==p_model_T.size()):
            print(p_model.size())
            p_model.data = p_model_T.data.contiguous().clone()
        else:
            break #try cloning the last layer too 
    epochs = 10
    lr = 1e-3 #5e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) #1e-3 (73.42)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader),
                                                pct_start=0.2,              # % of cycle spent increasing LR
                                                anneal_strategy='cos',      # cosine annealing
                                                div_factor=25.0,            # initial_lr = max_lr/div_factor
                                                final_div_factor=1e4        # min_lr = max_lr/final_div_factor
                                                )
    model, optimizer = fabric.setup(model, optimizer)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    temperature = 5.0  # Temperature for distillation
    alpha = 0.5      # Weight for distillation loss
    start = time.time()
    print('timing started')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    accmax = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        running_loss, correct, total = 0, 0, 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data=data.as_subclass(torch.Tensor).to(fabric.device)
            target=target.as_subclass(torch.Tensor).to(fabric.device)
            output = model(data)
            
            model_T = model_T.to(fabric.device)
            with torch.no_grad():
                teacher_output = model_T(data)  # Teacher outputs (no gradient required)
            loss = distillation_loss(output, teacher_output, target, temperature, alpha)

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            running_loss += loss.item() * data.size(0)
            _, preds = output.max(1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
        ll,acc = evaluate(fabric, model, test_loader)
        if(acc >accmax):
            accmax = acc
            torch.save(model.state_dict(),'/data/madhu/img100/ResNet19_IN100_k_75.pt') #50%
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
    
