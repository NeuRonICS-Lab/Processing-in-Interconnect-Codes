# Processing-in-Interconnect-Codes

This repository contains implementations of **π² (Processing-in-Interconnect)** neural network layers and architectures, along with training/evaluation scripts on MNIST, Fashion-MNIST, and CIFAR datasets. The project leverages [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) for distributed training, mixed precision, and scaling across GPUs.

---
## 🔧 Environment Setup

We recommend using **conda**:

'''
# Create a new environment
conda create -n fabric2 python==3.10 -y
conda activate fabric2
'''

# Install dependencies
pip install torch torchvision lightning fabric deepspeed

# Clone this repository
git clone https://github.com/NeuRonICS-Lab/Processing-in-Interconnect-Codes.git
cd Processing-in-Interconnect-Codes

This repo contains the following directories
Processing-in-Interconnect-Codes/
│
├── PI2_Layers/           # TEMP / π² custom PyTorch layers (Conv, FC, MLP blocks)
├── network_arch/         # Network definitions (ResNet9, LeNet, TEMP-MLP, etc.)
├── MNIST_EXP/            # Training/eval scripts for MNIST experiments
├── FMNIST_EXP/           # Training/eval scripts for Fashion-MNIST experiments
├── CIFAR10_EXP/          # Training/eval scripts for CIFAR10 experiments
├── CIFAR100_EXP/         # Training/eval scripts for CIFAR100 experiment
├── Trained_models/       # Pretrained checkpoints (saved after training)
└── README.md             # (this file)


To reproduce the MNIST experiment results run
python MNIST_EXP/fabric_PI2_1.py --current_dir /path/to/Processing-in-Interconnect-Codes

To reproduce the FMNIST training experiment results run
python FMNIST_EXP/pi_2_train.py --current_dir /path/to/Processing-in-Interconnect-Codes

To reproduce the CIFAR100 training experiment results run
python CIFAR100_EXP/pi_2_train.py --current_dir /path/to/Processing-in-Interconnect-Codes

​To reproduce the CIFAR10 training experiment results run
python CIFAR10_EXP/pi_2_train.py --current_dir /path/to/Processing-in-Interconnect-Codes

Run inference directly on saved π²-NN models
You can evaluate checkpoints without retraining:
# CIFAR-100
python CIFAR100_EXP/pi_2_inf1.py \
  --current_dir /path/to/Processing-in-Interconnect-Codes \
  --ckpt Trained_models/CIFAR100/ResNet9_cifar100_k_new_new.pt

# CIFAR-10
python CIFAR10_EXP/pi_2_inf2.py \
  --current_dir /path/to/Processing-in-Interconnect-Codes \
  --ckpt Trained_models/CIFAR10/ResNet9_cifar10_k_new.pt
