import torch
import torchmetrics
import os
import sys
import gc
import torchvision
import time
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import  CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.fabric import Fabric, seed_everything
from torchmetrics.classification import Accuracy
import torch.optim as optim
# Add the utils directory to the system path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
current_dir = "/home/madhu/.local/TEMP_CODES_FINAL"
print(current_dir)
arch_dir = os.path.join(current_dir, 'network_arch')
sys.path.append(arch_dir)
print(arch_dir)
result_dir = os.path.join(current_dir, 'Trained_models')
print(result_dir)
sys.path.append(result_dir)
training_dir = os.path.join(current_dir, 'Training_codes')
sys.path.append(training_dir)
print(training_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3407)

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
    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target.to(fabric.device), reduction="sum").item()
            accuracy(output.to(fabric.device), target.to(fabric.device))

    test_loss /= len(test_loader.dataset)
    acc = accuracy.compute().item() * 100

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n")
    return test_loss, acc

#define model arch and lightning template
class MPLayer_in_K(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    torch.manual_seed(43)
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)
    self.fn = MPLayer_in_K.SpikeK.apply

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn =F.relu((3+inputp))
        minusIn =F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp

      plusW = F.relu(1+self.weight)
      minusW = F.relu(1-self.weight)

      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)

      zPlus = self.fn(zPlus, self.gamma)
      zMinus = self.fn(zMinus, self.gamma)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return -(zPlus - zMinus)  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus

  @staticmethod
  class SpikeK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inMat1, gamma):
        ctx.gamma = gamma
        ctx.save_for_backward(inMat1)
        if gamma == 0 or gamma == 1:
            out = torch.kthvalue(inMat1, 1, dim=1).values
            return out
        thr, _ = torch.topk(inMat1, int(gamma), dim=1, largest=False, sorted=False)
        sum_nonzero = thr.sum(dim=1)
        return sum_nonzero / gamma  # average of the smallest gamma values

    @staticmethod
    def backward(ctx, grad_output):
        gamma = ctx.gamma
        inMat1, = ctx.saved_tensors
        grad_input = torch.zeros_like(inMat1)
        thr,indices = torch.topk(inMat1, gamma, dim=1, largest=False, sorted=False)
        grad_vals = (grad_output / gamma).unsqueeze(1).expand_as(indices)
        grad_input.scatter_(dim=1, index=indices, src=grad_vals)
        return grad_input, None

class MP_Network(torch.nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list,gamma1,gamma2,Temp=False):
    
        super(MP_Network, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        layers = []
        if(Temp):
            layers.append(MPLayer_in_K(in_features, num_hidden_list[0], gamma1,diff = 1))
        else:
            layers.append(nn.Linear(in_features, num_hidden_list[0],bias=False))
        for idx in range(num_layers - 1):
            layers.append(MPLayer_in_K(num_hidden_list[idx], num_hidden_list[idx+1], gamma2))
        
        self.linear = torch.nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(num_hidden_list[-1])
    def forward(self, inputs,scaling=1):
        inputs = inputs.flatten(start_dim=1)
        linearp,linearn = self.linear[0](inputs)
        linear = self.linear[1](linearp*scaling,linearn*scaling)
        return self.bn(linear)

class MP_Network_3(torch.nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list,gamma1,gamma2,gamma3,Temp=False,Temp1=False):
    
        super(MP_Network_3, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        layers = []
        if(Temp):
            layers.append(MPLayer_in_K(in_features, num_hidden_list[0], gamma1,diff = 1))
        else:
            layers.append(nn.Linear(in_features, num_hidden_list[0],bias=False))
        if(Temp1):
            layers.append(MPLayer_in_K(num_hidden_list[0], num_hidden_list[1], gamma2,diff = 1))
        else:
            layers.append(nn.Linear(num_hidden_list[0], num_hidden_list[1],bias=False))
        # for idx in range(num_layers - 2):
        layers.append(MPLayer_in_K(num_hidden_list[1], num_hidden_list[2], gamma3))
        
        self.linear = torch.nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(num_hidden_list[-1])
    def forward(self, inputs,scaling=1):
        inputs = inputs.flatten(start_dim=1)
        linearp,linearn = self.linear[0](inputs)
        linearp,linearn = self.linear[1](linearp*1,linearn*1)
        linear = self.linear[2](linearp*scaling,linearn*scaling)
        return self.bn(linear)

# class MP_Network_4(torch.nn.Module):
#     def __init__(self, in_features, num_layers, num_hidden_list,gamma1,gamma2,gamma3,gamma4,Temp=False,Temp1=False,Temp2=False):
    
#         super(MP_Network_4, self).__init__()
#         self.in_features = in_features
#         self.num_layers = num_layers
#         self.num_hidden_list = num_hidden_list
#         self.gamma1 = gamma1
#         self.gamma2 = gamma2
#         self.gamma3 = gamma3

#         layers = []
#         if(Temp):
#             layers.append(MPLayer_in_K(in_features, num_hidden_list[0], gamma1,diff = 1))
#         else:
#             layers.append(nn.Linear(in_features, num_hidden_list[0],bias=False))
#         if(Temp1):
#             layers.append(MPLayer_in_K(num_hidden_list[0], num_hidden_list[1], gamma2,diff = 1))
#         else:
#             layers.append(nn.Linear(num_hidden_list[0], num_hidden_list[1],bias=False))
#         if(Temp2):
#             layers.append(MPLayer_in_K(num_hidden_list[1], num_hidden_list[2], gamma3,diff = 1))
#         else:
#             layers.append(nn.Linear(num_hidden_list[1], num_hidden_list[2],bias=False))
#         # for idx in range(num_layers - 2):
#         layers.append(MPLayer_in_K(num_hidden_list[2], num_hidden_list[3], gamma4))
        
#         self.linear = torch.nn.Sequential(*layers)
#         self.bn = nn.BatchNorm1d(num_hidden_list[-1])
#     def forward(self, inputs,scaling=1):
#         inputs = inputs.flatten(start_dim=1)
#         linearp,linearn = self.linear[0](inputs)
#         linearp,linearn = self.linear[1](linearp,linearn)
#         linearp,linearn = self.linear[2](linearp,linearn)
#         linear = self.linear[3](linearp*scaling,linearn*scaling)
#         return self.bn(linear)
               
def Dataset(batch_size):
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='/home/madhu/.local/TEMP_CODES_FINAL/Datasets', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='/home/madhu/.local/TEMP_CODES_FINAL/Datasets', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader,test_loader

class MLP_Network(torch.nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list,gamma1,gamma2):

        super(MLP_Network, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        layers = []
        layers.append(nn.Linear(in_features, num_hidden_list[0],bias=False))
        for idx in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden_list[idx], num_hidden_list[idx+1],bias=False))

        self.linear = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = inputs.flatten(start_dim=1)
        linear = F.relu(self.linear[0](inputs))
        linear = self.linear[1](linear)
        return linear
def symmetric_quantize(tensor, n_bits=4):
    qmax = 2**(n_bits - 1) - 1  # e.g., 7 for 4-bit signed range [-8, 7]
    # Find scale using absolute max
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    # Quantize and dequantize
    q_tensor = (tensor / scale).round().clamp(-qmax, qmax)
    dq_tensor = q_tensor * scale
    return dq_tensor
    
if __name__ == '__main__':
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp", precision="16-mixed")
    fabric.launch()
    # torch.cuda.set_device('cuda:3')
    print("Torch CUDA available?", torch.cuda.is_available(), flush=True)
    torch.backends.cudnn.benchmark = True
    print(L.__version__)
    torch.manual_seed(42)

    #define epochs
    batch_size = 128
    #define model
    gamma = [1,100,10]
    model = MP_Network(in_features=784, num_layers=2, num_hidden_list=[800, 10], gamma1=gamma[0], gamma2=gamma[1],Temp=True) #- 95.5
    # model = MP_Network_3(in_features=784, num_layers=3, num_hidden_list=[100,100,10], gamma1=gamma[0], gamma2=gamma[1],gamma3=gamma[2],Temp=True,Temp1=True)
    model.load_state_dict(torch.load("/home/madhu/.local/TEMP_CODES_FINAL/Training_codes/Photonics/mnist_kk_0_800.pt"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    train_loader,test_loader,test_loader = Dataset(batch_size)

    accmax = 0  
    epochs = 10
    lr = 1e-4  
    
    # evaluate(fabric, model, test_loader)
    model_T = MLP_Network(in_features=784, num_layers=2, num_hidden_list=[500,10],gamma1=gamma[0], gamma2=gamma[1])
    model_T.load_state_dict(torch.load("/home/madhu/.local/TEMP_CODES_FINAL/Training_codes/Photonics/mnist_mlp_500.pt"))
    for p in model.parameters():
        print(p.data.shape)
    n_bits = 4
    with torch.no_grad():
        for name, param in model.named_parameters():
            quantized = symmetric_quantize(param.data, n_bits=n_bits)
            param.data.copy_(quantized)
            
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    model, optimizer = fabric.setup(model, optimizer)
    start = time.time()
    print('timing started')
    temperature = 5.0  # Temperature for distillation
    alpha = 0.5      # Weight for distillation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5, min_lr=1e-6)
    tl,acc = evaluate(fabric, model, test_loader)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        print("lr",optimizer.param_groups[0]['lr'])
        if(epoch == 10):
            optimizer.param_groups[0]['lr'] = 1e-3
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            
            model_T = model_T.to(fabric.device)
            with torch.no_grad():
                teacher_output = model_T(data.to(fabric.device))  # Teacher outputs (no gradient required)
            # Compute distillation + classification loss
            loss = distillation_loss(output, teacher_output, target.to(fabric.device), temperature, alpha)
            # loss = F.cross_entropy(output.to(fabric.device), target.to(fabric.device))
            fabric.backward(loss)
            optimizer.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    quantized = symmetric_quantize(param.data, n_bits=n_bits)
                    param.data.copy_(quantized)
            optimizer.zero_grad()
        tl,acc = evaluate(fabric, model, test_loader)
        # scheduler.step(acc)
        print("test acc")
        tl,acc1 = evaluate(fabric, model, test_loader)
        if(acc1 > accmax):
            accmax = acc1
            torch.save(model.state_dict(),'/home/madhu/.local/TEMP_CODES_FINAL/Training_codes/Photonics/mnist_kk_0_800_qat.pt') #mnist_kk_0_100 [ 1,100]
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")
    evaluate(fabric, model, test_loader)

        