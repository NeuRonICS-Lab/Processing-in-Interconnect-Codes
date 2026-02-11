from torch import nn
from torch.nn import functional as F
import sys
import os
import torch

current_dir = ""  #path of the current directory
layer_dir = os.path.join(current_dir, 'PI2_Layers')
sys.path.append(layer_dir)
print(layer_dir)
from PI2_CONV_in import temp_conv_k_opt1single,temp_conv_k_opt1s
from PI2_FC_in import MPLayer_in_Ksingle,MPLayer_in_Ks


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)

class BasicBlock_TEMP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,temp1=False,temp2=False,temp3=False,K_act=0):
        super(BasicBlock_TEMP, self).__init__()
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        
        if(temp1):
            self.conv1 = temp_conv_k_opt1single(
            in_channels=in_planes, out_channels=planes, k_act=K_act, kernel_size=3, dilation=1, padding=1, stride=stride
            )
        else:
            self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        if(temp2):
            self.conv2 = temp_conv_k_opt1single(
                in_channels=planes, out_channels=planes, k_act=K_act, kernel_size=3, dilation=1, padding=1, stride=1
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if(temp3):
                self.shortcut = nn.Sequential(
                    temp_conv_k_opt1single(
                        in_channels=in_planes,
                        out_channels=self.expansion * planes,
                        k_act=int(in_planes/2),
                        kernel_size=1,
                        stride=stride
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet_TEMP(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100,temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_act=0):
        super(ResNet_TEMP, self).__init__()
        self.in_planes = 64
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        self.temp6 = temp6

        if(self.temp1):
            self.conv1 = temp_conv_k_opt1single(
                in_channels=3, out_channels=64, k_act=K_act[0], kernel_size=3, stride=1, padding=1, 
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1,temp=self.temp2,K_act=K_act[1])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,temp=self.temp3,K_act=K_act[2]) #100,100
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,temp=self.temp4,K_act=K_act[3]) #256 256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,temp=self.temp5,K_act=K_act[4])
        if(temp6):
            self.linear = MPLayer_in_Ksingle(
                inp_node=512 * block.expansion, out_node=num_classes, k_act=K_act[5]
            )
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride,temp=False,K_in=1,K_act=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if(temp):
                layers.append(block(self.in_planes, planes,s,temp1=temp,temp2=temp,temp3=temp,K_act=K_act))
            else:
                layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = 10*self.linear(out)
        return out

def ResNet18_TEMP(temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_act=1,num_classes=10):
    return ResNet_TEMP(BasicBlock_TEMP, [2, 2, 2, 2],temp1=temp1,temp2=temp2,temp3=temp3,temp4=temp4,temp5=temp5,temp6=temp6,K_act=K_act,num_classes=num_classes)

class BasicBlock_TEMPs(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,temp1=False,temp2=False,temp3=False,K_in=0,K_act=0):
        super(BasicBlock_TEMPs, self).__init__()
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        
        if(temp1):
            self.conv1 = temp_conv_k_opt1s(
            in_channels=in_planes, out_channels=planes, k_act=K_act,k_in=K_in, kernel_size=3, dilation=1, padding=1, stride=stride
            )
        else:
            self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        if(temp2):
            self.conv2 = temp_conv_k_opt1s(
                in_channels=planes, out_channels=planes, k_act=K_act,k_in=K_in, kernel_size=3, dilation=1, padding=1, stride=1
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if(temp3):
                self.shortcut = nn.Sequential(
                    temp_conv_k_opt1single(
                        in_channels=in_planes,
                        out_channels=self.expansion * planes,
                        k_act=int(in_planes/2),
                        kernel_size=1,
                        stride=stride
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet_TEMPs(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100,temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_in=0,K_act=0):
        super(ResNet_TEMPs, self).__init__()
        self.in_planes = 64
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        self.temp6 = temp6

        if(self.temp1):
            self.conv1 = temp_conv_k_opt1single(
                in_channels=3, out_channels=64, k_act=K_act[0],k_in=K_in[0], kernel_size=3, stride=1, padding=1, 
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1,temp=self.temp2,K_act=K_act[1],K_in=K_in[1])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,temp=self.temp3,K_act=K_act[2],K_in=K_in[2]) #100,100
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,temp=self.temp4,K_act=K_act[3],K_in=K_in[3]) #256 256
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,temp=self.temp5,K_act=K_act[4],K_in=K_in[4])
        if(temp6):
            self.linear = MPLayer_in_Ks(
                inp_node=512 * block.expansion, out_node=num_classes, k_act=K_act[5],k_in=K_in[5]
            )
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride,temp=False,K_in=1,K_act=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if(temp):
                layers.append(block(self.in_planes, planes,s,temp1=temp,temp2=temp,temp3=temp,K_act=K_act,K_in=K_in))
            else:
                layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = 10*self.linear(out)
        return out

def ResNet18_TEMPs(temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_in=1,K_act=1,num_classes=10):
    return ResNet_TEMPs(BasicBlock_TEMPs, [2, 2, 2, 2],temp1=temp1,temp2=temp2,temp3=temp3,temp4=temp4,temp5=temp5,temp6=temp6,K_in=K_in,K_act=K_act,num_classes=num_classes)
