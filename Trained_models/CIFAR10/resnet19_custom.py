import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = "/home/madhu/.local/TEMP_CODES_FINAL"
layer_dir = os.path.join(current_dir, 'TEMP_Layers')
sys.path.append(layer_dir)
print(layer_dir)
from TEMP_CONV_in import temp_conv_k_opt1s,temp_conv_k_opt1single
from TEMP_FC_in import MPLayer_in_Ks

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion,256,bias=False)
        self.fc2 = nn.Linear(256,num_classes,bias=False)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_resnet19(num_classes=1000):
    return ResNet(BasicBlock, [3, 3, 2], num_classes=num_classes) #1conv,6conv,6conv,4conv,2FC =19 conv layers

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
                        k_act=int(in_planes/2),k_in=K_in,
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
    def __init__(self, block, layers, num_classes=100,temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_in=[0,0,0,0,0,0],K_act=[0,0,0,0,0,0]):
        super(ResNet_TEMPs, self).__init__()
        self.in_channels = 64
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        self.temp6 = temp6
        
        # Initial conv
        if(self.temp1):
            self.conv1 = temp_conv_k_opt1s(
                in_channels=3, out_channels=64, k_act=K_act[0],k_in=K_in[0], kernel_size=3, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 128, layers[0],temp=self.temp2,K_act=K_act[1],K_in=K_in[1])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,temp=self.temp3,K_act=K_act[2],K_in=K_in[2])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,temp=self.temp4,K_act=K_act[3],K_in=K_in[3])

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if(self.temp5):
             self.fc1 = MPLayer_in_Ks(
                inp_node=512 * block.expansion, out_node=256, k_act=K_act[4],k_in=K_in[4]
            )
        else:
             self.fc1 = nn.Linear(512 * block.expansion,256,bias=False)
        if(self.temp6):
             self.fc2 = MPLayer_in_Ks(
                inp_node=256, out_node=num_classes, k_act=K_act[5],k_in=K_in[5]
            )
        else:
             self.fc2 = nn.Linear(256,num_classes,bias=False)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1,temp=False,K_in=1,K_act=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride,temp1=temp,temp2=temp,temp3=temp,K_in=K_in,K_act=K_act))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,temp1=temp,temp2=temp,temp3=temp,K_in=K_in,K_act=K_act))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def resnet19_TEMPs(num_classes=100,temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,K_in=[0,0,0,0,0,0],K_act=[0,0,0,0,0,0]):
    return ResNet_TEMPs(block=BasicBlock_TEMPs, layers=[3, 3, 2], num_classes=num_classes,temp1=temp1,temp2=temp2,temp3=temp3,temp4=temp4,temp5=temp5,temp6=temp6,K_in=K_in,K_act=K_act) #1conv,6conv,6conv,4conv,2FC =19 conv layers