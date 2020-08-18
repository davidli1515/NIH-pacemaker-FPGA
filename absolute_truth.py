import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import sys 
from tqdm import tqdm
from tqdm._utils import _term_move_up



M1=48
N1=2
C1=16
H1=22
K1=7

M2=96
N2=48
C2=8
H2=12
K2=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d1=self.conv2d(N1,M1,K1)
        self.max1=nn.MaxPool2d(2, stride=2)
    def conv2d(self,input_channel, output_channel, k):
        conv2d_layers=[]
        conv2d_layers.append(nn.Conv2d(input_channel,output_channel,k, padding=k//2,bias=True))
        conv2d_layers.append(nn.Tanh())
        #conv2d_layers.append(nn.BatchNorm2d(output_channel))
        return nn.Sequential(*conv2d_layers)
    def forward(self, x):
        x =self.conv2d1(x)
        x =self.max1(x)
        return x
        
        
test_x=np.random.rand(1,N1,C1,C1)
test_x=test_x.astype(np.float32)
with open("test_x.inc",'w') as f:
    for i in range(N1):
        for j in range(C1):
            for k in range(C1):
                f.write(str(test_x[0][i][j][k])+"\n\r")

torch.manual_seed(123)           
net1=Net()

# for name, param in net1.named_parameters():
    # if param.requires_grad:
        # print (name, param.data)
print(net1.conv2d1[0].bias.size())
test_weight=net1.conv2d1[0].weight.data
test_bias=net1.conv2d1[0].bias.data
with open("test_weight.inc","w") as f:
    for i in range(M1):
        for j in range(N1):
            for k in range(K1):
                for l in range(K1):
                    f.write(str(float(test_weight[i][j][k][l]))+"\n\r")   
    for i in range(M1):
        f.write(str(float(test_bias[i]))+"\n\r")
        
        
tensor_test_x= torch.tensor(test_x)
tensor_output= net1(tensor_test_x)  
test_output=tensor_output.data

with open("test_output.inc","w") as f:
    for i in range(M1):
        for j in range(C2):
            for k in range(C2):
                f.write(str(float(test_output[0][i][j][k]))+"\n\r")   


        
        