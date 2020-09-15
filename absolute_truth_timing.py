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
import time
torch.manual_seed(0)    
np.random.seed(0)   

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


M3=4*4*60
C3=1
H3=1
K3=1

N4=4*4*40

N5=64
N6=4*4*40
N7=4*4*60
N8=4*4*96

N9=96
M9=48
K9=5
C9=4

N10=48
M10=96
K10=7
C10=8
def custom_fc(weight,bias,input0):
    output0=np.zeros([1,weight.shape[0],1,1])
    for out_ch in range(weight.shape[0]):
        for in_ch in range(weight.shape[1]):
            output0[0,out_ch,0,0]+=input0[0,in_ch,0,0]*weight[out_ch,in_ch,0,0]
    for out_ch in range(weight.shape[0]):
            output0[0,out_ch,0,0]=np.tanh(bias[out_ch]+output0[0,out_ch,0,0])
    return output0
def custom_deconv(weight, input0):
    return None
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d1=self.conv2d(N1,M1,K1)
        self.conv2d2=self.conv2d(N2,M2,K2)
        self.f1=self.conv2d(M2*C2//2*C2//2,M3,1)
        self.f2=self.conv2d(M3,N4,1)
        self.f3=self.conv2d(N4,N5,1)
        self.f4=self.conv2d(N5,N6,1)
        self.f5=self.conv2d(N6,N7,1)
        self.f6=self.conv2d(N7,N8,1)
        self.deconv2d1=self.deconv2d(N9,M9,K9)
        self.deconv2d2=self.deconv2d(N10,M10,K10)
        self.max1=nn.MaxPool2d(2, stride=2)
        self.upsample= nn.Upsample(scale_factor=2)
        self.f7=self.conv2d(M10*C10*C10*4,M10*C10*C10*4,1)
    def deconv2d(self,input_channel,output_channel,k):
        deconv2d_layers=[]
        deconv2d_layers.append(nn.ConvTranspose2d(input_channel, output_channel, k, 1,k//2))
        deconv2d_layers.append(nn.Tanh())
        return nn.Sequential(*deconv2d_layers)
    def conv2d(self,input_channel, output_channel, k):
        conv2d_layers=[]
        conv2d_layers.append(nn.Conv2d(input_channel,output_channel,k, padding=k//2,bias=True))
        conv2d_layers.append(nn.Tanh())
        #conv2d_layers.append(nn.BatchNorm2d(output_channel))
        return nn.Sequential(*conv2d_layers)
    def forward(self, x):
        x =self.conv2d1(x)
        x =self.max1(x)
        x =self.conv2d2(x)
        x =self.max1(x)
        #print('input before')
        #print(x.data)
        x =x.view(1,M2*C2//2*C2//2,1,1)
        #print('input afterwards')
        #print(x.data)
        x =self.f1(x)
        x =self.f2(x)
        x =self.f3(x)
        x =self.f4(x)
        x =self.f5(x)
        x =self.f6(x)
        x =x.view(1,N9,C9,C9)
        x =self.deconv2d1(x)
        x =self.upsample(x)
        x =self.deconv2d2(x)
        x =self.upsample(x)
        x =x.view(1,M10*C10*C10*4,1,1)
        x =self.f7(x)
        #x =self.f3(x)
        return x
        
        
test_x=np.random.rand(1,N1,C1,C1)
test_x=test_x.astype(np.float32)
net1=Net()

# for name, param in net1.named_parameters():
    # if param.requires_grad:
        # print (name, param.data)
        
tensor_test_x= torch.tensor(test_x)
start=time.time()
tensor_output= net1(tensor_test_x)  
print(time.time()-start)
test_output=tensor_output.data

#with open("test_output.inc","w") as f:
#    for i in range(N4):
#        for j in range(1):
#            for k in range(1):
#                f.write(str(float(test_output[0][i][j][k]))+"\n\r")   


#test_x=np.random.rand(1,M2*C2//2*C2//2,1,1).astype('float32')
#output0_from_torch=net1.f1(torch.tensor(test_x))
#output0=custom_fc(test_weight3,test_bias3,test_x)

#print(output0_from_torch.data[0,11,0,0])
#print(output0[0,11,0,0])
