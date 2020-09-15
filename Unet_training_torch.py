import os
import re
import scipy.misc
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import math
from sklearn.utils import shuffle
import scipy.signal as signal



freq = 10
dim = 16
learning_rate = 0.000001
input_dim = 2
code = 64
d2 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P2-2.npy")
l2 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P2-2.npy")
d3 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P3-2.npy")
l3 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P3-2.npy")
d4 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P4-2.npy")
l4 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P4-2.npy")
d5 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P5-2.npy")
l5 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P5-2.npy")
d7 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P7-2.npy")
l7 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P7-2.npy")
d8 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P8-2.npy")
l8 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P8-2.npy")
d9 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P9-2.npy")
l9 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P9-2.npy")
d13 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P13-2.npy")
l13 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P13-2.npy")
d17 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P17-2.npy")
l17 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P17-2.npy")
d18 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P18-2.npy")
l18 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P18-2.npy")
d19 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P19-2.npy")
l19 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P19-2.npy")
d24 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P24-2.npy")
l24 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P24-2.npy")
d25 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P25-2.npy")
l25 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P25-2.npy")
d26 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EGM_images_P26-2.npy")
l26 = np.load("/content/drive/My Drive/Colab Notebooks/Data/EKG_images_P26-2.npy")

data = d5
labels = l5
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42)




def conv2d(in_channel, out_channel, kshape, stride, padding):
    tmp_layers=[]
    tmp_layers.append(nn.Conv2d(in_channel,out_channel, kshape, stride=stride, padding=padding))
    tmp_layers.append(nn.Tanh())
    return nn.sequential(*tmp_layers)
    
def deconv2d(in_channel, out_channel, kshape, stride, padding):
    tmp_layers=[]
    tmp_layers.append(nn.ConvTranspose2d(in_channel,out_channel, kshape, stride=stride, padding=padding))
    tmp_layers.append(nn.Tanh())
    return nn.sequential(*tmp_layers)
    
def maxpool2d(ksize=2, stride=2):
    return nn.MaxPool2d(ksize, stride=stride)

def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor,mode='bilinear',align_corners=False)

def fullyConnected(input_features,output_features):
    tmp_layers=[]
    tmp_layers.append(nn.Linear(input_features,output_features))
    tmp_layers.append(nn.Tanh())
    return nn.sequential(*tmp_layers)

def fullyConnected2(input_features,output_features):
    return nn.Linear(input_features,output_features, bias=False)



def correlation(x, y):
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x-mx, y-my
    r_num = torch.mean(torch.mul(xm,ym))        
    r_den = torch.std(xm) * torch.std(ym)
    return -1 * r_num / r_den
    
def dropout(p):
    return nn.Dropout(p=p)
    
    
def find_correlation(output, original):
        length = 360
        f = 500
        nseg = 30
        nover = 6
        shape = np.shape(output)
        correlations = np.zeros((12, shape[0]))
        for j in range(shape[0]):
            sample_r = output[j, :, :, :]
            sample_o = original[j, :, :, :]
            signals_r = np.zeros((12, length))
            signals_o = np.zeros((12, length))
            for i in range(int(shape[3]/2)):
                real_r = sample_r[:, :, 2*i]
                real_o = sample_o[:, :, 2 * i]
                imaginary_r = sample_r[:, :, 2*i+1]
                imaginary_o = sample_o[:, :, 2 * i + 1]
                complex_signal_r = real_r + 1j*imaginary_r
                complex_signal_o = real_o + 1j * imaginary_o
                _, X_r = signal.istft(complex_signal_r, f, window="hann", nperseg=nseg, noverlap=nover)
                _, X_o = signal.istft(complex_signal_o, f, window="hann", nperseg=nseg, noverlap=nover)
                correlations[i, j] = np.corrcoef(X_r, X_o, rowvar=False)[0, 1]
                signals_r[i, :] = np.array(X_r)
                signals_o[i, :] = np.array(X_o)
        avg_corr_channel = np.mean(correlations, axis=1)
        avg_corr_overall = np.mean(avg_corr_channel)
        out = avg_corr_overall
        return avg_corr_channel, out


def batch_norm(num_features):
    return nn.BatchNorm2d(num_features,eps=0.001,momentum=0.01)
def batch_norm1d(num_features)
    return nn.BatchNorm2d(num_features,eps=0.001,momentum=0.01)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        keep_prob1=0.1
        keep_prob2=0.5
        keep_prob3=0
        self.lc1=conv2d(input_dim,48,7,1,7//2)
        self.lb1=batch_norm(48)
        self.lp1=maxpool2d()
        self.ldo1=dropout(keep_prob1)
        self.lc2=conv2d(48,96,5,1,5//2)
        self.lb2=batch_norm(96)
        self.lp2=maxpool2d()
        self.ldo2=dropout(keep_prob1)
        #reshape
        self.lfc1=fullyConnected(dim//4*dim//4*96,dim//4*dim//4*60)
        self.lbfc1=batch_norm1d(dim//4*dim//4*60)
        self.ldo3=dropout(keep_prob2)
        self.lfc2=fullyConnected(dim//4*dim//4*60,dim//4*dim//4*40)
        self.lbfc2=batch_norm1d(dim//4*dim//4*40)
        self.ldo4=dropout(keep_prob2)
        self.lfc3=fullyConnected(dim//4*dim//4*40,code)  
        self.lbfc3=batch_norm1d(code)
        
        
        self.lfc4=fullyConnected(code, dim//4*dim//4*40)
        self.lbfc4=batch_norm1d(dim//4*dim//4*40)
        self.ldo5=dropout(keep_prob2)
        self.lfc5=fullyConnected(dim//4*dim//4*40,dim//4*dim//4*60)
        self.lbfc5=batch_norm1d(dim//4*dim//4*60)
        self.ldo6=dropout(keep_prob2)
        self.lfc6=fullyConnected(dim//4*dim//4*60,dim//4*dim//4*96)
        self.lbfc6=batch_norm1d(dim//4*dim//4*96)
        self.ldo7=dropout(keep_prob2)
        #reshape
        self.ldc1=deconv2d(96,48,5,1,5//2)
        self.lbdc1=batch_norm(48)
        self.lup1=upsample()
        self.ldo8=dropout(keep_prob1)
        self.ldc2=deconv2d(48,24,7,1,7//2)
        self.lbdc2=batch_norm(24)
        self.lup2=upsample()
        self.ldo9=dropout(keep_prob1)
        self.lfc7=fullyConnected2(dim*dim*24,dim*dim*24)
        self.loutput=dropout(keep_prob3)
        
    def forward(x):
        x=self.lc1(x)
        x=self.lb1(x)
        x=self.lp1(x)
        x=self.ldo1(x)
        x=self.lc2(x)
        x=self.lb2(x)
        x=self.lp2(x)
        x=self.ldo2(x)
        x=x.view(-1,dim//4*dim//4*96,1,1)
        x=self.lfc1(x)
        x=self.lbfc1(x)
        x=self.ldo3(x)
        x=self.lfc2(x)
        x=self.lbfc2(x)
        x=self.ldo4(x)
        x=self.lfc3(x)
        x=self.lbfc3(x)
        x=self.lfc4(x)
        x=self.lbfc4(x)
        x=self.ldo5(x)
        x=self.lfc5(x)
        x=self.lbfc5(x)
        x=self.ldo6(x)
        x=self.lfc6(x)
        x=self.lbfc6(x)
        x=self.ldo7(x)
        x=x.view(-1,96,dim//4,dim//4)
        x=self.ldc1(x)
        x=self.lbdc1(x)
        x=self.lup1(x)
        x=self.ldo8(x)
        x=self.ldc2(x)
        x=self.lbdc2(x)
        x=self.lup2(x)
        x=self.ldo9(x)
        x=x.view(-1,dim*dim*24,1,1)
        x=self.lfc7(x)
        x=self.loutput(x)
        



# optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr,weight_decay=weight_decay)
# #optimizer = torch.optim.RMSprop(net.parameters(), lr=3e-3, momentum=0.9)
# #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,lr_step, eta_min=1e-7) 
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,lr_decay)
# train_loss_logged=0
        
