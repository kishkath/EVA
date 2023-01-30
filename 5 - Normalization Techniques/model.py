from __future__ import print_function
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets,transforms


class Setup(nn.Module):
  def __init__(self,layer):
    super(Setup,self).__init__()

    if layer=="BN":
      self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU())
        
      self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels=8,out_channels=10,kernel_size=(3,3)),
          nn.BatchNorm2d(10),
          nn.ReLU()) 
    
      self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.BatchNorm2d(16),
          nn.ReLU()
      )
      self.convblock5 = nn.Sequential(
          nn.Conv2d(in_channels=16,out_channels=20,kernel_size=(3,3)),
          nn.BatchNorm2d(20),
          nn.ReLU()
      )
      self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1)),
        nn.BatchNorm2d(10),
        nn.ReLU())
      self.convblock7 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.BatchNorm2d(16),
          nn.ReLU())
      self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(3,3)))
   
    
    elif layer=="GN":
      self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding=1),
        nn.GroupNorm(2,8),
        nn.ReLU())
        
      self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels=8,out_channels=10,kernel_size=(3,3)),
          nn.GroupNorm(2,10),
          nn.ReLU()) 
  
      self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.GroupNorm(2,16),
          nn.ReLU()
      )
      self.convblock5 = nn.Sequential(
          nn.Conv2d(in_channels=16,out_channels=20,kernel_size=(3,3)),
          nn.GroupNorm(2,20),
          nn.ReLU()
      )
      self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1)),
        nn.GroupNorm(2,10),
        nn.ReLU())
      self.convblock7 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.GroupNorm(2,16),
          nn.ReLU())

    elif layer=="LN":
      self.convblock1 = nn.Sequential(
          nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding=1),
          nn.GroupNorm(1,8),
          nn.ReLU())
      self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels=8,out_channels=10,kernel_size=(3,3)),
          nn.GroupNorm(1,10),
          nn.ReLU()) 
    
      self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.GroupNorm(1,16),
          nn.ReLU()
      )
      self.convblock5 = nn.Sequential(
          nn.Conv2d(in_channels=16,out_channels=20,kernel_size=(3,3)),
          nn.GroupNorm(1,20),
          nn.ReLU()
      )
      self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1)),
        nn.GroupNorm(1,10),
        nn.ReLU())
      self.convblock7 = nn.Sequential(
          nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3)),
          nn.GroupNorm(1,16),
          nn.ReLU())
   
   
    self.pool1 = nn.MaxPool2d(2,2)
    self.dropout1 = nn.Dropout(0.1)

  def forward(self,x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock5(x)
    x = self.dropout1(x)
    x = self.convblock6(x)
    x = self.pool1(x)
    x = self.convblock7(x)

    x = self.convblock8(x)
    x = x.view(-1,10)
    return F.log_softmax(x,dim=-1)
