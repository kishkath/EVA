import torch.nn as nn
import torch.nn.functional as F

# C1-C2-C3-C4-Output
# No Max Pooling
# Kernel (3X3), stride (2,2)
# RF must be 44
# one of depthwise
# one of dilated convs
# use GAP - can add FC after gap 
# Albumentations: Flip, shiftscaleRotate,Cutout
# 85% accuracy
# 200K parameters

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (32+2-3) + 1 = 32
            # Jump_in = 1, Jump_out = Jump_in*s
            # RF = 1
            # RF_out = Rin + (K-1) * Jin
            
            ## Jump_out = 1*1
            # RF_in = 1*1 = 1
            # RF_out = 1 + (3-1)*1 = 1 + (2*1) = 3 
            
            # RF_in = 1, RF_out = 3, Jin = 1
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32-3) + 1 = 30
            
            # Jump_out = Jump_in*s = 1*1 = 1
            # RF_in = 3 
            # RF_out = RF_in + (K-1) * Jin = 3 + (3-1)*1 = 3 + 2 = 5 
            
            # RF_in = 3, RF_out = 5, Jin = 1
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (30-3)+ 1 = 28
            
            # RF_in = 5 
            # Jump_out = Jin*s = 1 
            # RF_out = RF_in + (K-1)*Jin = 5 + (3-1)*1 = 5 + 2 = 7
            
        # RF_in = 5, RF_out = 7, Jin = 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(3,3),dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (28-5)/1+ 1 = 24
            
            # RF_in = 7 
            # Jump_out = Jin*s = 1 
            # RF_out = RF_in + (K-1)*Jin = 7 + (5-1)*1 = 7 + 4 = 11 
           
            # RF_in = 7, RF_out = 11, Jin = 1
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (24-2) + 1 = 22
            
            # RF_in = 11 
            # Jump_out = Jin*s = 1
            # RF_out = RF_in + (K-1)*Jin = 11 + (3-1)*1 = 11 + 2 = 13 
            
            # RF_in = 11,RF_out=13,Jin=1
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (22+2-3)/2+1 = 11
            
            # RF_in = 13 
            # Jump_out = Jin*s = 1*2 = 2
            # RF_out = RF_in + (K-1)*Jin = 13 + (3-1)*1 = 13 + 2 = 15
            
            # RF_in = 13, RF_out = 15, Jin = Jump_out = 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, groups=8),
            nn.BatchNorm2d(128),
            # 9
            
            # RF_in = 15 
            # Jump_out = Jin*s = 2*1 = 2 
            # RF_out = RF_in + (K-1)*Jin = 15 + (3-1)*2 = 15 + 4 = 19
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 9
            
            # RF_in = 19 
            # Jump_out = 2 
            # RF_out = RF_in+(K-1)*Jin = 19 + (1-1)*2 = 19
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (9-3)/1 + 1 = 7
            
            # RF_in = 19
            # Jump_out = 2
            # RF_out = RF_in + (K-1)*Jin = 19 + (3-1)* 2 = 19 + 4 = 23
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3)),
            # (7-3) + 1 = 5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # RF_in = 23 
            # Jump_out = 2 
            # RF_out = RF_in + (K-1)*Jn = 23 + (3-1)*2 = 23 + 4 = 27
            
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            # (5-1) + 1 = 5
            
            # RF_in = 27 
            # Jump_out = 2 
            # RF_out = RF_in + (K-1)*Jin = 27 + (1-1)*2 = 27
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(3,3)))
           
            # RF_in = 27 
            # Jump_out = 2
            # RF_out = RF_in + (K-1)*Jin = 27 + (3-1)*2 = 31
        self.gap = nn.AvgPool2d(3)
        
        # RF_n = 31, Jump_out = 2, RF_out = RF_n + (K-1)*Jin = 31 + (3-1)*2 = 31 + 4 = 35
        
        
        # RF : 35
        self.dropout = nn.Dropout2d(0.1)
        
      
        
    def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.dropout(x)
            x = self.conv4(x)
            x = self.gap(x)
            x = x.view(-1,10)
            return F.log_softmax(x,dim=1)


