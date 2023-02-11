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
            
            # RF_in = 1, Jump_in = 1, stride = 1, Jump_out = Jin*S
            # RF_out = RF_in + (K-1)*Jin 
            # RF_out = 1 + (3-1)*1 = 1+ 2 = 3
           
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32+2-3) + 1 = 32
            
            # RF_in = 3, Jin = 1, S = 1, Jout = Jin*S = 1 
            # RF_Out = 3 + (3-1)*1 = 3 + 2 = 5
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (32+2-3)/2+ 1 = 16
            
            # RF_in = 5, Jin = 1, S = 2, Jout = 1*2 = 2
            # RF_out = 5 + (3-1)*1 = 5 + 2 = 7
            
        # Jin = Jout
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(3,3),padding=1,dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (16+2-5) + 1 = 14
            
            # RF_in = 7, Jin = 2, S = 1, Jout = 2*1 = 2
            # RF_out = 7 + (5-1)*2 = 7+8 = 15

            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (14+2-3) + 1 = 14
            
            # RF_in = 15, Jin = 2, S = 1,Jout = 2 
            # RF_out = 15 + (3-1)*2 = 15 + 4 = 19
           
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (14+2-3)/2+1 = 7
            
            # RF_in = 19, Jin = 2, S = 2,Jout = 2*2 = 4
            # RF_out = 19 + (3-1)*2 = 19 + 4 = 23 
            
        # Jin = Jout
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32),
            # (7-3) + 1 = 5
            
            # RF_in = 23, Jin = 4, S = 1,Jout = 4
            # RF_out = 23 + (3-1)*4 = 23 + 8 = 31
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
            # (5-1) + 1 = 5
            
            # RF_in = 31, Jin = 4, S = 1,Jout = 4
            # RF_out = 31 + (1-1)*4 = 31 + 0 = 31
           
        # Jin = Jout
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (5-3) + 1 = 3
            
            # RF_in = 31, Jin = 4, S = 1,Jout = 4
            # RF_out = 31 + (3-1)*4 = 31 + 8 = 39
            
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3)),
            nn.BatchNorm2d(16),
            # (3-3) + 1 = 1
            
            # RF_in = 39, Jin = 4, S = 1,Jout = 4
            # RF_out = 39 + (3-1)*4 = 39 + 8 = 47
           
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1)))
            # (1-1) + 1 = 1
        
            # RF_in = 47, Jin = 4, S = 1,Jout = 4
            # RF_out = 47 + (1-1)*4 = 47 + 0 = 47

        self.gap = nn.AvgPool2d(1)
        # (1-1) + 1 = 1
        
        # RF_in = 47, Jin = 4, S = 1,Jout = 4
        # RF_out = 47 + (1-1)*4 = 47 + 0 = 47
        
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


