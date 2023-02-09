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
           
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32-3) + 1 = 30
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (30-3)+ 1 = 28
            
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(3,3),dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (28-5)/1+ 1 = 24
           
          
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (24-2) + 1 = 22
           
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (22+2-3)/2+1 = 11
            
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, groups=8),
            nn.BatchNorm2d(128),
            # 9
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
            # 9
           
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3)),
            # (7-3) + 1 = 5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            # (5-1) + 1 = 5
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1)))

        self.gap = nn.AvgPool2d(1)
        
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


