import os
import sys
import time
import math
# Lets do DataLoading, misclassifications, plots from here
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torchvision
import torchvision.transforms as transforms
# from model import Net 
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm 
import torch.backends.cudnn as cudnn 
import torch.nn as nn
import torch.optim 
from torch.utils.data import Dataset,DataLoader

import random

cv2.setNumThreads(0)
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize
cv2.ocl.setUseOpenCL(False)

class Draw:
    def plotings(image_set):
        images = image_set
        img = images
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        print(len(image_set),"images are plotted")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 128
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

class loader:
    def load_data():
        train_transforms = A.Compose([
                A.HorizontalFlip(p=0.49),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.45),
                A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734,p=round(random.uniform(0.35,0.52),2)),
                A.Normalize(mean = (0.5, 0.5, 0.5),std = (0.5, 0.5, 0.5)),
                A.pytorch.ToTensorV2()])


        trainset = Cifar10SearchDataset(root='./data', train=True,
                                                download=True, transform=train_transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                                  shuffle=True,**args().kwargs)

        transform = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, **args().kwargs)
        return trainloader,testloader

class class_accuracy:
    def rate(testloader,model,classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
              images, labels = data
              data, target = images.cuda(), labels.cuda()
              outputs = model(data)
              _, predicted = torch.max(outputs, 1)
              c = (predicted == target).squeeze()
              for i in range(4):
                  label = target[i]
                  class_correct[label] += c[i].item()
                  class_total[label] += 1
        dicts = {}
        for i in range(10):
            val = 100*class_correct[i]/class_total[i]
            dicts[classes[i]] = val 
        return dicts 