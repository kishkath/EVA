import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from model import Net

model = Net()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
if device=='cuda':
  print("GPU Is AVAILABLE")


# Defining loss & optimizers
criterion = nn.CrossEntropyLoss()

# Training & testing

from tqdm import tqdm 

train_losses = []
test_losses = []
train_acc = []
test_acc = []
class Performance:
    def __init__(self):

        self.test_loss = 0

    def train(self,model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = criterion(y_pred, target)
            train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} train-Accuracy={100*correct/processed:0.2f}')
            train_acc.append(100*correct/processed)

    def test(self,model, device, test_loader):
        model.eval()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                self.test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.test_loss = self.test_loss/len(test_loader.dataset)
        test_losses.append(self.test_loss)

        print('\nTest set: Average loss: {:.4f}, val-Accuracy: {}/{} ({:.2f}%)\n'.format(
            self.test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        test_acc.append(100. * correct / len(test_loader.dataset))
def scores():
  return train_acc,train_losses,test_acc,test_losses
