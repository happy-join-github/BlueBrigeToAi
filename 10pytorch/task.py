import torchvision

train = torchvision.datasets.MNIST(root='.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root='.', train=True, transform=torchvision.transforms.ToTensor(), download=True)

import torch

train_load = torch.utils.data.DataLoader(dateset=train, batch_siza=64, shuffle=True)
test_load = torch.utils.data.DataLoader(dateset=test, batch_siza=64, shuffle=False)

import torch.nn as nn
import torch.nn.modules as F

class Net(nn.modules):
    def __init__(self):
        super(Net, self).__init__()
    
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,10)
        
    def forword(self,x):
        x = F.ReLU(self.fc1(x))
        x = F.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net()

print(model(torch.randn(1, 784)))
