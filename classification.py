import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F

import numpy as np
import math

import lib
from layer import MERAlayer
import optim

lib.torchncon.ncon_check=False
torch.set_default_dtype(torch.float64)

def data_init(batch_size=1):

    train_set = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(16), transforms.ToTensor(), transforms.Lambda(lambda x: torch.reshape(x,(4,4,4,4)))])
    )
    train_size = train_set.train_data.size(0)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(16), transforms.ToTensor(), transforms.Lambda(lambda x: torch.reshape(x,(4,4,4,4)))])
    )

    test_size = test_set.test_data.size(0)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    return train_set, test_set, train_size, test_size

class MeraFC(nn.Module):
    def __init__(self, size=4, chi=8, totlv=3):
        super().__init__()
        self.totlv = totlv
        self.norm_list = []
        
        self.chi = [0] * (totlv + 1)
        self.chi[0] = 4
        for i in range(totlv):
            self.chi[i + 1] = min(chi, self.chi[i]** 3)

        layers = [MERAlayer.SimpleTernary(self.chi[i], self.chi[i+1], (3,1), torch.float64) for i in range(totlv)]
        layers = layers[::-1]
        self.net = torch.nn.ModuleList(layers)

        self.out = nn.Linear(in_features=4**4, out_features=10)
    
    def forward(self,x):
        for layer in self.net:
            x = layer(x)
        y = torch.unsqueeze(torch.flatten(x), 0)
        return self.out(y)


if __name__ == "__main__":
    device = torch.device('cuda:0')
    acc_pre = 0.5
    acc_lambda = 400

    train_loader, test_loader, train_size, test_size = data_init()

    network = MeraFC(chi=4,totlv=3).to(device)
    opt = optim.sgd.SGD(network.parameters(), lr=1.0, momentum=0.9)

    for epoch in range(1000):

        state_tmp = network.state_dict()
        # random.seed(random.random())
        # torch.random.manual_seed(random.random())
        
        network.train()
        
        total_loss = 0

        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            pred = network(images[0])

            loss = F.cross_entropy(pred,labels)
        
            total_loss += loss.item()
        
            opt.zero_grad()
            loss.backward()
            opt.step()

        network.eval()

        with torch.no_grad():
        
            total_correct = 0
            total_num = 0
            
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                
                pred = network(x)
                
                total_correct += pred.argmax(dim=1).eq(label).sum().item()
                total_num += x.size(0)
        
        acc = total_correct / total_num
        acc_dexp = math.exp((acc - acc_pre) * acc_lambda)
        print(total_loss, acc, acc_pre, acc_dexp)