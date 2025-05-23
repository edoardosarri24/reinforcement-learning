import torch.nn as nn
import torch.nn.functional as F
from utility import split_vector

class PolicyNet(nn.Module):
    def __init__(self, layers:list[int]):
        super(PolicyNet, self).__init__()
        modules = []
        layers = split_vector(layers)
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i][0], layers[i][1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-1][0], layers[-1][1]))
        self.modul = nn.Sequential(*modules)
    def forward(self, x):
        return F.softmax(self.modul(x), dim=-1)
    
class StateValueNet(nn.Module):
    def __init__(self, layers:list[int]):
        super(StateValueNet, self).__init__()
        modules = []
        layers = split_vector(layers)
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i][0], layers[i][1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-1][0], 1))
        self.modul = nn.Sequential(*modules)
    def forward(self, x):
        return self.modul(x)