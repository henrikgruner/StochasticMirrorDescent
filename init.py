import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import time
from models import ResNet18

from pathlib import Path


def initialization_maker(n=4):
    '''
    Parameters:
    -----------
    n: int default = 4
        how many initializations
    -----------

    Creates four initializations in four folders
    named iteration-i

    '''
    device = torch.device("cuda:0" if (
        torch.cuda.is_available()) else "cpu")

    directory = 'iteration-'
    PATH = 'Initialization-'
    network = ResNet18().to(device)
    network = torch.nn.DataParallel(network)
    for i in range(n):

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.01)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.uniform_(-0.01, 0.01)
                m.bias.data.uniform_(-0.1, 0.1)
        Path(directory+str(i+1)).mkdir(parents=True, exist_ok=True)
        network.apply(weights_init)
        torch.save(network.state_dict(), directory+str(i+1)+'/'+PATH+str(i+1))


gg = np.array([1, 2, 3, 4])
Path("Test").mkdir(parents=True, exist_ok=True)
if __name__ == '__main__':
    initialization_maker()
