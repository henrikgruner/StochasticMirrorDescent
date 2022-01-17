from collections import OrderedDict
import argparse
from data_loader import data_loader
from SMD import SMD
from models import ResNet18, Classifier
import time
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from acc_test import acc_test
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from run import training


device = torch.device("cuda:0" if (
    torch.cuda.is_available() and 1 > 0) else "cpu")


dataset = "CIFAR10"
batch_size = 128
classes = 10
q = 10

for i in [1, 2, 4, 5]:
    gg = []
    for q in [1, 2, 3, 10]:
        PATH = 'iteration-'+str(i)+'/final_trained_network_q='+str(q)
        network = ResNet18().to(device)
        gg.append(acc_test(PATH, q, device, dataset))

    print("Iteration: " + str(i))
    print(gg)
