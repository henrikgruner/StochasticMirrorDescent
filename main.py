
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
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from run import training
from init import initialization_maker

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-q', type=int, default=1,
                    help='The p-norm used as potential')
parser.add_argument('-lr', type=float, default=0.02,
                    help="Learning rate, default = 0.02")
parser.add_argument('-init', type=int, default=1,
                    help='The initialization used')
parser.add_argument('-create_init', type=int, default=0,
                    help="1 if creating initialization is needed, 0 elsewise")
parser.add_argument('-ngpu', type=int, default=2,
                    help='The initialization used')
args = parser.parse_args()


inits = 4

if(args.create_init == 1):
    initialization_maker(inits)

PATH_dict = {
    1: "iteration-1/",
    2: "iteration-2/",
    3: "iteration-3/",
    4: "iteration-4/",
    5: "iteration-5/"
}
init_PATH_dict = {
    1: "Initialization-1",
    2: "Initialization-2",
    3: "Initialization-3",
    4: "Initialization-4",
    5: "Initialization-5"
}

PATH = PATH_dict[args.init]
init_PATH = PATH + init_PATH_dict[args.init]

#######################################################
######################   SETUP  #######################
#######################################################


dataset = "CIFAR10"
batch_size = 128
classes = 10
max_epochs = 5000


device = torch.device("cuda:0" if (
    torch.cuda.is_available() and args.ngpu > 0) else "cpu")

data_sets = data_loader()

print("q: ", args.q, "dataset:", dataset,
      "lr:", args.lr, "iteration: ", args.init)

training(args.q, dataset, data_sets, device,
         args.lr, max_epochs, PATH, batch_size, init_PATH)
