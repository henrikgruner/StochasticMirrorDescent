
from collections import OrderedDict
import argparse
from data_sets import data_loader
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


def acc_test(PATH, q, device, dataset="CIFAR10", network="ResNet18"):

    batch_size = 128

    # Classifier for MNIST
    if(network == "Classifier"):
        network = Classifier()
    # ResNet18 for CIFAR10
    if(network == "ResNet18"):
        network = ResNet18()
    print(device)
    state_dict = torch.load(PATH, map_location=torch.device(device))
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if(k[:7] == "module."):
            name = k[7:]  # remove `module.` from data.paralell.nn
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    network.load_state_dict(new_state_dict)

    data_sets = data_loader()
    train, test, channels = data_sets.get_data(dataset, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = SMD(network.parameters(), lr=0.02, q=10)

    test_loss = []
    correct, total = 0, 0
    network.eval()
    with torch.no_grad():
        for j, data in enumerate(test, 0):
            dataiter = iter(test)
            img, labels = dataiter.next()
            img = img.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = network(img)

            correct += (prediction.argmax(1) == labels).sum()
            total += batch_size

            loss = criterion(prediction, labels)
            test_loss.append(loss)

    print('test loss: %.3f accuracy: %.3f' %
          (sum(test_loss) / len(test_loss), (correct.item()/total)))
    return correct.item()/total


device = torch.device("cuda:0" if (
    torch.cuda.is_available()) else "cpu")
print(device)
dataset = "CIFAR10"
q = 10
PATH = 'iteration-3/final_trained_network_q=10'

if __name__ == "__main__":
    acc_test(PATH, q, device, dataset)
