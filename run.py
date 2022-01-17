import torchvision.datasets as datasets
import torchvision

import torchvision.transforms as transforms
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import time
from models import ResNet18, Classifier
from SMD import SMD
from data_loader import data_loader
import argparse


def training(q, dataset, data_sets, device, lr, max_epochs, PATH, batch_size=128, init_PATH=None):
    '''
    """
        Parameters
        ----------
        q : int
            The name of the animal
        dataset : str
            The sound the animal makes
        data_sets : int, optional
            The number of legs the animal (default is 4)
        device : int
            The name of the animal
        dataset : str
            The sound the animal makes
        data_sets : int, optional
            The number of legs the animal (default is 4)

        """
    '''
    train, test, channels = data_sets.get_data(dataset, batch_size)

    training_loss, validationLoss = [], []

    network = ResNet18().to(device)
    network = torch.nn.DataParallel(network)
    criterion = nn.CrossEntropyLoss().cuda()

    # Stochastic Mirror Descent optmizier with the q-norm
    # to the power of q as potential function: 1/q ||x||_q^q
    optimizer = SMD(network.parameters(), lr=lr, q=q)

    if init_PATH is not None:
        network.load_state_dict(torch.load(init_PATH))

    #######################################################
    ###############   CNN TRAINING   ######################
    #######################################################

    print("size of dataset ("+str(dataset)+"): ", len(train) * batch_size)
    print("========== Network ==========")
    print(network)
    #print(summary(network, (1, 28, 28)))
    print("========== Training ==========")
    print("Starting the classifier-training:")

    for e in range(max_epochs):
        epoch_loss = []
        correct, total = 0, 0

        for i, data in enumerate(train, 0):
            dataiter = iter(train)
            img, labels = dataiter.next()
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            prediction = network(img)
            correct += (prediction.argmax(1) == labels).sum()
            total += batch_size

            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)

        print('[%i/%i] loss: %.5f accuracy: %.5f q: %.1f ' %
              (e, max_epochs, sum(epoch_loss) / len(epoch_loss), (correct.item()/total), q))
        if(correct.item()/total == 1):
            print(correct.item()-total)
            break

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

    torch.save(network.state_dict(), PATH +
               "/final_trained_network_q=" + str(q))
