import torchvision.datasets as datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch


class data_loader():
    def __init__(self):
        self.channel = 1
        self.dispatch = {"MNIST": self.MNIST, "KMNIST": self.KMNIST,
                         "Fashion_MNIST": self.Fashion_MNIST, "CIFAR10": self.CIFAR10, "EMNIST": self.EMNIST}

    def MNIST(self, batch_size, normalize=False):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))if normalize else NoneTransform()])

        train = datasets.MNIST(
            root='MNIST/data_train', train=True, download=True, transform=transform)

        test = datasets.MNIST(
            root='MNIST/data_test', train=False, download=True, transform=transform)
        return train, test

    def EMNIST(self, batch_size, normalize=False):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))if normalize else NoneTransform()])

        train = datasets.EMNIST(
            root='EMNIST/data_train', split="letters", train=True, download=True, transform=transform)

        test = datasets.EMNIST(
            root='EMNIST/data_test', split="letters", train=False, download=True, transform=transform)
        return train, test

    def KMNIST(self, batch_size, normalize=False):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))if normalize else NoneTransform()])

        train = datasets.KMNIST(
            root='KMNIST/data_train', train=True, download=True, transform=transform)

        test = datasets.KMNIST(
            root='KMNIST/data_test',  train=False, download=True, transform=transform)
        return train, test

    def Fashion_MNIST(self, batch_size, normalize=True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) if normalize else NoneTransform()])

        train = datasets.FashionMNIST(
            root='Fashion_MNIST/data_train', train=True, download=False, transform=transform)

        test = datasets.FashionMNIST(
            root='Fashion_MNIST/data_test', train=False, download=False, transform=transform)

        return train, test

    def CIFAR10(self, batch_size, normalize=True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), ])
        train = datasets.CIFAR10(
            root='CIFAR10/data_train', train=True, download=True, transform=transform_train)

        test = datasets.CIFAR10(
            root='CIFAR10/data_test', train=False, download=True, transform=transform_test)
        self.channel = 3
        return train, test

    def get_visualize(self, dataset, batch_size):
        train, _ = self.dispatch[dataset](batch_size, normalize=False)

        D1 = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=True)
        return D1

    def get_data(self, dataset, batch_size):

        train, test = self.dispatch[dataset](batch_size)

        train = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                            shuffle=True)
        test = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                           shuffle=True)

        return train, test, self.channel
