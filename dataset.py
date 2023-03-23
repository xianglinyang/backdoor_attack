from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def load_dataset(dataset_name):
    if dataset_name.lower() == "mnist":
        train_dataset = torchvision.datasets.MNIST("data/mnist", train=True, download=True)
        test_dataset = torchvision.datasets.MNIST("data/mnist", train=False, download=True)
        train_data = train_dataset.data.numpy()
        train_labels = train_dataset.targets
        test_data = test_dataset.data.numpy()
        test_labels = test_dataset.targets
    elif dataset_name.lower() == "fmnist":
        pass
    elif dataset_name.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root="data/cifar10", train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, download=True)
        train_data = train_dataset.data
        train_labels = np.array(train_dataset.targets)
        test_data = test_dataset.data
        test_labels = np.array(test_dataset.targets)
    else:
        raise NotImplementedError

    return train_data, train_labels, test_data, test_labels


def build_transform(dataset_name):
    if dataset_name == "MNIST":
        # train_transform = transforms.ToTensor()
        # test_transform = transforms.ToTensor()
        mean, std = (0.5,), (0.5,)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dataset_name == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise NotImplementedError

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    #     ])
    # mean = torch.as_tensor(mean)
    # std = torch.as_tensor(std)
    # detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return train_transform, test_transform


def save_dataset(dataloader, save_path, is_train):
        data = None
        label = None
        for (inputs,targets) in dataloader:
            if data != None:
                data = torch.cat((data, inputs), 0)
                label = torch.cat((label, targets), 0)
            else:
                data = inputs
                label = targets
        if is_train:
            torch.save(data, os.path.join(save_path, "training_dataset_data.pth"))
            torch.save(label, os.path.join(save_path, "training_dataset_label.pth"))
        else:
            torch.save(data, os.path.join(save_path, "testing_dataset_data.pth"))
            torch.save(label, os.path.join(save_path, "testing_dataset_label.pth"))


def save_sprite(data, path):
    for i in range(len(data)):
        img = Image.fromarray(data[i]).convert("RGB")
        img.save(os.path.join(path, "{}.png".format(i)))


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # x = Image.fromarray(x, mode="L")
        x = Image.fromarray(x)
        
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)