import os
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from dataset import DataHandler
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, CenterCrop, RandomCrop
import torch
import numpy as np
from utils import noisify



class MNISTData(pl.LightningDataModule):
    def __init__(self, args, path):
        super().__init__()
        self.hparams = args
        self.path = path

        transform = ToTensor()
        dataset = MNIST("mnist", train=True, download=True, transform=transform)
        targets = np.array(dataset.targets)
        # create noise labels
        noise_rate = self.hparams.noise_rate
        noise_type = self.hparams.noise_type
        train_noisy_labels, actual_noise_rate = noisify(nb_classes=10, train_labels=targets, noise_type=noise_type, noise_rate=noise_rate, random_state=0)
        print("Actual noise rate:\t{:.2f}".format(actual_noise_rate))

        with open(os.path.join(self.path, "clean_label.json"), 'w') as f:
            json.dump(targets.tolist(), f)
        with open(os.path.join(self.path, "noisy_label.json"), 'w') as f:
            json.dump(train_noisy_labels.tolist(), f)

        dataset.targets = torch.from_numpy(train_noisy_labels).to(device="cpu")
        self.noisy_trainset = dataset

    def train_dataloader(self):
        dataloader = DataLoader(
            self.noisy_trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = ToTensor()
        dataset = MNIST("mnist", train=False, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            shuffle=False
            # pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def save_train_data(self):
        dataloader = DataLoader(
            self.noisy_trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        trainset_data = None
        trainset_label = None
        for batch_idx, (inputs,targets) in enumerate(dataloader):
            if trainset_data != None:
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
            else:
                trainset_data = inputs
                trainset_label = targets
        
        training_path = os.path.join(self.path, "Training_data")
        if not os.path.exists(training_path):
            os.mkdir(training_path)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))

    def save_test_data(self):
        testset_data = None
        testset_label = None
        for batch_idx, (inputs, targets) in enumerate(self.test_dataloader()):
            if testset_data != None:
                # print(input_list.shape, inputs.shape)
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets

        testing_path = os.path.join(self.path, "Testing_data")
        if not os.path.exists(testing_path):
            os.mkdir(testing_path)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))