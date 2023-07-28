from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np

def get_dataset(train_transforms, test_transforms):
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataloader(batch_size, num_workers):

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    train_transforms, test_transforms = get_data_transform()

    trainset, testset = get_dataset(train_transforms, test_transforms)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader