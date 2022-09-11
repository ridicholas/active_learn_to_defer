import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

from neural_network import *
from utils import *
from metrics import *
from training_helpers import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)

k = 5 # number of classes expert can predict
n_dataset = 10
Expert = synth_expert(k, n_dataset)

use_data_aug = False
n_dataset = 10  # cifar-10
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

if use_data_aug:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

if n_dataset == 10:
    dataset = 'cifar10'
elif n_dataset == 100:
    dataset = 'cifar100'

kwargs = {'num_workers': 0, 'pin_memory': True}


train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                        transform=transform_train)
train_size = int(0.90 * len(train_dataset_all))
test_size = len(train_dataset_all) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
#train_loader = torch.utils.data.DataLoader(train_dataset,
#                                           batch_size=128, shuffle=True, **kwargs)
#val_loader = torch.utils.data.DataLoader(val_dataset,
#                                            batch_size=128, shuffle=True, **kwargs)


normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
kwargs = {'num_workers': 1, 'pin_memory': True}

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
test_dataset = datasets.__dict__["cifar10".upper()]('../data', train=False, transform=transform_test, download=True)
#test_loader = torch.utils.data.DataLoader(
#    datasets.__dict__["cifar100".upper()]('../data', train=False, transform=transform_test, download=True),
#    batch_size=128, shuffle=True, **kwargs)


class CifarExpertDataset(Dataset):
    def __init__(self, images, targets, expert_fn, labeled, indices = None):
        """
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_fn = expert_fn
        self.labeled = np.array(labeled)
        self.expert_preds = np.array(expert_fn(None, torch.FloatTensor(targets)))
        for i in range(len(self.expert_preds)):
            if self.labeled[i] == 0:
                self.expert_preds[i] = -1 # not labeled by expert
        if indices != None:
            self.indices = indices
        else:
            self.indices = np.array(list(range(len(self.targets))))
    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        image = transform_test(self.images[index])
        expert_pred = self.expert_preds[index]
        indice = self.indices[index]
        labeled = self.labeled[index]
        return torch.FloatTensor(image), label, expert_pred, indice, labeled

    def __len__(self):
        return len(self.targets)

dataset_train = CifarExpertDataset(np.array(train_dataset.dataset.data)[train_dataset.indices], np.array(train_dataset.dataset.targets)[train_dataset.indices], Expert.predict , [1]*len(train_dataset.indices))
dataset_val = CifarExpertDataset(np.array(val_dataset.dataset.data)[val_dataset.indices], np.array(val_dataset.dataset.targets)[val_dataset.indices], Expert.predict , [1]*len(val_dataset.indices))
dataset_test = CifarExpertDataset(test_dataset.data , test_dataset.targets, Expert.predict , [1]*len(test_dataset.targets))

dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderVal = DataLoader(dataset=dataset_val, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)

all_indices = list(range(len(train_dataset.indices)))
all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

intial_random_set = random.sample(all_indices, 20)
indices_labeled  = intial_random_set
indices_unlabeled= list(set(all_indices) - set(indices_labeled))

dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

MAX_TRIALS = 1
EPOCHS = [60]
EPOCHS_ALPHA = [15]
data_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
data_sizes = [0.2]
alpha_grid = [0.5]
joint_results = []
seperate_results = []
joint_semisupervised_results = []
data_size=0.01

all_indices = list(range(len(train_dataset.indices)))
all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

intial_random_set = random.sample(all_indices, math.floor(data_size*len(all_indices)))
indices_labeled  = intial_random_set
indices_unlabeled= list(set(all_indices) - set(indices_labeled))

dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)


for epoch in EPOCHS:
    for epoch_alpha in EPOCHS_ALPHA:
        net_h_params = [10] + [100,100,1000,500]
        net_r_params = [1] + [100,100,1000,500] 
        model_2_r = NetSimpleRejector(net_h_params, net_r_params).to(device)

        run_reject_class(model_2_r, epoch, dataLoaderTrain, dataLoaderTrainLabeled)
        print('Classifier Test Results, EPOCHS: {}, EPOCHS_ALPHA: {}, results: {}'.format(epoch, epoch_alpha, metrics_print_classifier(model_2_r, dataLoaderTest)))

        run_reject(model_2_r, 10, Expert.predict, epoch, 0.5, dataLoaderTrainLabeled, dataLoaderTrainLabeled, True, 1)
        print('joint model Test Results, EPOCHS: {}, EPOCHS_ALPHA: {}, results: {}'.format(epoch, epoch_alpha, metrics_print(model_2_r, Expert.predict, n_dataset, dataLoaderTest)))       