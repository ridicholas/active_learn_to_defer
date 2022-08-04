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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, images, targets, expert_fn, expert_labeled, target_labeled = None, indices = None):
        """
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_fn = expert_fn
        self.expert_labeled = np.array(expert_labeled)
        self.expert_preds = np.array(expert_fn(None, torch.FloatTensor(targets)))
        for i in range(len(self.expert_preds)):
            if self.expert_labeled[i] == 0:
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
        labeled = self.expert_labeled[index]
        return torch.FloatTensor(image), label, expert_pred, indice, labeled

    def __len__(self):
        return len(self.targets)

dataset_train = CifarExpertDataset(np.array(train_dataset.dataset.data)[train_dataset.indices], np.array(train_dataset.dataset.targets)[train_dataset.indices], Expert.predict , [1]*len(train_dataset.indices))
dataset_val = CifarExpertDataset(np.array(val_dataset.dataset.data)[val_dataset.indices], np.array(val_dataset.dataset.targets)[val_dataset.indices], Expert.predict , [1]*len(val_dataset.indices))
dataset_test = CifarExpertDataset(test_dataset.data , test_dataset.targets, Expert.predict , [1]*len(test_dataset.targets))

dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderVal = DataLoader(dataset=dataset_val, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)

MAX_TRIALS = 1
EPOCHS = 1
EPOCHS_ALPHA = 10
#data_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
data_sizes = [0.01]
#alpha_grid = [0, 0.1,  0.5, 1]
alpha_grid = [0]
seperate_results = []
joint_semisupervised_results = []
expert_semi_results = []

for trial in range(MAX_TRIALS):
    joint = []
    seperate = []
    joint_semisupervised = []
    expert_semi = []
    for data_size in data_sizes:
        print(f'\n \n datas size {data_size} \n \n')

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


       
        
        print(f' \n Seperate \n')
        # seperate
        model_expert = NetSimple(2,  100,100,1000,500).to(device)
        run_expert(model_expert,EPOCHS, dataLoaderTrainLabeled, dataLoaderVal)

        print('done training expert model')

        model_class = NetSimple(n_dataset, 100,100,1000,500).to(device)

        run_reject_class(model_class, EPOCHS, dataLoaderTrain, dataLoaderVal)
        seperate.append(metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest)['system accuracy'])

        print(f' \n semi_seperate \n')
        model_expert_semi = NetSimple(2, 100, 100, 1000, 500).to(device)
        # make predictions on unlabeled data - update later to account for confidence of prediction? 
        with torch.no_grad():
            for data in dataLoaderTrainUnlabeled:
                images, label, expert_pred, data_indices ,_ = data
                expert_pred = expert_pred.long()
                expert_pred = (expert_pred == label) *1
                images, labels = images.to(device), expert_pred.to(device)
                outputs = model_expert(images)
                _, predictions = torch.max(outputs.data, 1) # maybe no .data
                dataset_train_unlabeled.expert_preds[data_indices] = predictions
        
        #fine tune existing expert_model on newly labeled data
        model_expert_semi = run_expert(model_expert,EPOCHS, dataLoaderTrainUnlabeled, dataLoaderVal)
        expert_semi.append(metrics_print_2step(model_class, model_expert_semi, Expert.predict, 10, dataLoaderTest)['system accuracy'])
                
        



    
    expert_semi_results.append(expert_semi)
    seperate_results.append(seperate)
    


