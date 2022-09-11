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
from scipy.stats import entropy
import pickle
import pandas as pd
import time
import sklearn.metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)
print(device)

#experiment parameters
MAX_TRIALS = 1
EPOCHS = 1
EPOCHS_ALPHA = 100
EPOCHS_PSUEDO = 1
data_sizes = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
data_sizes = [0.1]
alpha_grid = [0, 0.1,  0.5, 1]
alpha_grid = [0]
PSUEDO_REPS = 3
psuedo_percent = 0.1


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

kwargs = {'num_workers': 2, 'pin_memory': True}


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
kwargs = {'num_workers': 2, 'pin_memory': True}

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
test_dataset = datasets.__dict__["cifar10".upper()]('../data', train=False, transform=transform_test, download=True)
#test_loader = torch.utils.data.DataLoader(
#    datasets.__dict__["cifar100".upper()]('../data', train=False, transform=transform_test, download=True),
#    batch_size=128, shuffle=True, **kwargs)

class CifarExpertDataset(Dataset):
    def __init__(self, images, targets, expert_fn, expert_labeled, target_labeled = True, indices = None):
        """
        """
        self.images = images
        self.targets = np.array(targets)
        if not(target_labeled):
            self.targets[:] = -1
            

        self.expert_fn = expert_fn
        self.expert_labeled = np.array(expert_labeled)
        self.expert_preds = np.array(expert_fn(None, torch.tensor(targets)))
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
        return torch.tensor(image), label, expert_pred, indice, labeled

    def __len__(self):
        return len(self.targets)




def get_least_confident_points(model, data_loader, budget):
    '''
    based on entropy score get points, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs = model(images)
        batch_size = outputs.size()[0]  
        for i in range(0, batch_size):
            output_i =  outputs.data[i].cpu().numpy()
            entropy_i = entropy(output_i)
            #entropy_i = 1 - max(output_i)
            uncertainty_estimates.append(entropy_i)
            indices_all.append(indices[i].item())
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    actual_indices = indices_all[top_budget_indices]
    uncertainty_estimates = np.array(uncertainty_estimates)
    return actual_indices

def get_most_confident_points(model, data_loader, budget):
    '''
    based on entropy score get points, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs = model(images)
        batch_size = outputs.size()[0]  
        for i in range(0, batch_size):
            output_i =  outputs.data[i].cpu().numpy()
            entropy_i = entropy(output_i)
            #entropy_i = 1 - max(output_i)
            uncertainty_estimates.append(entropy_i)
            indices_all.append(indices[i].item())
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[:budget]
    actual_indices = indices_all[top_budget_indices]
    uncertainty_estimates = np.array(uncertainty_estimates)
    return actual_indices, top_budget_indices



dataset_train = CifarExpertDataset(np.array(train_dataset.dataset.data)[train_dataset.indices], np.array(train_dataset.dataset.targets)[train_dataset.indices], Expert.predict , [1]*len(train_dataset.indices), target_labeled=True)
dataset_val = CifarExpertDataset(np.array(val_dataset.dataset.data)[val_dataset.indices], np.array(val_dataset.dataset.targets)[val_dataset.indices], Expert.predict , [1]*len(val_dataset.indices), target_labeled=True)
dataset_test = CifarExpertDataset(test_dataset.data , test_dataset.targets, Expert.predict , [1]*len(test_dataset.targets), target_labeled=True)

dataLoaderTrain = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderVal = DataLoader(dataset=dataset_val, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataset_test, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)





for trial in range(MAX_TRIALS):
    seperate_results = []
    joint_results = []
    joint_semisupervised_results = []
    truth_semi_results = []
    truth_expert_semi_results = []
    all_data_semi_results = []
    joint = []
    seperate = []
    joint_semisupervised = []
    expert_semi = []
    truth_semi = []
    truth_expert_semi = []
    all_data_semi = []
    for data_size in data_sizes:
        print(f'\n \n datas size {data_size} \n \n')

        all_indices = list(range(len(train_dataset.indices)))
        all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
        all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

        #create 4 sets of data, labeled by both, labeled by expert only, labeled by human only, no labeled
        
        initial_random_set = random.sample(all_indices, math.floor(data_size*len(all_indices)))
        indices_unlabeled = list(set(all_indices) - set(initial_random_set))
        initial_random_expert_only = random.sample(initial_random_set, math.floor(0.66*len(initial_random_set)))
        indices_labeled  = list(set(initial_random_set) - set(initial_random_expert_only))
        initial_random_truth_only = random.sample(initial_random_expert_only, math.floor(0.5*len(initial_random_expert_only)))
        indices_expert_only = list(set(initial_random_expert_only)-set(initial_random_truth_only))
        indices_truth_only = initial_random_truth_only
        
        indices_truth_labeled = list(set(indices_truth_only) | set(indices_labeled))
        indices_truth_unlabeled = list(set(all_indices) - set(indices_truth_labeled))
        indices_expert_unlabeled = list(set(all_indices) - set(indices_expert_only) - set(indices_labeled))

        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), True, indices_labeled)
        dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), False, indices_unlabeled)
        dataset_train_expert_only = CifarExpertDataset(all_data_x[indices_expert_only], all_data_y[indices_expert_only], Expert.predict , [1]*len(indices_expert_only), False, indices_expert_only)
        dataset_train_truth_only = CifarExpertDataset(all_data_x[indices_truth_only], all_data_y[indices_truth_only], Expert.predict , [0]*len(indices_truth_only), True, indices_truth_only)
        dataset_train_truth_labeled = CifarExpertDataset(all_data_x[indices_truth_labeled], all_data_y[indices_truth_labeled], Expert.predict , [0]*len(indices_truth_labeled), True, indices_truth_labeled) #this clears all expert labels but it shouldnt maybe
        dataset_train_truth_unlabeled = CifarExpertDataset(all_data_x[indices_truth_unlabeled], all_data_y[indices_truth_unlabeled], Expert.predict , [0]*len(indices_truth_unlabeled), False, indices_truth_unlabeled) #this too but opposite

        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
        dataLoaderTrainExpertOnly = DataLoader(dataset=dataset_train_expert_only, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
        dataLoaderTrainTruthOnly = DataLoader(dataset=dataset_train_truth_only, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)
        dataLoaderTrainTruthLabeled = DataLoader(dataset=dataset_train_truth_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainTruthUnlabeled = DataLoader(dataset=dataset_train_truth_unlabeled, batch_size=128, shuffle=False,  num_workers=0, pin_memory=True)

        '''

        intial_random_set = random.sample(all_indices, math.floor(data_size*len(all_indices)))
        indices_labeled  = intial_random_set
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))

        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), target_labeled=True, indices = indices_labeled)
        dataset_train_truth_only = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), target_labeled=True, indices = indices_unlabeled)

        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainTruthOnly = DataLoader(dataset=dataset_train_truth_only, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

        '''
        '''
        net_h_params = [10] + [100,100,1000,500]
        net_r_params = [1] + [100,100,1000,500]
        model_2_r = NetSimpleRejector(net_h_params, net_r_params).to(device)
        model_dict = run_reject(model_2_r, 10, Expert.predict, EPOCHS, 1, dataLoaderTrainLabeled, dataLoaderVal, True)
        best_score = 0
        best_model = None
        best_alpha = 1
        for alpha in alpha_grid:
            print(f'alpha {alpha}')
            model_2_r.load_state_dict(model_dict)
            model_dict_alpha = run_reject(model_2_r, 10, Expert.predict, EPOCHS_ALPHA, alpha, dataLoaderTrainLabeled, dataLoaderTrainLabeled, True, 1)
            model_2_r.load_state_dict(model_dict_alpha)
            score = metrics_print(model_2_r, Expert.predict, n_dataset, dataLoaderTrainLabeled)['system accuracy']
            if score >= best_score:
                best_score =  score
                best_model = model_dict_alpha
                best_alpha = alpha



        model_2_r.load_state_dict(best_model)
        joint.append(metrics_print(model_2_r, Expert.predict, n_dataset, dataLoaderTest))
        
        '''

        print(f'\n Joint semi-supervised')
        net_h_params = [10] + [100,100,1000,500]
        net_r_params = [1] + [100,100,1000,500]
        model_2_r = NetSimpleRejector(net_h_params, net_r_params).to(device)
        run_reject_class(model_2_r, EPOCHS, dataLoaderTrainTruthLabeled, dataLoaderTrainTruthLabeled)
        model_dict = copy.deepcopy(model_2_r.state_dict())
        best_score = 0
        best_model = None
        best_alpha = 1
        for alpha in alpha_grid:
            print(f'alpha {alpha}')
            model_2_r.load_state_dict(model_dict)
            model_dict_alpha = run_reject(model_2_r, 10, Expert.predict, EPOCHS_ALPHA, alpha, dataLoaderTrainLabeled, dataLoaderTrainLabeled, True, 1)
            model_2_r.load_state_dict(model_dict_alpha)
            score = metrics_print(model_2_r, Expert.predict, n_dataset, dataLoaderTrainLabeled)['system accuracy']
            if score >= best_score:
                best_score =  score
                best_model = model_dict_alpha
                best_alpha = alpha



        model_2_r.load_state_dict(best_model)
        joint_semisupervised.append(metrics_print(model_2_r, Expert.predict, n_dataset, dataLoaderTest))


        

        print(f' \n Seperate \n')
        # seperate
        model_expert = NetSimple(2,  100,100,1000,500).to(device)
        #expert needs both labels
        run_expert(model_expert,EPOCHS, dataLoaderTrainLabeled, dataLoaderVal)

        print('done training expert model')

        model_class = NetSimple(n_dataset, 100,100,1000,500).to(device)
        #classification model only needs truth labels

        run_reject_class(model_class, EPOCHS, dataLoaderTrainTruthLabeled, dataLoaderVal)
        seperate.append(metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest))

        
        #semi_seperate: 
        print(f' \n semi_seperate option 1\n')
        model_class = NetSimple(n_dataset, 100,100,1000,500).to(device)

        #train classifier using everything we have true labels for
        run_reject_class(model_class, EPOCHS, dataLoaderTrainTruthLabeled, dataLoaderVal)

        #make truth predictions on truth unlabeled data - update later to account for confidence of prediction?
        for rep in range(PSUEDO_REPS):
            actual_points, within_points = get_most_confident_points(model_class, dataLoaderTrainTruthUnlabeled, int(psuedo_percent * len(dataset_train_truth_unlabeled.targets)))
            with torch.no_grad():
                targets = []
                for data in dataLoaderTrainTruthUnlabeled:
                    images, label, expert_pred, data_indices , _ = data
                    expert_pred = expert_pred.long()
                    expert_pred = (expert_pred == label) * 1
                    images, labels, data_indices = images.to(device), expert_pred.to(device), data_indices.to(device)
                    outputs = model_class(images)
                    _, predictions = torch.max(outputs.data, 1) # maybe no .data
                    targets += predictions.tolist()
            
            #fine tune existing classifier_model on newly labeled data
            #dataset_train_truth_unlabeled.targets = targets
            #get most confident points and only use those to create new dataloader
            
            #print(sklearn.metrics.accuracy_score(all_data_y[actual_points], np.array(targets)[within_points]))
            dataset_train_truth_unlabeled_confident = CifarExpertDataset(all_data_x[actual_points], np.array(targets)[within_points], Expert.predict , [0]*len(actual_points), True, list(actual_points))
            dataLoaderTrainTruthUnlabeledConfident = DataLoader(dataset=dataset_train_truth_unlabeled_confident, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
            run_reject_class(model_class, EPOCHS_PSUEDO, dataLoaderTrainTruthUnlabeledConfident, dataLoaderVal)
    
        

        # make truth predictions on expert_only unlabeled data
        for rep in range(PSUEDO_REPS):
            actual_points, within_points = get_most_confident_points(model_class, dataLoaderTrainExpertOnly, int(psuedo_percent * len(dataset_train_expert_only.targets)))
            with torch.no_grad():
                targets = []
                for data in dataLoaderTrainExpertOnly:
                    images, label, expert_pred, data_indices ,_ = data
                    expert_pred = expert_pred.long()
                    expert_pred = (expert_pred == label) * 1
                    images, labels, data_indices = images.to(device), expert_pred.to(device), data_indices.to(device)
                    outputs = model_class(images)
                    _, predictions = torch.max(outputs.data, 1) # maybe no .data
                    targets += predictions.tolist()
                    

            
            
            
            #get most confident points and only use those to create new dataloader

            print(sklearn.metrics.accuracy_score(all_data_y[actual_points], np.array(targets)[within_points]))
            dataset_train_expert_only_confident = CifarExpertDataset(all_data_x[actual_points], np.array(targets)[within_points], Expert.predict , [1]*len(actual_points), True, list(actual_points))
            dataLoaderTrainExpertOnlyConfident = DataLoader(dataset=dataset_train_expert_only_confident, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
            run_expert(model_expert, EPOCHS_PSUEDO, dataLoaderTrainExpertOnly, dataLoaderVal)

        truth_semi.append(metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest))

        #make expert predictions on remaining truth only data
        for rep in range(PSUEDO_REPS):
            actual_points, within_points = get_most_confident_points(model_expert, dataLoaderTrainTruthOnly, int(psuedo_percent * len(dataset_train_truth_only.targets)))
            with torch.no_grad():
                expert_preds = []
                for data in dataLoaderTrainTruthOnly:
                    images, label, expert_pred, data_indices ,_ = data
                    expert_pred = expert_pred.long()
                    expert_pred = (expert_pred == label) * 1
                    images, labels, data_indices = images.to(device), expert_pred.to(device), data_indices.to(device)
                    outputs = model_expert(images)
                    _, predictions = torch.max(outputs.data, 1) # maybe no .data
                    expert_preds += predictions.tolist()
                    

            
            
            
            #get most confident points and only use those to create new dataloader
            dataset_train_truth_only_confident = CifarExpertDataset(all_data_x[actual_points], all_data_y[actual_points], Expert.predict , [0]*len(actual_points), True, list(actual_points))
            #here we are setting the expert predictions to the target where our expert model predicts the expert will be correct (note in reality the expert might not be correct on these instances)
            dataset_train_truth_only_confident.expert_preds[np.array(expert_preds)[within_points].astype(bool)] = all_data_y[actual_points][np.array(expert_preds)[within_points].astype(bool)]
            dataLoaderTrainTruthOnlyConfident = DataLoader(dataset=dataset_train_truth_only_confident, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
            #fine tune existing expert_model on truth only psuedo expert data
            run_expert(model_expert, EPOCHS_PSUEDO, dataLoaderTrainTruthOnlyConfident, dataLoaderVal)
            metrics_print_expert(model_expert, dataLoaderTest)
            metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest)

        truth_expert_semi.append(metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest))

        #make psuedo truth and psuedo expert predictions on completely unlabeled data
        for rep in range(PSUEDO_REPS):
            actual_points_truth, within_points_truth = get_most_confident_points(model_class, dataLoaderTrainUnlabeled, int(psuedo_percent * len(dataset_train_unlabeled.targets)))
            actual_points_expert, within_points_expert = get_most_confident_points(model_expert, dataLoaderTrainUnlabeled, int(psuedo_percent * len(dataset_train_unlabeled.targets)))
            actual_points = list(set(actual_points_truth).intersection(actual_points_expert))
            within_points = list(set(within_points_truth).intersection(within_points_expert))
            with torch.no_grad():
                targets = []
                expert_preds = []
                for data in dataLoaderTrainUnlabeled:
                    images, label, expert_pred, data_indices ,_ = data
                    expert_pred = expert_pred.long()
                    expert_pred = (expert_pred == label) * 1
                    images, labels, data_indices = images.to(device), expert_pred.to(device), data_indices.to(device)
                    outputs_expert = model_expert(images)
                    outputs_class = model_class(images)
                    _, predictions_expert = torch.max(outputs_expert.data, 1) # maybe no .data
                    _, predictions_class = torch.max(outputs_class.data, 1)
                    targets += predictions_class.tolist()
                    expert_preds += predictions_expert.tolist()
                    



            #fine tune expert model using complete pseudo data
            
            

            dataset_train_unlabeled_confident = CifarExpertDataset(all_data_x[actual_points], np.array(targets)[within_points], Expert.predict , [0]*len(actual_points), True, list(actual_points))
            dataLoaderTrainUnlabeledConfident = DataLoader(dataset=dataset_train_unlabeled_confident, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
            dataset_train_unlabeled_confident.expert_preds[np.array(expert_preds)[within_points].astype(bool)]  = all_data_y[actual_points][np.array(expert_preds)[within_points].astype(bool)] 
            run_expert(model_expert,EPOCHS, dataLoaderTrainUnlabeledConfident, dataLoaderVal)



        all_data_semi.append(metrics_print_2step(model_class, model_expert, Expert.predict, 10, dataLoaderTest))

        tempFrame = pd.DataFrame({'seperate': seperate, 'joint': joint, 'joint_semi': joint_semisupervised, 'truth_semi': truth_semi, 
        'truth_expert_semi': truth_expert_semi, 'all_data_semi': all_data_semi})
        tempFrame.to_pickle('temps/run{}_datasize{}_results_20confidence.pkl'.format(trial, data_size))

    
    
    seperate_results.append(seperate)
    truth_semi_results.append(truth_semi)
    truth_expert_semi_results.append(truth_expert_semi)
    all_data_semi_results.append(all_data_semi)
    joint_results.append(joint)
    joint_semisupervised_results.append(joint_semisupervised)

    frame = pd.DataFrame({'seperate': seperate_results, 'joint': joint_results, 'joint_semi': joint_semisupervised_results, 'truth_semi': truth_semi_results,
    'truth_expert_semi': truth_expert_semi_results, 'all_data_semi': all_data_semi_results}, index=data_sizes)

    #frame = pd.DataFrame({'seperate': seperate_results, 'joint': joint_results, 'joint_semi': joint_semisupervised_results})

    frame.to_pickle('results_20confidence_run{}.pkl'.format(trial))
    

    #each instance how likely to be from labeled vs full. probability of being in full/probability of being in subset 
    #every instances in training multiplied by that 
    #get that prob by sampling 
    #

    


