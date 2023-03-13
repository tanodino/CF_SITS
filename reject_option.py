#KOUMBIA
import numpy as np
#import tensorflow as tf
import torch
import torch.nn as nn
import os
import sys
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import time
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import TensorDataset, DataLoader

from model import Noiser, S2Classif



import time
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

def predictionAndCF(noiser, data, device):
    dataCF = []
    noiser.eval()
    for x in data:
        x = x[0]
        x = x.to(device)
        to_add = noiser(x)
        dataCF.append( (x+to_add).cpu().detach().numpy() )
    return np.concatenate(dataCF,axis=0)

def ClfPrediction(model, data, device):
    pred_all = []
    model.eval()
    for x in data:
        x = x[0]
        x = x.to(device)
        pred = model(x)
        pred_all.append((pred.argmax(1)).cpu().detach().numpy())
    return np.concatenate(pred_all,axis=0)

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    year = 2020#int(argv[1])

    torch.manual_seed(0)
    np.random.seed(0)
    print('\n=========\nManual seed activated for reproducibility\n=========')    

    x_test = np.load("x_test_%d.npy"%year)
    x_train = np.load("x_train_%d.npy"%year)
    x_test = np.moveaxis(x_test,(0,1,2),(0,2,1))
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    
    y_test = np.load("y_test_%d.npy"%year)-1.
    y_train = np.load("y_train_%d.npy"%year)-1.

    n_classes = len(np.unique(y_test))

    x_test = extractNDVI(x_test)
    x_train = extractNDVI(x_train)

    n_timestamps = x_test.shape[-1]
    
    x_test_pytorch = torch.Tensor(x_test) # transform to torch tensor
    x_train_pytorch = torch.Tensor(x_train)
    
    test_dataset = TensorDataset(x_test_pytorch) # create your datset  
    train_dataset = TensorDataset(x_train_pytorch)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=2048)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    model.to(device)
    noiser.to(device)
    
    file_path = "model_weights_tempCNN"
    model.load_state_dict(torch.load(file_path))

    path_file_noiser = "noiser_weights_paper"
    noiser.load_state_dict(torch.load(path_file_noiser))

    # Compute Counterfactuals
    CF_train= predictionAndCF(noiser, train_dataloader, device)
    CF_test= predictionAndCF(noiser, test_dataloader, device)

    # Build CF dataloaders
    CF_train = torch.Tensor(CF_train)
    CF_test = torch.Tensor(CF_test)
    CFtrain_dataset = TensorDataset(CF_train)
    CFtest_dataset = TensorDataset(CF_test)
    CFtrain_dataloader = DataLoader(CFtrain_dataset, shuffle=False, batch_size=2048)
    CFtest_dataloader = DataLoader(CFtest_dataset, shuffle=False, batch_size=2048)

    # Compute predictions
    pred_train = ClfPrediction(model, train_dataloader, device)
    pred_CFtrain = ClfPrediction(model, CFtrain_dataloader, device)
    pred_test = ClfPrediction(model, test_dataloader, device)
    pred_CFtest = ClfPrediction(model, CFtest_dataloader, device)

    CF_changed_train = (pred_train != pred_CFtrain)
    pred_correct_train = (pred_train == y_train)
    CF_changed_test = (pred_test != pred_CFtest)
    pred_correct_test = (pred_test == y_test)


    CFchangeRatio_global_train = CF_changed_train.sum()/len(CF_changed_train)
    CFchangeRatio_OnBadPred_train = (~pred_correct_train & CF_changed_train).sum()/((~pred_correct_train).sum())
    CFchangeRatio_OnGoodPred_train = (pred_correct_train & CF_changed_train).sum()/((pred_correct_train).sum())

    CFchangeRatio_global_test = CF_changed_test.sum()/len(CF_changed_test)
    CFchangeRatio_OnBadPred_test = (~pred_correct_test & CF_changed_test).sum()/((~pred_correct_test).sum())
    CFchangeRatio_OnGoodPred_test = (pred_correct_test & CF_changed_test).sum()/((pred_correct_test).sum())

    # Accuracy on:
    # 1) Full data
    acc_full_train = (pred_train == y_train).sum()/len(pred_train)
    acc_full_test = (pred_test == y_test).sum()/len(pred_test)    
    # 2) Only "reliable" samples (not changed by CF)
    acc_reliable_train = (pred_correct_train & ~CF_changed_train).sum()/(~CF_changed_train).sum()
    acc_reliable_test = (pred_correct_test & ~CF_changed_test).sum()/(~CF_changed_test).sum()    
    # 3) Only "unreliable" samples (changed by CF)
    acc_unreliable_train = (pred_correct_train & CF_changed_train).sum()/(CF_changed_train).sum()
    acc_unreliable_test = (pred_correct_test & CF_changed_test).sum()/(CF_changed_test).sum()

    print('Train Accuracy on:\n'
          f'Full data = {acc_full_train:.2f}\n'
          f'Reliable data = {acc_reliable_train:.2f}\n'
          f'Unreliable data = {acc_unreliable_train:.2f}\n')

    print('Test Accuracy on:\n'
          f'Full data = {acc_full_test:.2f}\n'
          f'Reliable data = {acc_reliable_test:.2f}\n'
          f'Unreliable data = {acc_unreliable_test:.2f}\n')    

if __name__ == "__main__":
   main(sys.argv)
