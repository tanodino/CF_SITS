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
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from torch.utils.data import TensorDataset, DataLoader

import time
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

from sklearn.ensemble import IsolationForest

def generateCF(noiser, loader, device):
    dataCF = []
    noiser.eval()
    for x_batch in loader:
        x_batch = x_batch[0]
        x_batch = x_batch.to(device)
        to_add = noiser(x_batch)
        #print(x_batch.shape)
        #print(to_add.shape)
        dataCF.append( (x_batch+to_add).cpu().detach().numpy() )
    return np.concatenate(dataCF,axis=0)



def applyIF(clf, x_test):
    pred = clf.predict(x_test) + 1
    pred[np.where(pred == 2)] = 1
    pred = pred.astype("int")
    return pred

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    year = 2020
    x_train = np.load("x_train_%d.npy"%year)
    x_test = np.load("x_test_%d.npy"%year)

    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    x_test = np.moveaxis(x_test,(0,1,2),(0,2,1))

    x_train = extractNDVI(x_train)
    x_test = extractNDVI(x_test)

    n_timestamps = x_train.shape[-1]

    noiser = Noiser(n_timestamps, .3)
    noiser = noiser.to(device)
    path_file_noiser = "noiser_weights"
    noiser.load_state_dict(torch.load(path_file_noiser))

    x_test_pytorch = torch.Tensor(x_test)
    test_dataset = TensorDataset(x_test_pytorch)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    dataCF = generateCF(noiser, test_dataloader, device)
    dataCF = np.squeeze(dataCF)


    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    clf = IsolationForest(n_estimators=300).fit(x_train)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)

    pred_orig = applyIF(clf, x_test)
    print(np.bincount(pred_orig))
    
    pred_cf = applyIF(clf, dataCF)
    print(np.bincount(pred_cf))
    
    # Metrics
    print("\nISOLATION FOREST RESULTS:")
    print("\nNMI score: %f"%( normalized_mutual_info_score(pred_orig, pred_cf)) )    
    print("\nAccuracy: %f"%( accuracy_score(pred_orig, pred_cf)) )

    cm = confusion_matrix(pred_orig, pred_cf)
    print("\nConfusion matrix: (isolation forest prediction on original data vs. IF prediction on CF)")
    print("[")
    for row in cm:
        row_str = ",".join( [str(el) for el in row] )
        print("["+row_str+"],")
    print("]")

    exit()

    my_dict = {'ABC': pred, 'DEF': pred_cf}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.savefig("boxplots.jpg" )

    exit()

    '''
    class_id = 0
    idx = np.where(y_train == class_id)[0]
    x_train = x_train[idx]
    y_train = y_train[idx]
    '''
    print(x_train.shape)
    #exit()

    x_train = extractNDVI(x_train)
    x_valid = extractNDVI(x_valid)

    n_timestamps = x_train.shape[-1]
    
    
    
    x_train = torch.Tensor(x_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)
    
    x_valid = torch.Tensor(x_valid) # transform to torch tensor
    y_valid = torch.Tensor(y_valid)

    train_dataset = TensorDataset(x_train, y_train) # create your datset
    #test_dataset = TensorDataset(x_test, y_test) # create your datset
    valid_dataset = TensorDataset(x_valid, y_valid) # create your datset


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)
    #test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model1 = MLPBranch(.5)
    #model = MLPClassif(n_classes, .5)
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    discr = Discr(.2)
    model.to(device)
    noiser.to(device)
    discr.to(device)
    #model1.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    
    optimizer = torch.optim.Adam(noiser.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizerD = torch.optim.Adam(discr.parameters(), lr=0.0001, weight_decay=1e-4)
    
    loss_ce = nn.CrossEntropyLoss().to(device)
    loss_bce = nn.BCELoss().to(device)
    n_epochs = 1000
    #file_path = "model_weights"
    file_path = "model_weights_tempCNN"
    #file_path = "model_weights"
    model.load_state_dict(torch.load(file_path))
    for p in model.parameters():
        p.requires_grad = False

    path_file_noiser = "noiser_weights"
    #trainModel(model, train_dataloader, valid_dataloader, n_epochs, loss_ce, optimizer, file_path, device)
    trainModelNoise(model, noiser, discr, train_dataloader, n_epochs, n_classes, optimizer, optimizerD, loss_bce, device, path_file_noiser)
    
    #print( model.parameters() )
    #exit()
    '''
    import random
    
    for p in model.parameters():
        #if random.uniform(0, 1) > .5:
        p.requires_grad = True
        #print("\tFALSE")
    '''
    


if __name__ == "__main__":
   main(sys.argv)
