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

from torch.utils.data import TensorDataset, DataLoader

import time
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model import S2Classif, MLPClassif

#torch.save(model.state_dict(), PATH)

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()

def prediction(model, valid, device):
    labels = []
    pred_tot = []
    model.eval()
    for x, y in valid:
        labels.append( y.cpu().detach().numpy() )
        x = x.to(device)
        pred = model(x)
        pred_tot.append( np.argmax( pred.cpu().detach().numpy() ,axis=1) )
    labels = np.concatenate(labels, axis=0)
    pred_tot = np.concatenate(pred_tot, axis=0)
    return f1_score(labels,pred_tot,average="weighted")

def trainModel(model, train, valid, n_epochs, loss_ce, optimizer, path_file, device):
    model.train()
    best_validation = 0
    for e in range(n_epochs):
        loss_acc = []
        for x_batch, y_batch in train:
            model.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_ce(pred, y_batch.long())
            loss.backward()
            optimizer.step()
            loss_acc.append( loss.cpu().detach().numpy() )
        
        print("epoch %d with loss %f"%(e, np.mean(loss_acc)))
        score_valid = prediction(model, valid, device)
        print("\t val on VALIDATION %f"%score_valid)
        if score_valid > best_validation:
            best_validation = score_valid
            torch.save(model.state_dict(), path_file)
            print("\t\t BEST VALID %f"%score_valid)
        
        sys.stdout.flush()

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    year = 2020#int(argv[1])

    x_train = np.load("x_train_%d.npy"%year)
    x_valid = np.load("x_valid_%d.npy"%year)
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    x_valid = np.moveaxis(x_valid,(0,1,2),(0,2,1))

    y_train = np.load("y_train_%d.npy"%year)-1.
    y_valid = np.load("y_valid_%d.npy"%year)-1.

    n_classes = len(np.unique(y_train))
    print(x_train.shape)

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


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)
    #test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate=.5)
    #model = MLPClassif(n_classes, dropout_rate=.5)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    
    loss_ce = nn.CrossEntropyLoss().to(device)
    n_epochs = 1000
    file_path = "model_weights_tempCNN"
    #file_path = "model_weights"
    trainModel(model, train_dataloader, valid_dataloader, n_epochs, loss_ce, optimizer, file_path, device)    


if __name__ == "__main__":
   main(sys.argv)
