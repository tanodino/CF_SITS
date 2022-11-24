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


import time
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
    plt.clf()
    x_axis= np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


def computeOrig2pred(orig_label, pred):
    classes = np.unique(orig_label)
    n_classes = len( classes )
    hashOrig2Pred = {}
    for v in classes:
        idx = np.where(orig_label == v)[0]
        hashOrig2Pred[v] = np.bincount( pred[idx], minlength=n_classes )
    return hashOrig2Pred


def predictionAndCF(noiser, data, device):
    dataCF = []
    noiser.eval()
    for x in data:
        x = x[0]
        x = x.to(device)
        to_add = noiser(x)
        dataCF.append( (x+to_add).cpu().detach().numpy() )
    return np.concatenate(dataCF,axis=0)

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    year = 2020#int(argv[1])

    x_test = np.load("x_test_%d.npy"%year)
    x_train = np.load("x_train_%d.npy"%year)
    x_test = np.moveaxis(x_test,(0,1,2),(0,2,1))
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    
    y_test = np.load("y_test_%d.npy"%year)-1.
    y_train = np.load("y_train_%d.npy"%year)-1.

    n_classes = len(np.unique(y_test))

    x_test = extractNDVI(x_test)
    x_train = extractNDVI(x_train)

    clf = RandomForestClassifier()
    clf.fit(np.squeeze(x_train), y_train)
    pred = clf.predict(np.squeeze(x_test))
    fmeasure = f1_score(y_test, pred, average="weighted")
    print("F1 score on original data %f"%fmeasure)

    n_timestamps = x_test.shape[-1]
    
    x_train_pytorch = torch.Tensor(x_train) # transform to torch tensor
    
    train_dataset = TensorDataset(x_train_pytorch) # create your datset

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=2048)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    #model.to(device)
    noiser.to(device)
    
    #file_path = "model_weights_tempCNN"
    #model.load_state_dict(torch.load(file_path))

    path_file_noiser = "noiser_weights_UNI"
    noiser.load_state_dict(torch.load(path_file_noiser))

    dataCF = predictionAndCF(noiser, train_dataloader, device)
    new_train = np.squeeze( np.concatenate([x_train, dataCF],axis=0) )
    new_label = np.concatenate([y_train, y_train],axis=0)

    clf = RandomForestClassifier()
    clf.fit(new_train, new_label)
    pred = clf.predict(np.squeeze(x_test))
    fmeasure = f1_score(y_test, pred, average="weighted")
    print("F1 score on augmented data %f"%fmeasure)

    

if __name__ == "__main__":
   main(sys.argv)
