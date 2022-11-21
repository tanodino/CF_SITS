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


def predictionAndCF(model, noiser, data, device):
    labels = []
    pred_tot = []
    dataCF = []
    pred_CF = []
    model.eval()
    noiser.eval()
    for x in data:
        x = x[0]
        x = x.to(device)
        pred = model(x)
        to_add = noiser(x)
        pred_cf = model(x+to_add)
        dataCF.append( (x+to_add).cpu().detach().numpy() )
        pred_tot.append( np.argmax( pred.cpu().detach().numpy() ,axis=1) )
        pred_CF.append( np.argmax( pred_cf.cpu().detach().numpy() ,axis=1) )
    
    pred_tot = np.concatenate(pred_tot, axis=0)
    pred_CF = np.concatenate(pred_CF, axis=0)
    return pred_tot, pred_CF, np.concatenate(dataCF,axis=0)

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    year = 2020#int(argv[1])

    x_test = np.load("x_test_%d.npy"%year)
    x_test = np.moveaxis(x_test,(0,1,2),(0,2,1))
    
    y_test = np.load("y_test_%d.npy"%year)-1.

    n_classes = len(np.unique(y_test))

    x_test = extractNDVI(x_test)

    n_timestamps = x_test.shape[-1]
    
    
    
    x_test_pytorch = torch.Tensor(x_test) # transform to torch tensor
    
    test_dataset = TensorDataset(x_test_pytorch) # create your datset

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    model.to(device)
    noiser.to(device)
    
    file_path = "model_weights_tempCNN"
    model.load_state_dict(torch.load(file_path))

    path_file_noiser = "noiser_weights"
    noiser.load_state_dict(torch.load(path_file_noiser))

    pred, pred_CF, dataCF = predictionAndCF(model, noiser, test_dataloader, device)
    idx = np.where(pred == y_test)[0]


    pred = pred[idx]
    pred_CF = pred_CF[idx]
    dataCF = dataCF[idx]
    x_test = x_test[idx]

    hashOrig2Pred = computeOrig2pred(pred, pred_CF)
    for k in hashOrig2Pred.keys():
        print("\t ",k," -> ",hashOrig2Pred[k])
    print("========")
    exit()
    out_path = "CF"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    x_test = np.squeeze(x_test)
    dataCF = np.squeeze(dataCF)

    for i in range(len(pred)):
        if pred[i] != pred_CF[i]:
            print("%d out of %d"%(i,len(pred)))
            saveFig(i, pred[i], pred_CF[i], x_test[i], dataCF[i], out_path)
            #exit()

if __name__ == "__main__":
   main(sys.argv)
