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

from cfsits_tools.model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.utils import computeOrig2pred, predictionAndCF
from cfsits_tools.viz import saveFig
from cfsits_tools.data import loadSplitNpy, extractNDVI

MODEL_DIR = 'models'
DATA_DIR = 'data'

def main(argv):
    year = 2020#int(argv[1])

    x_train, y_train = loadSplitNpy('train', data_path=DATA_DIR, year=year)
    x_valid, y_valid = loadSplitNpy('valid', data_path=DATA_DIR, year=year)
    x_test, y_test = loadSplitNpy('test', data_path=DATA_DIR, year=year)



    n_classes = len(np.unique(y_test))


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
    file_path = os.path.join(MODEL_DIR, file_path)
    model.load_state_dict(torch.load(file_path))

    path_file_noiser = "noiser_weights"
    path_file_noiser = os.path.join(MODEL_DIR, path_file_noiser)
    noiser.load_state_dict(torch.load(path_file_noiser))

    pred, pred_CF, dataCF, _ = predictionAndCF(model, noiser, test_dataloader, device)
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
