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

from cfsits_tools.model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.utils import computeOrig2pred, generateCF
from cfsits_tools.data import loadSplitNpy, extractNDVI

MODEL_DIR = 'models'
DATA_DIR = 'data'

def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
    plt.clf()
    x_axis= np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


def main(argv):
    year = 2020#int(argv[1])

    x_train, y_train = loadSplitNpy('train', data_path=DATA_DIR, year=year)
    x_valid, y_valid = loadSplitNpy('valid', data_path=DATA_DIR, year=year)
    x_test, y_test = loadSplitNpy('test', data_path=DATA_DIR, year=year)


    n_classes = len(np.unique(y_test))


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
    # file_path = os.path.join(MODEL_DIR, file_path)
    #model.load_state_dict(torch.load(file_path))

    path_file_noiser = "noiser_weights_UNI"
    path_file_noiser = os.path.join(MODEL_DIR, path_file_noiser)
    noiser.load_state_dict(torch.load(path_file_noiser))

    dataCF = generateCF(noiser, train_dataloader, device)
    new_train = np.squeeze( np.concatenate([x_train, dataCF],axis=0) )
    new_label = np.concatenate([y_train, y_train],axis=0)

    clf = RandomForestClassifier()
    clf.fit(new_train, new_label)
    pred = clf.predict(np.squeeze(x_test))
    fmeasure = f1_score(y_test, pred, average="weighted")
    print("F1 score on augmented data %f"%fmeasure)

    

if __name__ == "__main__":
   main(sys.argv)
