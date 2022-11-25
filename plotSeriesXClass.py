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
import chord
from chord import Chord
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def writeImage(class_id, avgProfile, stdProfile, output_folder):    
    output_name = output_folder+"/cl%d.png"%(class_id)
    plt.clf()
    x_axis= np.arange(len(avgProfile))
    plt.xlim([0, len(avgProfile)-1])
    plt.plot(x_axis, avgProfile, color='#CC4F1B')
    plt.fill_between(x_axis, avgProfile-stdProfile, avgProfile+stdProfile, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(output_name )

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)


def main(argv):
    year = 2020
    x_train = np.load("x_train_%d.npy"%year)
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    y_train = np.load("y_train_%d.npy"%year)-1.
    n_classes = len(np.unique(y_train))

    x_train = extractNDVI(x_train)
    x_train = np.squeeze(x_train)


    output_folder = "PerClassProfiles"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for cl_id in np.unique(y_train):
        mean = np.mean( x_train[np.where(y_train == cl_id)[0]], axis=0)
        std = np.std( x_train[np.where(y_train == cl_id)[0]], axis=0)
        writeImage(cl_id, mean, std, output_folder)

if __name__ == "__main__":
   main(sys.argv)
