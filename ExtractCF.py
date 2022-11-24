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


import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

def writeImage(source_k, sink_k, avgProfile, stdProfile, output_folder):    
    output_name = output_folder+"/cl%d_moved2_cl%d.png"%(source_k, sink_k)
    plt.clf()
    x_axis= np.arange(len(avgProfile))
    plt.xlim([0, len(avgProfile)-1])
    plt.plot(x_axis, avgProfile, color='#CC4F1B')
    plt.plot(x_axis, np.zeros(len(x_axis)), color='#000000')
    plt.fill_between(x_axis, avgProfile-stdProfile, avgProfile+stdProfile, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(output_name )


def extractProfile(pred, pred_CF, noiseCF):
    hash_source_sink_avg = {}
    hash_source_sink_std = {}
    for v in range(len(pred)):
        source_id = pred[v]
        sink_id = pred_CF[v]
        if source_id != sink_id:
            if source_id not in hash_source_sink_avg.keys():
                hash_source_sink_avg[source_id] = {}
            if sink_id not in hash_source_sink_avg[source_id].keys():
                hash_source_sink_avg[source_id][sink_id] = []
            hash_source_sink_avg[source_id][sink_id].append( noiseCF[v] )
    
    for source_id in hash_source_sink_avg.keys():
        for sink_id in hash_source_sink_avg[source_id].keys():
            temp = hash_source_sink_avg[source_id][sink_id]
            temp = np.array(temp)
            hash_source_sink_avg[source_id][sink_id] = np.mean(np.array(temp),axis=0)
            if source_id not in hash_source_sink_std.keys():
                hash_source_sink_std[source_id] = {}
            hash_source_sink_std[source_id][sink_id] = np.std(np.array(temp),axis=0)
                
    return hash_source_sink_avg, hash_source_sink_std   



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
    noise_CF = []
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
        noise_CF.append( np.squeeze( to_add.cpu().detach().numpy() ) )
    pred_tot = np.concatenate(pred_tot, axis=0)
    pred_CF = np.concatenate(pred_CF, axis=0)
    return pred_tot, pred_CF, np.concatenate(dataCF,axis=0), np.concatenate(noise_CF,axis=0)

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

    n_timestamps = x_train.shape[-1]
    
    x_train_pytorch = torch.Tensor(x_train) # transform to torch tensor
    
    train_dataset = TensorDataset(x_train_pytorch) # create your datset

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=2048)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    model.to(device)
    noiser.to(device)
    
    file_path = "model_weights_tempCNN"
    model.load_state_dict(torch.load(file_path))

    #path_file_noiser = "noiser_weights"
    path_file_noiser = "noiser_weights_UNI"
    noiser.load_state_dict(torch.load(path_file_noiser))

    pred, pred_CF, dataCF, noiseCF = predictionAndCF(model, noiser, train_dataloader, device)
    
    cm = confusion_matrix(pred, pred_CF)
    print("[")
    for row in cm:
        row_str = ",".join( [str(el) for el in row] )
        print("["+row_str+"],")
    print("]")

    output_folder = "avgPattern"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    avgProfileHash, stdProfileHash = extractProfile(pred, pred_CF, noiseCF)
    for source_k in avgProfileHash.keys():
        for sink_k in avgProfileHash[source_k].keys():
            writeImage(source_k, sink_k, avgProfileHash[source_k][sink_k], stdProfileHash[source_k][sink_k], output_folder)


    exit()
    
    
    
    idx = np.where(pred == y_test)[0]
    pred = pred[idx]
    pred_CF = pred_CF[idx]
    dataCF = dataCF[idx]
    x_test = x_test[idx]

    hashOrig2Pred = computeOrig2pred(pred, pred_CF)
    for k in hashOrig2Pred.keys():
        print("\t ",k," -> ",hashOrig2Pred[k])
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
