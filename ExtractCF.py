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
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
# import chord
# from chord import Chord
from mpl_chord_diagram import chord_diagram # pip install mpl-chord-diagram
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif


def writeImages(mtxHash, output_folder, dates):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    norm_mtx = np.zeros((len(mtxHash.keys()),len(mtxHash.keys())))
    nonzero_mtx = np.zeros_like(norm_mtx)
    notsmall_mtx = np.zeros_like(norm_mtx)
    sum_mtx = np.zeros_like(norm_mtx)
    pca_mtx70 = np.zeros_like(norm_mtx)
    pca_mtx90 = np.zeros_like(norm_mtx)

    threshold = getPercentile(mtxHash,40)

    for source_k in mtxHash.keys():
        for sink_k in mtxHash[source_k].keys():
            mtx = mtxHash[source_k][sink_k]

            # Average profile (per transition)
            output_path = output_folder + 'avgPattern'
            AvgProfileMean = np.mean(mtx,axis=0)
            AvgProfileStd = np.std(mtx,axis=0)
            writeImageMeanProfile(source_k, sink_k, AvgProfileMean , AvgProfileStd, dates, output_path, mtx.shape[0])

            output_path = output_folder + 'avgPatternAbs'
            AvgProfileMeanAbs = np.mean(np.abs(mtx),axis=0)
            AvgProfileStd = np.std(np.abs(mtx),axis=0)
            writeImageMeanProfile(source_k, sink_k, AvgProfileMeanAbs , AvgProfileStd, dates, output_path, mtx.shape[0])

            # Support histogram per transition
            histogram = (mtx>threshold).sum(axis=0)
            plt.clf()
            plt.bar(range(len(dates)),histogram)
            plt.title(f'Support histogram cl{source_k}->{sink_k} ({mtx.shape[0]} CFs)')
            output_path = output_folder + "histSupport/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_name = output_path + "cl%d_moved2_cl%d.png"%(source_k, sink_k)
            plt.savefig(output_name, bbox_inches = "tight")

            # PCA analysis
            n_components = min(10,mtx.shape[0])
            pca = PCA(n_components=n_components)
            pca.fit(mtx)
            plt.clf()
            explained = np.cumsum(pca.explained_variance_ratio_) * 100
            plt.plot(range(1,n_components+1), explained)
            plt.title(f'PCA analysis cl{source_k}->{sink_k} ({mtx.shape[0]} CFs)')
            plt.ylabel('Explained variance (%)')
            plt.xlabel('Number of components')
            output_path = output_folder + "PCA/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_name = output_path + "cl%d_moved2_cl%d.png"%(source_k, sink_k)
            plt.savefig(output_name, bbox_inches = "tight")

            pca_mtx70[source_k,sink_k] = np.argmax(explained>70) + 1 if any(explained>70) else float('NaN')
            pca_mtx90[source_k,sink_k] = np.argmax(explained>90) + 1 if any(explained>90) else float('NaN')

            # Color-coded transition matrices
            norm_mtx[source_k,sink_k] = np.mean(np.linalg.norm(mtx,axis=1))
            nonzero_mtx[source_k,sink_k] = np.mean(np.count_nonzero(mtx,axis=1))/mtx.shape[1]
            notsmall_mtx[source_k,sink_k] = np.mean(np.sum(mtx>1e-8,axis=1))/mtx.shape[1]
            sum_mtx[source_k,sink_k] = mtx.shape[0]

    output_path = output_folder + 'mtx'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    title = f'PCA components required to explain 70% of variance'
    writeTable(pca_mtx70, title, output_path+"/pcaMtx70.png")

    title = f'PCA components required to explain 90% of variance'
    writeTable(pca_mtx90, title, output_path+"/pcaMtx90.png")

    title = f'Average norm of CF perturbation per class transition'
    writeImageMtx(norm_mtx, title, output_path+"/normMtx.png")

    title = f'Average norm (excluding water class)'
    writeImageMtx(norm_mtx[:-1,:-1], title, output_path+"/normMtx_noWater.png")

    weightedAvg_nonzero = np.sum(sum_mtx * nonzero_mtx) / np.sum(sum_mtx)
    title= ('Average % non-zero entries in CF pertubation per class transition' +
            f'\n Overall average = {weightedAvg_nonzero:.2f}')
    writeImageMtx(nonzero_mtx, title, output_path+"/nonzeroMtx.png", log=False)

    weightedAvg_notsmall = np.sum(sum_mtx * notsmall_mtx) / np.sum(sum_mtx)
    title = ('Average not small entries in CF pertubation per class transition' +
            f'\n Overall average = {weightedAvg_notsmall:.2f}')
    writeImageMtx(notsmall_mtx, title, output_path+"/notSmallMtx.png", log=False)

def writeChord(flux, output_name):
    names = ["CEREALS", "COTTON", "OLEAGINOUS", "GRASSLAND",
            "SHRUBLAND", "FOREST", "B.", "W."]  # "BUILT-UP", "WATER"
    colors = ["#dead0a", "#cfbc8d", "#867025",
            "#69ef73", "#21a52b", "#02650a", "#333435", "#0a5ade"]

    plt.clf()
    chord_diagram(flux, names, gap=0.05, sort="distance", directed=True,
                colors=colors, chordwidth=0.5)
    plt.savefig(output_name, format="pdf", bbox_inches="tight")


def writeImageMtx(mtx, title, output_name, log=True):
    plt.clf()
    if log:
        plt.imshow(mtx, norm=colors.LogNorm())
    else:
        plt.imshow(mtx)
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_name, bbox_inches = "tight")


def writeTable(data, title, output_name):
    fig = plt.figure(figsize=(8,4))
    plt.title(title)
    ax = fig.add_subplot(111, frameon=True, xticks = [], yticks = [])
    img = plt.imshow(data, cmap="YlOrRd") #hot_r
    # plt.colorbar()
    img.set_visible(False)
    tb = plt.table(cellText = data,
        rowLabels = range(data.shape[0]),
        colLabels = range(data.shape[0]),
        loc = 'center',
        cellColours = img.to_rgba(data))
    ax.add_table(tb)
    plt.savefig(output_name, bbox_inches = "tight")

    # fig, ax = plt.subplots()
    # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    # ax.matshow(data, cmap='seismic')
    # for (i, j), z in np.ndenumerate(data):
    #     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


def getPercentile(mtxHash,percent):
    mtx_all = []
    for source_k in mtxHash.keys():
        for sink_k in mtxHash[source_k].keys():
            mtx_all.append(np.abs(mtxHash[source_k][sink_k]))
    return np.percentile(np.concatenate(mtx_all), percent)


def writeImageMeanProfile(source_k, sink_k, avgProfile, stdProfile, x_axis, output_folder, n_CF='?'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_name = output_folder+"/cl%d_moved2_cl%d.png"%(source_k, sink_k)

    plt.clf()
    # x_axis= np.arange(len(avgProfile))
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.plot(x_axis, avgProfile, color='#CC4F1B')
    plt.plot(x_axis, np.zeros(len(x_axis)), color='#000000')
    plt.fill_between(x_axis, avgProfile-stdProfile, avgProfile+stdProfile, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(f'cl{source_k}->{sink_k} ({n_CF} CFs)')
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    plt.savefig(output_name, bbox_inches = "tight")


def extractTransitions(pred, pred_CF, noiseCF):
    hash_source_sink_mtx = {}
    for v in range(len(pred)):
        source_id = pred[v]
        sink_id = pred_CF[v]
        if source_id != sink_id:
            if source_id not in hash_source_sink_mtx.keys():
                hash_source_sink_mtx[source_id] = {}
            if sink_id not in hash_source_sink_mtx[source_id].keys():
                hash_source_sink_mtx[source_id][sink_id] = []
            hash_source_sink_mtx[source_id][sink_id].append( noiseCF[v] )

    for source_id in hash_source_sink_mtx.keys():
        for sink_id in hash_source_sink_mtx[source_id].keys():
            mtx = np.array(hash_source_sink_mtx[source_id][sink_id])
            hash_source_sink_mtx[source_id][sink_id] = mtx

    return hash_source_sink_mtx


def saveFig(i, pred, pred_cf, sample, sampleCF, x_axis, out_path):
    plt.clf()
    # x_axis= np.arange(len(sample))
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
    dates = pd.to_datetime(
            [20200105, 20200125, 20200209, 20200224, 20200305, 20200325,
             20200404, 20200429, 20200514, 20200524, 20200613, 20200623,
             20200628, 20200703, 20200723, 20200921, 20200926, 20201006,
             20201021, 20201031, 20201115, 20201130, 20201215, 20201230],
             format='%Y%m%d')

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

    mtxHash = extractTransitions(pred, pred_CF, noiseCF)

    output_folder = "img/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    writeImages(mtxHash, output_folder, dates)

    writeChord(cm, output_folder + "chord_graph_CF.pdf")

    '''
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
            saveFig(i, pred[i], pred_CF[i], x_test[i], dataCF[i], dates, out_path)
            #exit()
    '''
if __name__ == "__main__":
   main(sys.argv)
