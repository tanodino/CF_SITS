import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def saveFig(i, pred, pred_cf, sample, sampleCF, out_path, x_axis=None):
    # from extractCF.py
    plt.clf()
    x_axis = x_axis or np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
    # from generateCF.py
    plt.clf()
    x_axis= np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
    # from DataAugmentation.py
    plt.clf()
    x_axis= np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )

def plotConfusionMatrix(cm, title, filename, vmax=False, figsize=(2.8,2.1)):
    _, ax = plt.subplots(figsize=figsize)
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['Inlier', 'Outlier'])
    cmd_obj.plot(colorbar=False,cmap='Oranges',ax=ax)
    cmd_obj.ax_.set(title= title,
                    xlabel='Counterfactual', 
                    ylabel='Real')
    if vmax:
        for im in ax.get_images(): # set clim manually (to match with ablated model)
            im.set_clim(vmin=1,vmax=vmax)

    # Save figure
    output_folder = 'img/IF_evaluation/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + filename, bbox_inches = "tight")

def plotContingencyMatrix(cm, cm_norm, title, filename, figsize=(2.1,2.1)):
    classes = ['Inlier', 'Outlier']
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = '%.1f%%\n(%d)' % (cm_norm[i, j]*100, cm[i, j])
    cm_norm = pd.DataFrame(cm_norm)
    cm_norm = cm_norm * 100
    cm_norm.index.name = 'Real'
    cm_norm.columns.name = 'Counterfactual'
    _, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')
    plt.title(title)

    sns.heatmap(cm_norm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=False,
                cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Oranges")

    # Save figure
    output_folder = 'img/IF_evaluation/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + filename, bbox_inches = "tight")    
