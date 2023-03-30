import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# import chord
# from chord import Chord
from mpl_chord_diagram import chord_diagram # pip install mpl-chord-diagram


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

DATES = pd.to_datetime(
        [20200105, 20200125, 20200209, 20200224, 20200305, 20200325,
            20200404, 20200429, 20200514, 20200524, 20200613, 20200623,
            20200628, 20200703, 20200723, 20200921, 20200926, 20201006,
            20201021, 20201031, 20201115, 20201130, 20201215, 20201230],
            format='%Y%m%d')
NAMES = ["Cereals", "Cotton", "Oleaginous", "Grassland",
        "Shrubland", "Forest", "Built-up", "Water"] 
NAMES_CHORD = ["CEREALS", "COTTON", "OLEAGINOUS", "GRASSLAND",
        "SHRUBLAND", "FOREST", "B.", "W."]  # "BUILT-UP", "WATER"



def saveFig(i, pred, pred_cf, sample, sampleCF, out_path, x_axis=None):
    # from extractCF.py
    plt.clf()
    x_axis = x_axis or np.arange(len(sample))
    plt.plot(x_axis, sample,'b')
    plt.plot(x_axis, sampleCF,'r')
    plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


# def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
#     # from generateCF.py
#     plt.clf()
#     x_axis= np.arange(len(sample))
#     plt.plot(x_axis, sample,'b')
#     plt.plot(x_axis, sampleCF,'r')
#     plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )


# def saveFig(i, pred, pred_cf, sample, sampleCF, out_path):
#     # from DataAugmentation.py
#     plt.clf()
#     x_axis= np.arange(len(sample))
#     plt.plot(x_axis, sample,'b')
#     plt.plot(x_axis, sampleCF,'r')
#     plt.savefig(out_path+"/sample_%d_from_cl_%d_2cl_%d.jpg"%(i, pred, pred_cf) )

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


def writeImages(mtxHash, output_folder):

    norm_mtx = np.zeros((len(mtxHash.keys()),len(mtxHash.keys())))
    nonzero_mtx = np.zeros_like(norm_mtx)
    notsmall_mtx = np.zeros_like(norm_mtx)
    sum_mtx = np.zeros_like(norm_mtx)
    pca_mtx50 = np.zeros_like(norm_mtx)
    pca_mtx70 = np.zeros_like(norm_mtx)
    pca_mtx90 = np.zeros_like(norm_mtx)
    density_nz = []

    threshold = getPercentile(mtxHash,40)

    for source_k in mtxHash.keys():
        for sink_k in mtxHash[source_k].keys():
            mtx = mtxHash[source_k][sink_k]

            # Average profile (per transition)
            output_path = Path(output_folder, 'avgPattern')
            os.makedirs(output_path, exist_ok=True)
            AvgProfileMean = np.mean(mtx,axis=0)
            AvgProfileStd = np.std(mtx,axis=0)
            writeImageMeanProfile(source_k, sink_k, AvgProfileMean , AvgProfileStd,  DATES, output_path, NAMES,  mtx.shape[0])

            output_path = Path(output_folder, 'avgPatternAbs')
            os.makedirs(output_path, exist_ok=True)
            AvgProfileMeanAbs = np.mean(np.abs(mtx),axis=0)
            AvgProfileStd = np.std(np.abs(mtx),axis=0)
            writeImageMeanProfile(source_k, sink_k, AvgProfileMeanAbs , AvgProfileStd,  DATES, output_path, NAMES,  mtx.shape[0])

            # Support histogram per transition
            histogram = (mtx>threshold).sum(axis=0)
            plt.clf()
            plt.bar(range(len(DATES)),histogram)
            plt.title(f'Support histogram {NAMES[source_k]}->{NAMES[sink_k]} ({mtx.shape[0]} CFs)')
            output_path = Path(output_folder, "histSupport")
            os.makedirs(output_path, exist_ok=True)
            output_name = Path(output_path, "cl%d_moved2_cl%d.png"%(source_k, sink_k))
            plt.savefig(output_name, bbox_inches = "tight")

            # PCA analysis
            n_components = min(10,mtx.shape[0])
            pca = PCA(n_components=n_components)
            pca.fit(mtx)
            plt.clf()
            explained = np.cumsum(pca.explained_variance_ratio_) * 100
            plt.plot(range(1,n_components+1), explained)
            plt.title(f'PCA analysis {NAMES[source_k]}->{NAMES[sink_k]} ({mtx.shape[0]} CFs)')
            plt.ylabel('Explained variance (%)')
            plt.xlabel('Number of components')
            output_path = Path(output_folder, "PCA")
            os.makedirs(output_path, exist_ok=True)
            output_name = Path(output_path,
                               "cl%d_moved2_cl%d_var.png"%(source_k, sink_k))
            plt.savefig(output_name, bbox_inches = "tight")
            # writeImageMeanProfile(source_k, sink_k, pca.components_[0], np.zeros_like(pca.components_[0]),  output_path,  mtx.shape[0])

            pca_mtx50[source_k,sink_k] = np.argmax(explained>50) + 1 if any(explained>50) else float('NaN')
            pca_mtx70[source_k,sink_k] = np.argmax(explained>70) + 1 if any(explained>70) else float('NaN')
            pca_mtx90[source_k,sink_k] = np.argmax(explained>90) + 1 if any(explained>90) else float('NaN')

            # Color-coded transition matrices
            norm_mtx[source_k,sink_k] = np.mean(np.linalg.norm(mtx,axis=1))
            nonzero_mtx[source_k,sink_k] = np.mean(np.count_nonzero(mtx,axis=1))/mtx.shape[1]
            notsmall_mtx[source_k,sink_k] = np.mean(np.sum(mtx>1e-8,axis=1))/mtx.shape[1]
            sum_mtx[source_k,sink_k] = mtx.shape[0]

            # Non-zero ratio histogram
            density_nz.append(np.sum(mtx>1e-8,axis=1)/mtx.shape[1])


    output_path = Path(output_folder, 'mtx')
    os.makedirs(output_path, exist_ok=True)

    title = f'Non-zero density'
    plt.clf()
    plt.hist(np.ravel(np.concatenate(density_nz)))
    plt.title(title)
    plt.savefig(Path(output_path, 'nonZero_density.png'), bbox_inches="tight")

    title = f'PCA components required to explain 70% of variance'
    rowLabels = range(pca_mtx90.shape[0])
    colLabels = range(pca_mtx90.shape[0])
    pca_mtx70[pca_mtx70==0] = float('NaN')
    writeTable(pca_mtx70, title, rowLabels, colLabels, 
               Path(output_path,"pcaMtx70.png"))

    title = f'PCA components required to explain 90% of variance'
    pca_mtx90[pca_mtx90==0] = float('NaN')
    writeTable(pca_mtx90, title, rowLabels, colLabels, 
               Path(output_path,"pcaMtx90.png"))

    title = f'Average PCA components'
    idx = (sum_mtx != 0)&(~np.isnan(pca_mtx90))
    pca_avg50 = np.average(pca_mtx50[idx], weights=sum_mtx[idx])
    pca_std50 = np.sqrt(np.average((pca_mtx50[idx]-pca_avg50)**2, weights=sum_mtx[idx]))
    pca_avg70 = np.average(pca_mtx70[idx], weights=sum_mtx[idx])
    pca_std70 = np.sqrt(np.average((pca_mtx70[idx]-pca_avg70)**2, weights=sum_mtx[idx]))
    pca_avg90 = np.average(pca_mtx90[idx], weights=sum_mtx[idx])
    pca_std90 = np.sqrt(np.average((pca_mtx90[idx]-pca_avg90)**2, weights=sum_mtx[idx]))
    cellText = [[f'{pca_avg50:.1f} \u00B1 {pca_std50:.1f}', f'{pca_avg70:.1f} \u00B1 {pca_std70:.1f}', f'{pca_avg90:.1f} \u00B1 {pca_std90:.1f}']]
    colLabels = ['50%', '70%', '90%']
    rowLabels = ['Nb. components']
    writeTable(cellText, title, rowLabels, colLabels, 
               Path(output_path,"pca_avg.png"), colors=False)

    title = f'Average norm of CF perturbation per class transition'
    writeImageMtx(norm_mtx, title, Path(output_path,"normMtx.png"))

    title = f'Average norm (excluding water class)'
    writeImageMtx(norm_mtx[:-1,:-1], title, 
                  Path(output_path,"normMtx_noWater.png"))

    weightedAvg_nonzero = np.sum(sum_mtx * nonzero_mtx) / np.sum(sum_mtx)
    title= ('Average % non-zero entries in CF pertubation per class transition' +
            f'\n Overall average = {weightedAvg_nonzero:.2f}')
    writeImageMtx(nonzero_mtx, title, 
                  Path(output_path,"nonzeroMtx.png"), 
                  log=False)

    weightedAvg_notsmall = np.sum(sum_mtx * notsmall_mtx) / np.sum(sum_mtx)
    title = ('Average not small entries in CF pertubation per class transition' +
            f'\n Overall average = {weightedAvg_notsmall:.2f}')
    writeImageMtx(notsmall_mtx, title, 
                  Path(output_path, "notSmallMtx.png"), 
                  log=False)


def writeChord(flux, output_name):

    colors = ["#dead0a", "#cfbc8d", "#867025",
            "#69ef73", "#21a52b", "#02650a", "#333435", "#0a5ade"]

    plt.clf()
    chord_diagram(flux, NAMES_CHORD, gap=0.05, sort="distance", directed=True,
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


def writeTable(data, title, rowLabels, colLabels, output_name, colors=True):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    cellColours = None
    if colors:
        img = plt.imshow(data, cmap="YlOrRd") #hot_r
        img.set_visible(False)
        cellColours = img.to_rgba(data)
    plt.table(cellText = data,
        rowLabels = rowLabels,
        colLabels = colLabels,
        loc = 'center',
        cellColours = cellColours)
    # ax.set_title(title)
    fig.tight_layout()
    plt.savefig(output_name, bbox_inches = "tight")


def getPercentile(mtxHash,percent):
    mtx_all = []
    for source_k in mtxHash.keys():
        for sink_k in mtxHash[source_k].keys():
            mtx_all.append(np.abs(mtxHash[source_k][sink_k]))
    return np.percentile(np.concatenate(mtx_all), percent)


def writeImageMeanProfile(source_k, sink_k, avgProfile, stdProfile, x_axis, output_folder, names, n_CF='?'):
    os.makedirs(output_folder, exist_ok=True)
    output_name = Path(
        output_folder,"cl%d_moved2_cl%d.png"%(source_k, sink_k))

    plt.clf()
    # x_axis= np.arange(len(avgProfile))
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.plot(x_axis, avgProfile, color='#CC4F1B')
    plt.plot(x_axis, np.zeros(len(x_axis)), color='#000000')
    plt.fill_between(x_axis, avgProfile-stdProfile, avgProfile+stdProfile, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(fr'{names[source_k]} $\rightarrow$ {NAMES[sink_k]} ({n_CF} CFs)')
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


def printConfMatrix(cm):
    print("Confusion matrix (original prediction vs. CF prediction):")    
    print("[")
    for row in cm:
        row_str = ",".join( ['{:5d}'.format(el) for el in row] )
        print("["+row_str+"],")
    print("]")    


def printSomeMetrics(y_true, y_pred, pred_CF, noiseCF):
    print(f"\nFiltering for classifier's correct predictions ({sum(y_pred==y_true)} out of {y_pred.shape[0]})\n")
    correct_idx = (y_pred==y_true)

    number_of_changes = sum(y_pred[correct_idx] != pred_CF[correct_idx])
    print("NUMER OF CHANGED PREDICTIONS : %d over %d, original size is %d"%(number_of_changes, y_pred[correct_idx].shape[0], y_pred.shape[0]))

    print(f'\nNoise avg. L2 norm: {np.linalg.norm(noiseCF, ord=2, axis=1).mean()} (+- {np.linalg.norm(noiseCF, ord=2, axis=1).std()})')
    print(f'Noise avg. L1 norm: {np.linalg.norm(noiseCF, ord=1, axis=1).mean()} (+- {np.linalg.norm(noiseCF, ord=1, axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-2): {np.mean(noiseCF>1e-2,axis=1).mean()} (+- {np.sum(noiseCF>1e-2,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-3): {np.mean(noiseCF>1e-3,axis=1).mean()} (+- {np.sum(noiseCF>1e-3,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-6): {np.mean(noiseCF>1e-6,axis=1).mean()} (+- {np.sum(noiseCF>1e-6,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-8): {np.mean(noiseCF>1e-8,axis=1).mean()} (+- {np.sum(noiseCF>1e-8,axis=1).std()})')
    print(f'Noise avg. L0 norm (>0): {np.mean(noiseCF>0,axis=1).mean()} (+- {np.sum(noiseCF>0,axis=1).std()})')


    print(f'\nNoise avg. L2 norm: {np.linalg.norm(noiseCF[correct_idx], ord=2, axis=1).mean()} (+- {np.linalg.norm(noiseCF[correct_idx], ord=2, axis=1).std()})')
    print(f'Noise avg. L1 norm: {np.linalg.norm(noiseCF[correct_idx], ord=1, axis=1).mean()} (+- {np.linalg.norm(noiseCF[correct_idx], ord=1, axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-2): {np.mean(noiseCF[correct_idx]>1e-2,axis=1).mean()} (+- {np.sum(noiseCF[correct_idx]>1e-2,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-3): {np.mean(noiseCF[correct_idx]>1e-3,axis=1).mean()} (+- {np.sum(noiseCF[correct_idx]>1e-3,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-6): {np.mean(noiseCF[correct_idx]>1e-6,axis=1).mean()} (+- {np.sum(noiseCF[correct_idx]>1e-6,axis=1).std()})')
    print(f'Noise avg. L0 norm (>1e-8): {np.mean(noiseCF[correct_idx]>1e-8,axis=1).mean()} (+- {np.sum(noiseCF[correct_idx]>1e-8,axis=1).std()})')
    print(f'Noise avg. L0 norm (>0): {np.mean(noiseCF[correct_idx]>0,axis=1).mean()} (+- {np.sum(noiseCF[correct_idx]>0,axis=1).std()})')


def plotSomeCFExamples(y_true, y_pred, pred_CF, noiseCF, dataCF,
                         output_path):
    # Plot some CF examples
    correct_idx = (y_pred==y_true)
    sources = [4, 5]
    sinks = [5, 4]
    for source_k, sink_k in zip(sources, sinks):
        CF = dataCF[correct_idx & (y_pred==source_k) & (pred_CF==sink_k)].squeeze()
        x = CF - noiseCF[correct_idx & (y_pred==source_k) & (pred_CF==sink_k)] 
        idx = np.random.randint(CF.shape[0], size=min(46,CF.shape[0]))
        if CF.shape[0] > 2207: # selected indices
            idx = np.append(idx, [1177, 2207, 1729, 1136]) 
        for k in idx:
            output_name = Path(
                output_path,
                f'cl{source_k}_to_cl{sink_k}_{k}.png')
            plt.clf()
            plt.figure(figsize=(3.6,2.7)) #(4,3)
            plt.plot(DATES, x[k], label=f'Real ({NAMES[source_k]})')
            plt.plot(DATES, CF[k], label=f'CF ({NAMES[sink_k]})')
            plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
            plt.ylabel('NDVI')
            plt.legend(frameon=False)
            plt.savefig(output_name, bbox_inches = "tight")
            plt.close()


def plotTSNE(y_true, pred, pred_CF, noiseCF, output_folder, **tsne_kwargs):
    # TSNE
    n_classes = np.unique(y_true).shape[0]
    tsne = TSNE(n_components=2, **tsne_kwargs)
    X_2d = tsne.fit_transform(noiseCF)
    plt.clf()
    plt.figure()
    colors = ["#dead0a", "#cfbc8d", "#867025",
            "#69ef73", "#21a52b", "#02650a", "#333435", "#0a5ade"]    
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
    for source_k in range(n_classes):
        for sink_k in range(n_classes):
            idx = (pred==source_k) & (pred_CF==sink_k)
            if idx.sum() > 300 and (source_k != sink_k): #any(idx)
                label = f'cl{source_k}->{sink_k}'
                plt.scatter(X_2d[idx,0], X_2d[idx,1], color=colors[source_k], alpha=(1-sink_k/len(colors)), label=label)
    plt.legend()
    plt.savefig(Path(output_folder,"TSNE.png"), bbox_inches = "tight")