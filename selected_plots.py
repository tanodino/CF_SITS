#%%
import numpy as np
from matplotlib import pyplot as plt
from cfsits_tools.data import loadAllDataNpy
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import getCurrentDevice, loadWeights
from cfsits_tools.utils import loadPreds


split='test'
year=2020
model_name = 'model_weights_tempCNN'

fullData = loadAllDataNpy(year=year, squeeze=True)
n_classes = fullData['n_classes']
X, y_true = fullData[split]
y_pred = loadPreds(model_name, split, year)
X_train, y_train = fullData['train']


model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
model.to(getCurrentDevice())
loadWeights(model, model_name)
#%% Load NG data
# NG 
from cfsits_tools.data import npyData2DataLoader
from cfsits_tools.utils import ClfPrediction
from competitor_NG import loadCFs


cfs_path = "logs/competitor_NG/cf_data"
NG_dataCF = loadCFs(cfs_path, split, use_cam=True)
# compute model preds for dataCF
NG_y_predCF = ClfPrediction(
    model, npyData2DataLoader(NG_dataCF, batch_size=4))
NG_noiseCF = NG_dataCF - X
# Ours
from cfsits_tools.model import Noiser
from cfsits_tools.utils import predictionAndCF

noiser_name = 'noiser_weights_paper'

# load noiser
n_timestamps = X.shape[-1]
noiser = Noiser(n_timestamps, .3)
noiser.to(getCurrentDevice())
loadWeights(noiser, noiser_name)

_, y_predCF, dataCF, noiseCF = predictionAndCF(
        model, noiser, npyData2DataLoader(X, y_true, batch_size=4))

dataCF=dataCF.reshape([-1,24])
## comparing and filtering
is_correct = y_pred == y_true
is_valid_cf = y_predCF != y_pred
same_class_cf = y_predCF == NG_y_predCF
np.sum(same_class_cf), np.sum(~same_class_cf)
sum(is_correct & is_valid_cf)

#%% Load NN data
# kNN CFs
from cfsits_tools.data import npyData2DataLoader
from cfsits_tools.utils import ClfPrediction, loadPkl
from competitor_knn import predictCfSamplesFromData
from competitor_knn import KNeighborsCounterfactual

# load CF model
cfModel = loadPkl('logs/competitor_knn/noiser_1nn.pkl')


NN_dataCF, NN_idxCF, NN_dstClass = \
    predictCfSamplesFromData(X, n_classes, y_pred, cfModel)

NN_y_predCF = ClfPrediction(
    model, npyData2DataLoader(NN_dataCF, batch_size=2048))

#%%

from cfsits_tools.viz import DATES, NAMES

candidate_idx = [
    # shown in the paper:
    19201, 15658, 26033,
    # other interesting cases:
    # 256, 860,1414,1419,1435,2511,2911,2912,5222,
    # 5825,6393,8564,9275,9292,9307,11758,15544,15658,19201,
    # 7606,19500,
    # 20190, 
    # 8989,10197,25885,25964,25999,26033,26047
]

for k in candidate_idx:
    source_k =y_pred[k]
    sink_k = y_predCF[k]
    NG_sink_k = NG_y_predCF[k]
    k_NN_NG = np.where((NN_idxCF==k) & (NN_y_predCF == NG_sink_k))[0][0].squeeze()
    NN_sink_k_NG = int(NN_y_predCF[k_NN_NG])
    skipNN=False
    try:
        k_NN = np.where((NN_idxCF==k) & (NN_y_predCF == sink_k))[0][0].squeeze()
        NN_sink_k = int(NN_y_predCF[k_NN])
    except IndexError:
        skipNN=True

    fig, axs = plt.subplots(1, 3, figsize=(3*3.4,2.9), sharey=True)
    # plot the real data in all axs
    for ax in axs:
        ax.plot(DATES, X[k], label=f'Real ({NAMES[source_k]})')
    # Ax 0 - ours
    axs[0].plot(DATES, dataCF[k], label=f'CFE4SITS ({NAMES[sink_k]})')
    axs[0].set_ylabel('NDVI')
    # Ax 1 - NG
    axs[1].plot(DATES, NG_dataCF[k], label=f'NG({NAMES[NG_sink_k]})')
    # Ax2 - kNN
    if skipNN:
        axs[2].plot(DATES, NN_dataCF[k_NN_NG], 
                    label=f'k-NNC ({NAMES[NN_sink_k_NG]})')
    else:
        axs[2].plot(DATES, NN_dataCF[k_NN], label=f'k-NNC ({NAMES[NN_sink_k]})')
    # format fig presentation
    for ax in axs:
        # Rotates X-Axis Ticks by 45-degrees
        ax.tick_params(axis='x', labelrotation=45) 
        ax.legend(frameon=False)
    # arrange layout and save
    plt.tight_layout(pad=0.1,w_pad=-0.7)
    plt.savefig(f'img/comparison_sample_{k}.png')
    plt.close()