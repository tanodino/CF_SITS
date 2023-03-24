# KOUMBIA
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


from cfsits_tools.cli import getBasicParser
from cfsits_tools.model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.utils import getDevice, loadWeights, predictionAndCF
from cfsits_tools.viz import extractTransitions, plotSomeCFExamples, printConfMatrix, printSomeMetrics, writeChord, writeImages
from cfsits_tools.data import loadSplitNpy, npyData2DataLoader, VALID_SPLITS


def produceResults(split, out_path, y_true, y_pred, y_predCF, dataCF, noiseCF):
    # print some metrics
    print(f"{split.upper()} DATA INFO")
    printSomeMetrics(y_true, y_pred, y_predCF, noiseCF)

    # print confusion matrix for somples correctly predicted
    correct_idx = y_true == y_pred
    cm = confusion_matrix(y_pred[correct_idx], y_predCF[correct_idx])
    printConfMatrix(cm)

    # Write some CF examples
    output_path = Path(out_path, 'examplesCF')
    # ensure output path exists
    os.makedirs(output_path, exist_ok=True)
    plotSomeCFExamples(y_true, y_pred, y_predCF, noiseCF, dataCF,
                       output_path)

    # Plot chord diagram
    # Analyzing generated perturbations
    mtxHash = extractTransitions(y_pred, y_predCF, noiseCF)
    writeImages(mtxHash, out_path)

    writeChord(cm,
               Path(out_path,
                    f"chord_graph_CF_{split}.pdf"))


def main(args):

    X, y_true = loadSplitNpy(args.split, args.year)
    dataloader = npyData2DataLoader(X, y_true, batch_size=2048)

    # Load model
    model = S2Classif(n_class=len(np.unique(y_true)), dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)

    # load noiser
    n_timestamps = X.shape[-1]
    noiser = Noiser(n_timestamps, .3)
    noiser.to(getDevice())
    loadWeights(noiser, args.noiser_name)

    # CF data of the chosen split
    y_pred, y_predCF, dataCF, noiseCF = predictionAndCF(
        model, noiser, dataloader, getDevice())

    # write plots and tables
    # Prepare output path
    output_folder = Path(
        args.out_path,
        f"{args.model_name}_{args.noiser_name}",
        f"{args.split}")
    os.makedirs(output_folder, exist_ok=True)

    produceResults(args.split, output_folder, y_true,
                   y_pred, y_predCF, dataCF, noiseCF)

    # TSNE
    # plotTSNE(y_true, pred, predCF, noise_CF, output_folder,
    # random_state=args.seed)

    '''
    exit()

    # Other
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
            saveFig(i, pred[i], pred_CF[i], x_test[i], dataCF[i], out_path, dates)
            #exit()
    '''


if __name__ == "__main__":
    parser = getBasicParser()
    parser.add_argument(
        '--out-path',
        default='img'
    )
    parser.add_argument(
        '--split',
        choices=VALID_SPLITS,
        default='test'
    )

    args = parser.parse_args()

    main(args)
