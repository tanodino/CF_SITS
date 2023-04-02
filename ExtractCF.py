# KOUMBIA
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from cfsits_tools.cli import getBasicParser
from cfsits_tools.metrics import compactness, metricsReport, plausibility, proximity, stability, validity
from cfsits_tools.model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.utils import ClfPrediction, getCurrentDevice, loadWeights, predictionAndCF, setFreeDevice
from cfsits_tools.viz import extractTransitions, plotSomeCFExamples, plotSomeFailedCFExamples, printConfMatrix, printSomeMetrics, writeChord, writeImages
from cfsits_tools.data import loadAllDataNpy, loadSplitNpy, npyData2DataLoader, VALID_SPLITS


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
    
    
    # Write some CF examples
    output_path = Path(out_path, 'examplesCF_failed')
    # ensure output path exists
    os.makedirs(output_path, exist_ok=True)
    plotSomeFailedCFExamples(y_true, y_pred, y_predCF, noiseCF, dataCF,
                       output_path)

    # Plot chord diagram
    # Analyzing generated perturbations
    mtxHash = extractTransitions(y_pred, y_predCF, noiseCF)
    writeImages(mtxHash, out_path)

    writeChord(cm,
               Path(out_path,
                    f"chord_graph_CF_{split}.pdf"))


def main(args):
    logging.info(f"Data split: {args.split}")

    # Load data
    fullData = loadAllDataNpy(year=args.year, squeeze=True)
    n_classes = fullData['n_classes']
    X, y_true = fullData[args.split]
    # X, y_true = loadSplitNpy(args.split, args.year)
    dataloader = npyData2DataLoader(X, y_true, batch_size=2048)

    # globally set a free gpu as device (if available)
    setFreeDevice()

    # Load model
    logging.info('Loading classifier')
    model = S2Classif(n_class=len(np.unique(y_true)), dropout_rate=.5)
    model.to(getCurrentDevice())
    loadWeights(model, args.model_name)

    # load noiser
    logging.info('Loading noiser')
    n_timestamps = X.shape[-1]
    noiser = Noiser(n_timestamps, .3)
    noiser.to(getCurrentDevice())
    loadWeights(noiser, args.noiser_name)

    # CF data of the chosen split
    y_pred, y_predCF, dataCF, noiseCF = predictionAndCF(
        model, noiser, dataloader)

    if args.do_plots:
        # write plots and tables
        # Prepare output path
        output_folder = Path(
            args.img_path,
            f"{args.model_name}_{args.noiser_name}",
            f"{args.split}")
        os.makedirs(output_folder, exist_ok=True)

        produceResults(args.split, output_folder, y_true,
                    y_pred, y_predCF, dataCF, noiseCF)

        # TSNE
        # plotTSNE(y_true, pred, predCF, noise_CF, output_folder,
        # random_state=args.seed)



    # CF data of train is needed for stability computation
    _, _, trainCF, _ = predictionAndCF(
        model, noiser, npyData2DataLoader(fullData['train'].X, batch_size=2048))

    # reindex train X so it matches the train cf array
    y_pred_train = ClfPrediction(
        model, npyData2DataLoader(fullData['train'].X, batch_size=2048))
    logging.info(f"Metrics computed on {args.split} data")
    metricsReport(
        X=X, Xcf=dataCF.squeeze(), y_cf_pred=y_predCF,
        nnX=fullData['train'].X, nnXcf=trainCF.squeeze(), nny=y_pred_train,
        ifX=fullData['train'].X, k=args.n_neighbors, model=model)

    # def calc_metric(metric_fn, *metric_args, **metric_kwargs):
    #     result = metric_fn(X, dataCF.squeeze(), *metric_args, **metric_kwargs)
    #     return result

    # proximity_avg = np.mean(calc_metric(proximity))
    # logging.info(f"avg proximity: {proximity_avg:0.4f}")



    # stability_avg = np.mean(calc_metric(
    #     stability, k=args.n_neighbors, 
    #     nnX=fullData['train'].X, 
    #     nnXcf=trainCF.squeeze()))
    # logging.info(f"avg stability: {stability_avg:0.4f}")

    # outlier_estimator = IsolationForest(n_estimators=300).fit(X)
    # plausibility_avg = np.mean(calc_metric(
    #     plausibility, estimator=outlier_estimator))
    # logging.info(f"avg plausibility: {plausibility_avg:0.4f}")

    # validity_avg = np.mean(calc_metric(validity, model=model))
    # logging.info(f"avg validity: {validity_avg:0.4f}")


    # for threshold in [1e-2, 1e-3, 1e-4, 1e-8]:
    #     compactness_avg = np.nanmean(calc_metric(
    #         compactness, threshold=threshold))
    #     logging.info(f"avg compactness @ threshold={threshold:0.1e}: {compactness_avg:0.4f}")




if __name__ == "__main__":
    LOG_DIR = os.path.join('logs', os.path.basename(
        os.path.splitext(__file__)[0]))
    os.makedirs(LOG_DIR, exist_ok=True)
    parser = getBasicParser()
    parser.add_argument(
        '--img-path',
        default='img',
        help='Directory in which images are saved'
    )
    parser.add_argument(
        '--split',
        choices=VALID_SPLITS,
        default='test',
        help='Data partition used to compute results.'
    )
    parser.add_argument(
        '-k','--n-neighbors',
        type=int,
        default=5
    )
    parser.add_argument(
        '--do-plots',
        action='store_true',
        default=False,
        help='Runs plotting functions and writes results to IMG_PATH'
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=Path(LOG_DIR, 'log.txt'),
        filemode='w',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=args.log_level)
    logging.info("-"*20 + 'NEW RUN' + "-"*20)
    main(args)
