# KOUMBIA
import os
from pathlib import Path
import logging
import sys

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
from cfsits_tools.viz import printConfMatrix, printSomeMetrics
from cfsits_tools.data import loadAllDataNpy, loadSplitNpy, npyData2DataLoader, VALID_SPLITS
from cfsits_tools.viz import produceResults


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
    noiser = Noiser(n_timestamps, .3, shrink=args.shrink)
    noiser.to(getCurrentDevice())
    loadWeights(noiser, args.noiser_name)

    # CF data of the chosen split
    y_pred, y_predCF, dataCF, noiseCF = predictionAndCF(
        model, noiser, dataloader)

    # print confusion matrix for somples correctly predicted
    correct_idx = y_true == y_pred
    cm = confusion_matrix(y_pred[correct_idx], y_predCF[correct_idx])
    printConfMatrix(cm)

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

    # Compute predictions for train data
    y_pred_train = ClfPrediction(
        model, npyData2DataLoader(fullData['train'].X, batch_size=2048))
    logging.info(f"Metrics computed on {args.split} data")
    metricsReport(
        X=X, Xcf=dataCF.squeeze(),
        y_pred_cf=y_predCF,
        X_train=fullData['train'].X, 
        y_pred_train=y_pred_train,
        Xcf_train=trainCF.squeeze(), 
        ifX=fullData['train'].X, k=args.n_neighbors, model=model)


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
    parser.add_argument(
        '--shrink',
        action='store_true',
        default=False,
        help='To be used when the Noiser was trained with shrink=True'
    )    

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        datefmt='%Y/%m/%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                Path(LOG_DIR, 'log.txt'), mode="w"
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )
    main(args)

