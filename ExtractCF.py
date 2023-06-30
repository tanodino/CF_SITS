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
from cfsits_tools import log
from cfsits_tools import cli
from cfsits_tools import utils
from cfsits_tools import viz
from cfsits_tools import metrics

from cfsits_tools.cli import getBasicParser
from cfsits_tools.metrics import compactness, metricsReport, plausibility, proximity, stability, validity
from cfsits_tools.model import Inception, MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.utils import ClfPrediction, getCurrentDevice, loadWeights, predictionAndCF, setFreeDevice
from cfsits_tools.viz import printConfMatrix, printSomeMetrics
from cfsits_tools.data import DEFAULT_DATA_DIR, load_UCR_dataset, loadAllDataNpy, loadSplitNpy, npyData2DataLoader, VALID_SPLITS
from cfsits_tools.viz import produceResults


MODEL_DIR = utils.DEFAULT_MODEL_DIR
DATA_DIR = DEFAULT_DATA_DIR



def launchInference(args):
    # Load training data and 
    # Load data from requested split
    logger.info(f"Loading dataset {args.dataset} - {args.split} split")
    if args.dataset == "koumbia":
        x_train, y_train = loadSplitNpy(
            'train', data_path=DATA_DIR, year=args.year, squeeze=True)
        X, y_true = loadSplitNpy(
            args.split, data_path=DATA_DIR, year=args.year, squeeze=True)
    else:  # load UCR dataset
        x_train, y_train = load_UCR_dataset(args.dataset, split='train')

        X, y_true = load_UCR_dataset(args.dataset, split=args.split)

    n_classes = len(np.unique(y_train))
    logger.info(f'x_train shape: {x_train.shape}')
    logger.info(f'X shape: {X.shape}')


    # Classification model
    if args.model_arch == 'TempCNN':
        model = S2Classif(n_classes)
    elif args.model_arch == 'Inception':
        model = Inception(n_classes)
    elif args.model_arch == 'MLP':
        model = MLPClassif(n_classes)

    # Load classification model weights
    logger.info('Loading classifier')
    model_params = utils.loadWeightsAndParams(model, args.model_name)
    logger.info(f'Classifier params: {model_params}')


    # noiser model
    n_timestamps = X.shape[-1]
    noiser = Noiser(
        out_dim=n_timestamps,
        shrink=args.shrink,
        base_arch=args.noiser_arch)

    # Load noiser model weights
    logger.info('Loading noiser')
    noiser_params = utils.loadWeightsAndParams(noiser, args.noiser_name)
    logger.info(f'Noiser params: {noiser_params}')


    # device setup
    utils.setFreeDevice()
    device = utils.getCurrentDevice()
    model.to(device)
    noiser.to(device)

    # setup dataloaders
    train_dataloader = npyData2DataLoader(x_train, batch_size=2048)
    dataloader_y_true = npyData2DataLoader(X, y_true, batch_size=2048)

    # compute CF data of the chosen split
    y_pred, y_predCF, dataCF, noiseCF = utils.predictionAndCF(
        model, noiser, dataloader_y_true)

    # print confusion matrix for somples correctly predicted
    correct_idx = y_true == y_pred
    cm = confusion_matrix(y_pred[correct_idx], y_predCF[correct_idx])
    viz.printConfMatrix(cm)

    if args.do_plots:
        # write plots and tables
        # Prepare output path
        output_folder = IMG_PATH
        # output_folder = Path(
        #     IMG_PATH,
        #     # f"{args.model_name}_{args.noiser_name}",
        #     f"{args.split}")
        os.makedirs(output_folder, exist_ok=True)

        viz.produceResults(args.split, output_folder, y_true,
                           y_pred, y_predCF, dataCF, noiseCF)

        # TSNE
        # plotTSNE(y_true, pred, predCF, noise_CF, output_folder,
        # random_state=args.seed)


    # CF data of train is needed for stability computation
    _, _, trainCF, _ = predictionAndCF(model, noiser, train_dataloader)

    # Compute predictions for train data
    y_pred_train = ClfPrediction(model, train_dataloader)


    logger.info(f"Metrics computed on {args.split} data")
    metrics.metricsReport(
        X=X, Xcf=dataCF.squeeze(),
        y_pred_cf=y_predCF,
        X_train=x_train,
        y_pred_train=y_pred_train,
        Xcf_train=trainCF.squeeze(),
        ifX=x_train, k=args.stability_k, model=model)


if __name__ == "__main__":
    parser = cli.getBasicParser()
    parser = cli.addClfLoadArguments(parser)
    parser = cli.addNoiserLoadArguments(parser)
    parser = cli.addInferenceParams(parser)
    parser.add_argument(
        '--stability-k',
        type=int,
        default=5,
        help='Number of neighbors used to compute stability metric.'
        'Used only during evaluation (mode=pred).'
    )

    parser.add_argument(
        '--do-plots',
        action='store_true',
        default=False,
        help='Runs plotting functions and writes results to logdir.'
    )

    args = parser.parse_args()

    logger = log.setupLogger(__file__, parser)

    logger.info(f"Setting manual seed={args.seed} for reproducibility")
    utils.setSeed(args.seed)

    # Create img dir within log dir if needed
    IMG_PATH = os.path.join(log.getLogdir(), 'img')
    if args.do_plots:
        os.makedirs(IMG_PATH, exist_ok=True)

    launchInference(args)

    # main(args)

