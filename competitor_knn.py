"""
Script to run our experiments using the kNN based CF generator from Karlsson et al 2020

Their CF models are implemented in a package called wildboar, that can be installed with pip.

For more info please see notes_on_Karlsons_code.md

author: tdrumond
date: 2023-03-13 16:39:09

"""
from pathlib import Path
import os
import numpy as np
import logging

from wildboar.explain.counterfactual import KNeighborsCounterfactual as OriginalKNeighborsCounterfactual

from cfsits_tools import cli
from cfsits_tools.cli import getBasicParser
from cfsits_tools.data import loadAllDataNpy, VALID_SPLITS, npyData2DataLoader
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import loadModel, savePklModel, loadPklModel, torchModelPredict

class KNeighborsCounterfactual(OriginalKNeighborsCounterfactual):
    def _validate_estimator(self, estimator, allow_3d=False):
        """override estimator validation to allow using our model"""
        return estimator


class ModelWrapper():
    """Wrap our models in order to provide the necessary interface for the nn counterfactual """

    def __init__(
            self, 
            model, 
            n_neighbors:int=1):
        self._model = model

        # n_neigbors is used in two situations:
        # 1) to compute n_clusters = n_samples// n_neighbors
        # 2) to compute the majority threshold: 1 + n_neighbors//n_classes
        self.n_neighbors = n_neighbors
        self._fit_X = None
        self._y = None
        self.n_features_in_ = None
        self.classes_ = None

    def fit(self, X, y):
        # wildboar's KNeighborsCounterfactual expects estimator
        # to have attributes _fit_X, _y, classes_ and n_features_in_
        self._fit_X = np.squeeze(X)
        # XXX maybe need to adapt _y for multiclass? Folowing sklearn's implemntation at
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_base.py#L451

        # _y should be the y_pred for the samples in _fit_X
        X_dl = npyData2DataLoader(X, batch_size=64)
        self._y = torchModelPredict(self._model, X_dl)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        


def trainCfModel(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year)

    # load classification model 
    model = S2Classif(fullData["n_classes"])
    loadModel(model, file_path=args.model_name)

    # wrap model for wildboar
    clfModel = ModelWrapper(
        model,
        n_neighbors=args.n_neighbors)
    X_train, y_train = fullData["train"]
    clfModel.fit(X_train, y_train)
    logging.info("Clf model ready")

    # train CF model
    cfModel = KNeighborsCounterfactual(random_state=args.seed)
    logging.info("training CF model...")
    if not args.dry_run:
        cfModel.fit(clfModel)
    logging.info("Cf model trained")

    savePklModel(cfModel, args.noiser_name)
    logging.info("Cf model saved")

def predictCfSamples(args):

    # Load data
    fullData = loadAllDataNpy(year=args.year)

    # load CF model
    cfModel = loadPklModel(args.noiser_name)

    # produce CF samples using the method explain
    # explain takes X, y as input
    # it does account for multi-class y
    # here y should contain the "destination class"
    # into which the CF explainer is going to try to push X
    # I can generate CFs to all possible classes
    
    split = args.split
    X, y = fullData[split]
    X = np.squeeze(X)
    n_samples = X.shape[0]
    for src_label in fullData["classes"]:
        logging.info(f"Building Cfs from class {src_label}...")
        X_src = X[y==src_label]
        for dst_label in fullData['classes']:
            if dst_label != src_label:
                logging.info(f"Building Cfs to class {dst_label}...")
                dest_y = dst_label * np.ones(X_src.shape[0])
                if not args.dry_run:
                    X_cfs = cfModel.explain(X_src, dest_y)
                else:
                    X_cfs = np.zeros_like(X_src)
                logging.info(f"Cfs to class {dst_label} done")
                # save CFs from src to dst classes
                output_fname = f"CFs_{split}_from_{int(src_label)}_to_{int(dst_label)}.npy"
                output_fpath = Path(args.out_path, output_fname)
                os.makedirs(output_fpath.parent, exist_ok=True)
                np.save(output_fpath, X_cfs)
                logging.info(f"Cfs from class {int(src_label)} saved to {output_fname}")


if __name__ == "__main__":

    parser = getBasicParser()
    parser.set_defaults(noiser_name='noiser_knn')

    subparsers = parser.add_subparsers(
        dest='subcommand',
        required=True)
    # train command parsing
    train_cmd = subparsers.add_parser(
        'train',
        description='trains a knn cf model'
    )
    train_cmd.add_argument(
        "-k", "--n-neighbors",
        default=1
    )

    # pred command parsing
    pred_cmd = subparsers.add_parser(
        'pred',
        description='preds CF samples from previously trained CF odel'
    )
    pred_cmd.add_argument(
        "--out-path",
        default="logs/knn_CFs"
    )
    pred_cmd.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level)

    if args.subcommand == 'train':
        trainCfModel(args)
    elif args.subcommand == 'pred':
        predictCfSamples(args)

