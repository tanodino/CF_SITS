"""
date: 2023-03-13 16:39:09

Script to run our experiments using the kNN based CF generator from Karlsson et al 2020

Their CF models are implemented in a package called wildboar, that can be installed with pip.

For more info please see notes_on_Karlsons_code.md

"""
import os
from pathlib import Path
import numpy as np
import logging
import re
from glob import glob
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from wildboar.explain.counterfactual import KNeighborsCounterfactual as OriginalKNeighborsCounterfactual

from cfsits_tools import cli
from cfsits_tools.cli import getBasicParser
from cfsits_tools.data import loadAllDataNpy, VALID_SPLITS, npyData2DataLoader, loadSplitNpy
from cfsits_tools.metrics import compactness, metricsReport, plausibility, proximity, stability, validity
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import loadWeights, savePklModel, loadPklModel, ClfPrediction, getCurrentDevice, ClfPrediction, setFreeDevice


from ExtractCF import produceResults


def trainCfModel(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year)

    # load classification model
    model = S2Classif(fullData["n_classes"])
    loadWeights(model, file_path=args.model_name)

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

    savePklModel(cfModel, getNoiserName(args), args.model_dir)
    logging.info("Cf model saved")


def predictCfSamples(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year, squeeze=True)
    X, _ = fullData[args.split]

    # load CF model
    cfModel = loadPklModel(getNoiserName(args), args.model_dir)

    # load classification model
    setFreeDevice()
    model = S2Classif(fullData["n_classes"])
    model.to(getCurrentDevice())
    loadWeights(model, file_path=args.model_name)
    y_pred = ClfPrediction(model, npyData2DataLoader(X, batch_size=2048))

    # Generate CFs to all possible classes
    for src_label in fullData["classes"]:
        logging.info(f"Building Cfs from class {src_label}...")
        X_src = X[y_pred == src_label]
        X_src_idx = np.arange(X.shape[0])[y_pred == src_label]
        for dst_label in fullData['classes']:
            if dst_label != src_label:
                logging.info(f"Building Cfs to class {dst_label}...")
                # produce CF samples using the method explain
                # explain takes X, y as input
                # it does account for multi-class y
                # here y should contain the "destination class"
                # into which the CF explainer is going to try to push X
                dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)
                if not args.dry_run:
                    X_cfs = cfModel.explain(X_src, dest_y)
                else:
                    X_cfs = np.zeros_like(X_src)
                logging.info(f"Cfs to class {dst_label} done")
                # save CFs from src to dst classes
                base_dir = Path(args.cfs_path, args.split)
                os.makedirs(base_dir, exist_ok=True)
                output_fname = f"CFs_from_{int(src_label)}_to_{int(dst_label)}.npy"
                np.save(Path(base_dir, output_fname), X_cfs)
                logging.info(
                    f"Cfs from class {int(src_label)}"
                    "saved to {Path(base_dir, output_fname)}")

                # save indexes in the original X array
                idx_fname = output_fname.replace("CFs", "IDX")
                if not args.dry_run:
                    np.save(Path(base_dir, idx_fname), X_src_idx)


def predictCfSamplesFromData(X, n_classes, y_pred, cfModel):
    """Predict CF samples given data and a NN CF model from wildboar

    Args:
        X (ndarray): input data
        n_classes (int): _description_
        y_pred (ndarray): predicitons for each sample in input data
        cfModel: wildboar NN CF model

    Returns:
        tuple[ndarray]: 
               (2d array) arr : all Cfs in a long 2D array
               (array) idx: idx of original data corresponding to CF
               (array) dstClass: intended dst class when CF was generated
    """    
    n_samples = X.shape[0]
    sample_shape = X.shape[1:]
    arr = []
    idx = []
    dstClass = []
    logging.info("predicting CFs...")
    for dst in range(n_classes):
        not_dst = np.where(y_pred != dst)[0]
        y_target = np.full((not_dst.shape[0]), dst)        
        Xcf =  cfModel.explain(X[not_dst], y_target)
        arr.append(Xcf)
        idx.append(not_dst)
        dstClass += [dst] * Xcf.shape[0]
    arr = np.concatenate(arr, axis=0)
    idx = np.concatenate(idx, axis=0)
    dstClass = np.array(dstClass)
    logging.info("predicting CFs done")
    return arr, idx, dstClass


def loadCFExamples(path):
    """Load CF samples from previously generated file

    Args:
        path (str-like): place where to search for npy file

    Returns:
        tuple[ndarray]: 
               (2d array) dataCF : all Cfs in a long 2D array
               (1d array) idxCF: idx of original data corresponding to CF
               (array) srcClass: src class of data corresponding to CF
               (array) dstClass: intended dst class when CF was generated
    """    
    file_list = glob(str(path)+"/CFs*.npy")
    idx_list = glob(str(path)+"/IDX*.npy")
    dataCF = []
    srcClass = []
    dstClass = []
    idxCF = []
    for fpath, idx_path in zip(file_list, idx_list):
        # load data
        this_dataCF = np.load(fpath)
        dataCF.append(this_dataCF)
        idxCF.append(np.load(idx_path))
        # extract class info from filename
        fname = Path(fpath).name
        src, dst = re.search(r"from_(\d+)_to_(\d+)", fname).groups()
        srcClass += [int(src)] * this_dataCF.shape[0]
        dstClass += [int(dst)] * this_dataCF.shape[0]
    dataCF = np.concatenate(dataCF, axis=0)
    idxCF = np.concatenate(idxCF, axis=0)
    srcClass = np.array(srcClass)
    dstClass = np.array(dstClass)
    return dataCF, idxCF, srcClass, dstClass


def getResults(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year, squeeze=True)
    X, y_true = fullData[args.split]
    n_classes = fullData['n_classes']
    n_samples = X.shape[0]
    dataloader = npyData2DataLoader(X, y_true, batch_size=2048)

    # Load model and make predictions
    setFreeDevice()
    model = S2Classif(n_class=len(np.unique(y_true)), dropout_rate=.5)
    model.to(getCurrentDevice())
    loadWeights(model, args.model_name)
    y_pred = ClfPrediction(model, dataloader)

    # load CF model
    cfModel = loadPklModel(getNoiserName(args), args.model_dir)

    # # load counter factuals for split
    # dataCF, idxCF, srcClass, dstClass = loadCFExamples(
    #     Path(args.cfs_path, args.split))
    dataCF, idxCF, dstClass = \
        predictCfSamplesFromData(X, n_classes, y_pred, cfModel)

    # keep only correctly classified samples
    is_correct=y_true[idxCF] == y_pred[idxCF]
    # reindex dataCF and co.
    dataCF = dataCF[is_correct]
    idxCF = idxCF[is_correct]
    # srcClass = srcClass[is_correct]
    dstClass = dstClass[is_correct]
    # reindex X, y_true and y_pred to make them coherent with dataCF
    X = X[idxCF]
    y_true = y_true[idxCF]
    y_pred = y_pred[idxCF]

    y_cf_pred = ClfPrediction(
        model, npyData2DataLoader(dataCF, batch_size=2048))





    #gen train CFs
    train_cfs = []
    train_cf_idx = []
    train_cf_dst = []
    for dst in range(n_classes):
        dest_y = dst * np.ones(fullData['train'].X.shape[0])
        train_cfs.append(cfModel.explain(fullData['train'].X, dest_y))
        train_cf_dst.append(dest_y)
        train_cf_idx.append(np.arange(dest_y.shape[0]))
    train_cfs = np.concatenate(train_cfs, axis=0)
    train_cf_idx = np.concatenate(train_cf_idx, axis=0)
    train_cf_dst = np.concatenate(train_cf_dst, axis=0)
    # reindex train X so it matches the train cf array
    nnX = fullData['train'].X[train_cf_idx].squeeze()
    # compute model predicitons for reindexed X train
    nny = ClfPrediction(
        model, npyData2DataLoader(nnX, batch_size=2048))

    metricsReport(X=X, Xcf=dataCF, 
                  y_cf_pred=y_cf_pred,
                  nnX = nnX, 
                  nnXcf = train_cfs, 
                  nny=nny,
                  k=args.stability_k, 
                  ifX=fullData['train'].X, 
                  model=model, 
                  nnDstClass=train_cf_dst,
                  dstClass=dstClass)

    if args.do_plots:
        # Calculate noise
        noiseCF = dataCF - X
        # make predictions on data CF to evaluate classifier perf on them
        dataCF_dataloader = npyData2DataLoader(
            dataCF[:, np.newaxis, :], batch_size=2048)
        y_predCF = ClfPrediction(model, dataCF_dataloader)
        
        # Prepare output path
        output_folder = Path(
            args.out_path,
            f"{args.model_name}_{getNoiserName(args)}",
            f"{args.split}")
        os.makedirs(output_folder, exist_ok=True)

        produceResults(args.split, output_folder, y_true,
                    y_pred, y_predCF, dataCF, noiseCF)


class KNeighborsCounterfactual(OriginalKNeighborsCounterfactual):
    def _validate_estimator(self, estimator, allow_3d=False):
        """override estimator validation to allow using our model"""
        return estimator


class ModelWrapper():
    """Wrap our models in order to provide the necessary interface for the nn counterfactual """

    def __init__(
            self,
            model,
            n_neighbors: int = 1):
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
        self.n_features_in_ = self._fit_X.shape[1]
        self.classes_ = np.unique(y)

        # _y should be the y_pred for the samples in _fit_X
        X_dl = npyData2DataLoader(X, batch_size=64)
        self._y = ClfPrediction(self._model, X_dl)
        # XXX maybe need to adapt _y for multiclass? Folowing sklearn's implemntation at
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_base.py#L451


def getNoiserName(args):
    return args.noiser_name.replace(
        "knn",
        f"{args.n_neighbors}nn")


if __name__ == "__main__":
    LOG_DIR = os.path.join('logs', os.path.basename(
        os.path.splitext(__file__)[0]))

    parser = getBasicParser()
    parser.set_defaults(noiser_name='noiser_knn')
    parser.add_argument(
        "-k", "--n-neighbors",
        help="number of neighbors passed to the NN counterfactual model",
        default=1
    )

    parser.add_argument(
        "--model-dir",
        help=f"Dir where noiser model is dumped/loaded. Defaults to {LOG_DIR}.",
        default=Path(LOG_DIR),
        type=Path
    )

    subparsers = parser.add_subparsers(dest='subcommand')

    # train command parsing
    train_cmd = subparsers.add_parser(
        'train',
        help='trains a knn cf model'
    )

    # pred command parsing
    pred_cmd = subparsers.add_parser(
        'pred',
        help='preds CF samples from previously trained CF model'
    )
    pred_cmd.add_argument(
        "--cfs-path",
        default=Path(LOG_DIR, "cf_data"),
        type=Path
    )
    pred_cmd.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )

    # results command parsing
    result_cmd = subparsers.add_parser(
        'results',
        help='prints and saves results, plots',
    )
    result_cmd.add_argument(
        "--cfs-path",
        default=Path(LOG_DIR, "cf_data"),
        type=Path
    )
    result_cmd.add_argument(
        "--out-path",
        default=Path("img"),
        type=Path
    )
    result_cmd.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )
    result_cmd.add_argument(
        "--do-plots",
        action="store_true"
    )
    result_cmd.add_argument(
        "--stability-k",
        help="n neighbors for stability metric",
        type=int,
        default=5
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=Path(LOG_DIR, f'{args.subcommand}_log.txt'),
        filemode='w',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=args.log_level)
    logging.info("-"*30 + 'NEW RUN' + "-"*30)
    if args.subcommand == 'train':
        trainCfModel(args)
    elif args.subcommand == 'pred':
        predictCfSamples(args)
    elif args.subcommand == 'results':
        getResults(args)
