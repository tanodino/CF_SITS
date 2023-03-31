"""
Native-guide (NG)
      - Paper: https://arxiv.org/abs/2009.13211  \cite{Delaney2021}
      - Code: https://github.com/e-delaney/Instance-Based_CFE_TSC
      - Description: replace parts from nearest neighbor; use CAM activations to choose the parts.

Implementation notes:
- Y pred used for selecting pool of NUN samples in train
- Y pred also used when generating CFs

"""
from collections import defaultdict
from functools import wraps
import logging
import os
from pathlib import Path
from glob import glob
import re
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from scipy import signal as sig
import torch

from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset
from ExtractCF import produceResults

from cfsits_tools.cli import getBasicParser
from cfsits_tools.data import dummyTrainTestData, loadAllDataNpy, VALID_SPLITS, npyData2DataLoader
from cfsits_tools.metrics import compactness, metricsReport, proximity, plausibility, stability, validity
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import ClfPredProba, loadWeights, ClfPrediction, getDevice, ClfPrediction, setFreeDevice

NGCAMCF_FNAME = f"NGCAMCFs.npy"
NGCF_FNAME = f"NGCFs_IDX.npy"


def trainCFmodel(args, fullData=None):
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    y_pred = loadPreds(args.model_name, args.split, args.year)
    cfModel = NGCounterfactual(metric='dtw')
    if not args.dry_run:
        train_pred = loadPreds(args.model_name, 'train', args.year)
        cfModel.fit(fullData["train"].X, train_pred)
    return cfModel


def predictNGCFSamples(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']

    # Load preds
    y_pred = loadPreds(args.model_name, args.split, args.year)

    # create CF model
    cfModel = trainCFmodel(args, fullData)

    # Prepare save of CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCF_FNAME
    out_path = Path(base_dir, output_fname)

    # Avoid overwriting existing results
    try:
        logging.info(
            f"{out_path} (size:{out_path.stat().st_size}) exists, loading it instead")
        X_cfs_dict = loadCFs(args.cfs_path, args.split)
        return X_cfs_dict
    except FileNotFoundError:
        pass

    # Find and save NG samples as CFs for samples in X
    X_cfs_dict = _predNGCFToDict(X, y_pred, n_classes, cfModel, args)

    if not args.dry_run:
        np.save(out_path, X_cfs_dict, allow_pickle=True)
    logging.info(f"NGCfs saved to {out_path}")

    return X_cfs_dict


def predictNGCFSamplesClassPair(args, fullData=None):
    # prepare out path
    src_label = args.src_class
    dst_label = args.dst_class
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCFs_IDX_cl{src_label}_to_cl{dst_label}.npy"
    out_path = Path(base_dir, output_fname)

    # Avoid overwriting existing result
    if out_path.exists():
        logging.info(
            f"{out_path} (size:{out_path.stat().st_size}) exists, loading it instead")
        X_cfs_dict = np.load(out_path, allow_pickle=True)[np.newaxis][0]
        return X_cfs_dict

    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']

    # Load preds
    y_pred = loadPreds(args.model_name, args.split, args.year)

    # create CF model
    cfModel = trainCFmodel(args, fullData)

    # Find and save NG samples as CFs for samples in X
    is_src = y_pred == src_label
    src_idx = np.where(is_src)[0]  # np.where returns a tuple!
    X_src = X[is_src]
    cfs_idx = _predNGCFForClassPair(src_label, dst_label, X_src, cfModel, args)

    # save CFs indices
    to_save = cfs_idx
    if not args.dry_run:
        np.save(out_path, to_save, allow_pickle=True)
        logging.info(f"NGCfs saved to {out_path}")
    else:
        logging.info(f"Save path is {out_path}")
    return cfs_idx


def genCAMMixCounterfactualsClassPair(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']

    # Load model
    setFreeDevice()
    model = S2Classif(n_class=n_classes, dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    # create cam mixer
    mixer = CAMMixer(model)

    # Find and save NGCAM samples as CFs for samples in X
    src_label = args.src_class
    dst_label = args.dst_class

    logging.info(f"Load NGCfs from class {src_label} to {dst_label}...")
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    input_fname = f"NGCFs_IDX_cl{src_label}_to_cl{dst_label}.npy"
    in_path = Path(base_dir, input_fname)
    ng_cfs_idx = np.load(in_path, allow_pickle=True)[np.newaxis][0]
    logging.info(f"Loaded {in_path}")

    #load preds
    y_pred = loadPreds(args.model_name, args.split, args.year)
    is_src = y_pred == src_label
    X_src = X[is_src]

    cam_cfs = _genCAMMixCFForClassPair(
        src_label, dst_label, ng_cfs_idx, X_src, fullData, mixer, args)

    to_save = cam_cfs

    # save CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCAMCFs_cl{src_label}_to_cl{dst_label}.npy"
    if not args.dry_run:
        np.save(Path(base_dir, output_fname), to_save, allow_pickle=True)
    logging.info(f"NGCAMCfs saved to {Path(base_dir, output_fname)}")
    return cam_cfs


def genCAMMixCounterfactuals(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']
    # Load model
    model = S2Classif(n_class=n_classes, dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    # create cam mixer
    mixer = CAMMixer(model)
    #  Load preds
    y_pred = loadPreds(args.model_name, args.split, args.year)
    # Load NG CFs
    base_dir = Path(args.cfs_path, args.split)
    try:
        ng_cfs_dict = loadCFs(args.cfs_path, args.split, use_cam=True)
        return ng_cfs_dict
    except FileNotFoundError:
        ng_cfs_dict = loadCFs(args.cfs_path, args.split, use_cam=False)

    cam_cfs_dict = _genCAMMixCFFromDict(
        ng_cfs_dict, mixer, X, y_pred, fullData['train'].X, args)

    # save CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCAMCF_FNAME
    out_path = Path(base_dir, output_fname)
    if not args.dry_run:
        np.save(out_path, cam_cfs_dict, allow_pickle=True)
    logging.info(f"NGCAMCfs saved to {out_path}")
    return cam_cfs_dict


def savePreds(model_name, split, year):
    base_dir = Path("data")
    out_name = f'y_pred_{split}_{year}_{model_name}.npy'
    out_path = Path(base_dir, out_name)
    print(out_path)

    fullData = loadAllDataNpy(year=year, squeeze=False)
    # Load model
    setFreeDevice()
    model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, model_name)
    y_pred = ClfPrediction(
        model, npyData2DataLoader(fullData[args.split].X, batch_size=2048))
    np.save(out_path, y_pred)


def loadPreds(model_name, split, year):
    base_dir = Path("data")
    out_name = f'y_pred_{split}_{year}_{model_name}.npy'
    out_path = Path(base_dir, out_name)
    if not out_path.exists():
        # compute predictions if they do not exist
        #load data
        fullData = loadAllDataNpy(split=split, year=year)
        data = npyData2DataLoader(fullData[split].X, batch_size=1048)
        # load model
        setFreeDevice()
        model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
        model.to(getDevice())
        loadWeights(model, args.model_name)
        y_pred = ClfPrediction(model, data)
        return y_pred
    else:
        return np.load(out_path)


def computeResults(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year, squeeze=True)
    X, y_true = fullData[args.split]
    n_classes = fullData['n_classes']

    # Load model
    model = S2Classif(n_class=n_classes, dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    y_pred = loadPreds(args.model_name, args.split, args.year)

    # load cfs
    cfs_dict = loadCFs(args.cfs_path, args.split, args.use_cam)
    train_cfs_dict = loadCFs(args.cfs_path, 'train', args.use_cam)

    # get Xcf if dict contains only indexes
    if args.use_cam == False:
        cfs_dict = idx_dict_to_cf_dict(cfs_dict, fullData['train'].X)
        train_cfs_dict = idx_dict_to_cf_dict(train_cfs_dict, fullData['train'].X)
    
    # # Not needed anymore ------------------
    # logging.getLogger(__name__).setLevel(logging.WARNING)
    # if not args.use_cam:
    #     mode = 'load+transform'
    #     @repeatCrossClass(args.split, fullData, mode, cfs_dict)
    #     def get_X_cfs(X_src, dest_y, X_cfs): return X_cfs
    #     cfs_dict = get_X_cfs()
    #     @repeatCrossClass(args.split, fullData, mode, train_cfs_dict)
    #     def get_X_cfs(X_src, dest_y, X_cfs): return X_cfs
    #     train_cfs_dict = get_X_cfs()
    # logging.getLogger(__name__).setLevel(logging.INFO)
    # # -------------------------------------


    # XXX fixed, not needed anymore
    # files were saved following y_true as src class
    # reorganize using y_pred
    # cfs_dict = swich_true_to_pred_class(cfs_dict, y_true, y_pred)
    # XXX 


    # convert test cf dict to long array
    dataCF, idxCF, dstClass = cf_dict_to_long_array(cfs_dict, y_pred)
    # keep only correctly classified samples
    is_correct=y_true[idxCF] == y_pred[idxCF]
    # reindex dataCF and co.
    dataCF = dataCF[is_correct]
    idxCF = idxCF[is_correct]
    dstClass = dstClass[is_correct]
    # reindex X, y_true and y_pred to make them coherent with dataCF
    X = X[idxCF]
    y_true = y_true[idxCF]
    y_pred = y_pred[idxCF]

    # compute model predcs for dataCF
    y_cf_pred = ClfPrediction(
        model, npyData2DataLoader(dataCF, batch_size=2048))


    # do same for x train
    train_pred = loadPreds(args.model_name, 'train', args.year)

    # XXX fixed, not needed anymore
    # # files were saved following y_true as src class
    # # reorganize using y_pred
    # train_cfs_dict = swich_true_to_pred_class(
    #     train_cfs_dict, fullData['train'].y, train_pred)
    # XXX fixed, not needed anymore


    train_cfs, train_cf_idx, train_cf_dst = cf_dict_to_long_array(train_cfs_dict, train_pred)

    

    # reindex train X so it matches the train cf array
    nnX = fullData['train'].X[train_cf_idx]
    # compute model predicitons for reindexed X train
    nny = ClfPrediction(
        model, npyData2DataLoader(nnX, batch_size=2048))

    metricsReport(
        X=X, Xcf=dataCF, 
        y_cf_pred=y_cf_pred,
        nnX = nnX, 
        nnXcf = train_cfs, 
        nny=nny,
        k=args.n_neighbors, 
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
            f"{args.model_name}_noiser_NG"+"CAM" if args.use_cam else "",
            f"{args.split}")
        os.makedirs(output_folder, exist_ok=True)

        produceResults(args.split, output_folder, y_true,
                    y_pred, y_predCF, dataCF, noiseCF)


def class_pair_dict_to_square_array(src_dst_dict):
    """when src dst dict contains scalars, reorganize those as a n_class square matrix

    Args:
        src_dst_dict (dict[dict[scalar]]): src dst dict containing scalars

    Returns:
        array: (n_class, n_class) square array
    """
    nrows = len(src_dst_dict.keys())
    array = np.zeros((nrows, nrows))
    for src in src_dst_dict.keys():
        for dst in src_dst_dict[src].keys():
            src, dst = int(src), int(dst)
            array[src, dst] = src_dst_dict[src][dst]
    return array


def switch_src_dst_dict(src_dst_dict):
    dst_src_dict = defaultdict(dict)
    for src in src_dst_dict.keys():
        for dst in src_dst_dict[src].keys():
            dst_src_dict[dst][src] = src_dst_dict[src][dst]
    return dst_src_dict


def swich_true_to_pred_class(src_dst_dict, y_true, y_pred):
    long_array = cf_dict_to_array(src_dst_dict, y_true)
    new_dict = defaultdict(dict)
    for src in src_dst_dict.keys():
        for dst in src_dst_dict[src].keys():
            x_cf = np.atleast_2d(long_array[y_pred == src, dst, ...])
            new_dict[src][dst] = x_cf
    return new_dict


def idx_dict_to_cf_dict(ng_idx_dict, X_train):
    new_dict = defaultdict(dict)
    for src, src_content in ng_idx_dict.items():
        for dst, idx in src_content.items():
            new_dict[src][dst] = X_train[idx]
    return new_dict


def cf_dict_to_array(cfs_dict, y_src):
    n_classes = len(np.unique(y_src))
    n_samples = y_src.shape[0]
    sample_shape = cfs_dict[0][1].shape[1:]
    arr = np.full((n_samples, n_classes, *sample_shape), np.NaN)
    for src, src_content in cfs_dict.items():
        src = int(src)
        og_idx = np.where(y_src == src)[0]
        for dst in src_content.keys():
            arr[og_idx, dst, ...] = src_content[dst]
    return arr


def cf_dict_to_long_array(cfs_dict, y_src):
    """ Produces a 2D array containing all CFs.
        returns it along with :
        idx: original indices in data
        dstClass: targeted class for each CF sample
    """
    n_classes = len(np.unique(y_src))
    n_samples = y_src.shape[0]
    sample_shape = cfs_dict[0][1].shape[1:]
    arr = []
    idx = []
    dstClass = []
    for src, src_content in cfs_dict.items():
        src = int(src)
        og_idx = np.where(y_src == src)[0]
        for dst in src_content.keys():
            Xcf = src_content[dst]
            arr.append(Xcf)
            idx.append(og_idx)
            dstClass += [dst] * Xcf.shape[0]
    arr = np.concatenate(arr, axis=0)
    idx = np.concatenate(idx, axis=0)
    dstClass = np.array(dstClass)
    return arr, idx, dstClass


def loadCFs(cfs_path, split, use_cam=False):
    # Load NG CFs
    base_dir = Path(cfs_path, split)
    input_fname = NGCAMCF_FNAME if use_cam else NGCF_FNAME
    try:
        in_path = Path(base_dir, input_fname)
        logging.info(f"Loading {in_path}")
        ng_cfs_dict = np.load(in_path, allow_pickle=True)[np.newaxis][0]
    except FileNotFoundError:
        logging.info(f"file {input_fname} not found, "
                     "trying to load info from class-pair NGCF files...")
        ng_cfs_dict = joinNGCFFiles(base_dir, use_cam)
        if ng_cfs_dict is None:
            raise FileNotFoundError("No *_cl#_to_cl#.npy files found.")
        logging.info("loading done")
    return ng_cfs_dict


def joinNGCFFiles(root_dir, use_cam=False):
    pat = 'NGCAM*cl*.npy' if use_cam else 'NGCF*cl*.npy'
    files = glob(pat, root_dir=root_dir)
    files.sort()
    if files:
        src_dst = [re.match(r'\D+_cl(\d)_to_cl(\d)', s).groups()
                   for s in files]
        src_dst_dict = defaultdict(dict)
        for (src, dst), fname in zip(src_dst, files):
            src, dst = int(src), int(dst)
            src_dst_dict[src][dst] = np.load(Path(root_dir, fname))

        return src_dst_dict
    else:
        return None


def _predNGCFToDict(X, y, n_classes, cfModel, args):
    X_cfs_dict = defaultdict(dict)
    for src_label in range(n_classes):
        is_src = y == src_label
        src_idx = np.where(is_src)[0]  # np.where returns a tuple!
        X_src = X[is_src]
        X_cfs_dict[src_label]['og_idx'] = src_idx
        possible_labels = set(range(n_classes)) - {src_label}
        for dst_label in possible_labels:
            cfs_idx = _predNGCFForClassPair(
                src_label, dst_label, X_src, cfModel, args)
            X_cfs_dict[src_label][dst_label] = cfs_idx
    return X_cfs_dict


def _predNGCFForClassPair(src_label, dst_label, X_src, cfModel, args):
    logging.info(f"Find NGCfs from class {src_label}"
                 " to class {dst_label}...")
    dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)
    if not args.dry_run:
        cfs_idx = cfModel.explain(
            X_src, dest_y, return_indices=True)
    else:
        cfs_idx = np.zeros(X_src.shape[0])
    logging.info(f"Cfs to class {dst_label} done")
    return cfs_idx


def _genCAMMixCFFromDict(ng_cfs_dict, mixer, X, y_pred, X_train, args):
    classes = np.unique(y_pred)
    for src_label in classes:
        # og_idx = ng_cfs_dict[src_label]['og_idx']
        is_src = y_pred == src_label
        # src_idx = np.where(is_src)[0]  # np.where returns a tuple!
        X_src = X[is_src]
        possible_targets = set(classes) - {src_label}
        for dst_label in possible_targets:
            ng_cfs_idx = ng_cfs_dict[src_label][dst_label]
            cam_cfs = _genCAMMixCFForClassPair(
                src_label, dst_label, ng_cfs_idx, X_src, X_train, mixer, args)
            ng_cfs_dict[src_label][dst_label] = cam_cfs
        return ng_cfs_dict


def _genCAMMixCFForClassPair(
        src_label, dst_label, ng_cfs_idx, X_src, X_train, mixer, args):
    dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)

    ng_cfs = X_train[ng_cfs_idx]
    if not args.dry_run:
        logging.info(
            f"Computing CAM CFs from class {src_label} to {dst_label}...")
        cam_cfs = mixer.mix(X_src, ng_cfs, dest_y)
        logging.info(f"NGCAMCfs from class {src_label} to {dst_label} done")
    else:
        cam_cfs = ng_cfs
    return cam_cfs


def repeatCrossClass(split, fullData, mode, input_dict=defaultdict(dict),
                     y_pred=None):
    """Decorated function should follow the signature depending on mode
        mode='create'
        --> func(X_src, dest_y, *fn_args, **fn_kwargs)
        mode='transform'
        --> func(X_src, dest_y, X_cfs, *fn_args, **fn_kwargs)
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*fn_args, **fn_kwargs):
            # Load data
            X, y = fullData[split]
            y = y_pred or y
            src_dst_dict = defaultdict(dict)
            # loop over src-dst class pairs
            for src_label in fullData['classes']:
                logging.info(f"From class {src_label}...")
                # form src label mask
                is_src = y == src_label
                X_src = X[is_src]
                if mode == 'create':
                    src_idx = np.where(is_src)[0]  # np.where returns a tuple!
                    src_dst_dict[src_label]['og_idx'] = src_idx
                elif mode == 'transform':
                    # src_idx = src_dst_dict[src_label]['og_idx']
                    pass
                # iterate over all possible target classes
                possible_targets = set(fullData['classes']) - {src_label}
                for dst_label in possible_targets:
                    logging.info(f"... to class {dst_label}...")
                    dest_y = dst_label * \
                        np.ones(X_src.shape[0], dtype=np.int32)
                    # call wrapped func and save results
                    if mode == 'create':
                        cfs_idx = func(X_src, dest_y, *fn_args, **fn_kwargs)
                        src_dst_dict[src_label][dst_label] = cfs_idx
                    elif 'transform' in mode:
                        nn_cfs = input_dict[src_label][dst_label]
                        if 'load' in mode:
                            nn_cfs_idx = nn_cfs
                            nn_cfs = fullData['train'].X[nn_cfs_idx]
                        X_cfs = func(
                            X_src, dest_y, nn_cfs, *fn_args, **fn_kwargs)
                        src_dst_dict[src_label][dst_label] = X_cfs
            return src_dst_dict
        return wrapper
    return decorate


def predictNGCFSamplesWithDecorator(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)

    # create CF model
    cfModel = NGCounterfactual(metric='dtw')
    if not args.dry_run:
        cfModel.fit(*fullData["train"])

    # Prepare save of CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCF_FNAME
    out_path = Path(base_dir, output_fname)

    # Avoid overwriting existing results
    try:
        logging.info(
            f"{out_path} (size:{out_path.stat().st_size}) exists, loading it instead")
        X_cfs_dict = loadCFs(args.cfs_path, args.split)
        return X_cfs_dict
    except FileNotFoundError:
        pass

    @repeatCrossClass(args.split, fullData, mode='create')
    def genNGCF(X_src, dest_y):
        if not args.dry_run:
            cfs_idx = cfModel.explain(
                X_src, dest_y, return_indices=True)
        else:
            cfs_idx = np.zeros(X_src.shape[0])
        return cfs_idx

    X_cfs_dict = genNGCF()

    # save CFs indices
    if not args.dry_run:
        np.save(out_path, X_cfs_dict, allow_pickle=True)
    logging.info(f"NGCfs saved to {out_path}")
    return X_cfs_dict


def genCAMMixCounterfactualsWithDecorator(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)

    # Load model
    model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)

    # create cam mixer
    mixer = CAMMixer(model)

    # Load NG CFs
    try:
        ng_cfs_dict = loadCFs(args.cfs_path, args.split, use_cam=True)
        return ng_cfs_dict
    except FileNotFoundError:
        ng_cfs_dict = loadCFs(args.cfs_path, args.split, use_cam=False)

    @repeatCrossClass(args.split, fullData, 'load+transform', ng_cfs_dict)
    def genCAMCF(X_src=None, dest_y=None, ng_cfs=None):
        cam_cfs = mixer.mix(X_src, ng_cfs, dest_y)
        return cam_cfs

    X_cfs_dict = genCAMCF()

    # save CAM CFs
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCAMCF_FNAME
    out_path = Path(base_dir, output_fname)
    if not args.dry_run:
        np.save(out_path, X_cfs_dict, allow_pickle=True)
    logging.info(f"NGCfs saved to {out_path}")
    return X_cfs_dict


@dataclass  # using dataclass decorator to simplify constructor declaration
class NGCounterfactual():
    metric: str = 'dtw'
    subarray_length: int = 1

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.n_classes_ = n_classes
        self.explainer_ = []
        for target_label in np.unique(y):
            knn = KNeighborsTimeSeriesClassifier(
                n_neighbors=1, metric=self.metric, verbose=2)
            X_not_label = X[y != target_label]
            y_not_label = y[y != target_label]
            # train NN model used for NG retrieval
            knn.fit(to_time_series_dataset(X_not_label), y_not_label)
            original_indexes = np.where(y != target_label)[0]
            self.explainer_.append((knn, original_indexes))

        return self

    def explain(self, X, y, return_indices=False):
        # version with one knn per target class
        if return_indices:
            output = np.zeros((X.shape[0]), dtype=int)
        else:
            output = X.copy()
        # find ng samples
        for i, sample in tqdm(enumerate(X), desc='samples', total=X.shape[0]):

            target_label = y[i]
            nn, og_idx = self.explainer_[target_label]
            ng_idx = nn.kneighbors(
                to_time_series_dataset(sample),
                return_distance=False)

            if return_indices:
                output[i] = og_idx[ng_idx.squeeze()]
            else:
                output[i] = nn._ts_fit.squeeze()[ng_idx]

        return output


@dataclass
class CAMMixer():
    model: Any
    pred_treshold: float = 0.5
    initial_subarray_length: int = 1

    def mix(self, X, X_cfs, y):
        # y has target classes
        mix_cfs = X.copy()
        labels = np.unique(y.astype(int))
        for target_label in labels:
            is_target = y == target_label
            # mix original and ng sample via CAM
            mix_cfs[is_target] = self._mix_query_ng_arrays_via_cam(
                X[is_target],  X_cfs[is_target], target_label)

        return mix_cfs

    def _mix_query_ng_arrays_via_cam(self, X, ng_samples, target_class):
        # mix original and ng sample via CAM
        cam_weights = getCAM(X, target_class, self.model)
        X_cfs = []
        for i, query in tqdm(enumerate(X), desc='samples', total=X.shape[0]):
            X_cf = self._mix_one_pair_via_cam(
                target_class,
                query,
                ng_samples[i],
                cam_weights[i])
            X_cfs.append(X_cf)
        X_cfs = np.stack(X_cfs, axis=0)
        return X_cfs

    def _mix_one_pair_via_cam(
            self,
            target_class,
            query,
            ng_sample,
            cam_weights_for_ng_sample):
        """

        """
        subarray_length = self.initial_subarray_length

        # initialize probability target to min value so we enter the loop
        prob_target = 0

        # Note: subaray_length gets increased at each iteration,
        # so it is checked against array lenght here to guarantee the loop ends
        while (prob_target < self.pred_treshold
               and subarray_length < ng_sample.shape[0]):
            # find the most influential portion of the NG sample
            # based on its cam weights
            most_influencial_array = self._findSubarray(
                cam_weights_for_ng_sample, subarray_length)
            # find the index corrsponding to its start
            starting_point = np.where(
                cam_weights_for_ng_sample == most_influencial_array[0])[0][0]
            end_point = subarray_length + starting_point
            # mix query and NG sample using the start and end points defined above
            X_example = np.concatenate((
                query[:starting_point],
                ng_sample[starting_point:end_point],
                query[end_point:]))

            # predict model probas for the new example
            dl_example = npyData2DataLoader(
                X_example[np.newaxis, :], batch_size=1)

            probas = ClfPredProba(self.model, dl_example)
            prob_target = probas.squeeze()[target_class]

            subarray_length += 1

        return X_example

    def _findSubarray(self, array, k):
        # used to find the maximum contigious subarray of length k in the explanation weight vector

        n = len(array)

        vec = []

        # Iterate to find all the sub-arrays
        for i in range(n-k+1):
            temp = []

            # Store the sub-array elements in the array
            for j in range(i, i+k):
                temp.append(array[j])

            # Push the vector in the container
            vec.append(temp)

        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return (vec[np.argmax(sum_arr)])


def native_guide_retrieval(
        query, target_label, X_train, y_train,
        n_neighbors=1, distance='dtw'):

    X_not_label = X_train[y_train != target_label]
    original_indexes = np.where(y_train != target_label)[0]

    # finding the nearest unlike neighbour.
    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_not_label)
    dist, ind = knn.kneighbors(query.reshape(1, -1), return_distance=True)
    # return the distance and the index in the orginal data
    return dist[0], original_indexes[ind[0]]


def getFeaturePrediction(model, data, feature_layer_name):
    # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().detach().numpy())

    model.get_submodule(feature_layer_name).register_forward_hook(hook_feature)

    pred = ClfPrediction(model, data)

    return pred, features_blobs


def getCAM(X, target_class, model):
    last_conv_feat = 'encoder.conv3'
    # get model's CAM weights for data X
    data = npyData2DataLoader(X, batch_size=2048)
    _, features = getFeaturePrediction(model, data, last_conv_feat)
    features = np.concatenate(features, axis=0).transpose((0, 2, 1))

    # get weights from last_conv to classif layer
    Ws = []
    reached_last_conv = False
    for name, w in model.named_parameters():
        if last_conv_feat in name:
            reached_last_conv = True
        elif (reached_last_conv
              and ".bn" not in name  # skip batch norm weights
              and name.endswith('weight')):
            Ws.append(w)
    # compute "equivalent" classifier weights
    # softmax_weights has shape (n_classes, n_units)
    softmax_weights = torch.linalg.multi_dot(
        Ws[::-1]).data.cpu().detach().numpy()

    target_class_w = softmax_weights[target_class, :]
    weights = features.dot(target_class_w.T)
    # resample weight series so it has the same size as the data series
    weights = np.apply_along_axis(
        sig.resample, 1, weights, X.shape[-1])

    return weights


def test_NGCounterfactual(args):
    # Load data
    fullData = loadAllDataNpy(squeeze=True)
    #  laod classif model
    model = S2Classif(
        n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    # create NG model
    cfModel = NGCounterfactual()
    X_train, y_train = fullData['train']
    cfModel.fit(*fullData['train'])
    # find one
    logging.debug("cf model ready, starting explain")
    cfModel.explain(fullData['test'].X[0:1], [1])


def test_CAM(args):
    fullData = loadAllDataNpy(squeeze=True)
    model = S2Classif(
        n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    data_loader = npyData2DataLoader(fullData['train'].X, batch_size=2048)

    pred = ClfPrediction(model, data_loader)

    pred, feat = getFeaturePrediction(model, data_loader, 'encoder')

    getCAM(fullData['train'].X[0:10], 0, model)

    q = fullData['test'].X[-2:]
    ng = fullData['train'].X[0:2]
    target = fullData['train'].y[0:2]
    mixer = CAMMixer(model)
    mixer.mix(q, ng, target)


def test_CrossClassDecorator(args):
    args.cfs_path = Path(args.cfs_path.parent, 'cf_dummydata')
    fullData = dummyTrainTestData(
        n_samples_per_class=20, n_timestamps=24, n_classes=8, test_size=0.1)

    X_cfs_idx2 = predictNGCFSamples(args, fullData)
    X_cfs_idx = predictNGCFSamplesWithDecorator(args, fullData)

    X_cfs2 = genCAMMixCounterfactuals(args, fullData)
    X_cfs = genCAMMixCounterfactualsWithDecorator(args, fullData)
    logging.debug("done")


if __name__ == "__main__":
    LOG_DIR = os.path.join('logs', os.path.basename(
        os.path.splitext(__file__)[0]))
    os.makedirs(LOG_DIR, exist_ok=True)

    parser = getBasicParser()
    parser.set_defaults(noiser_name='noiser_NGCAM')

    parser.add_argument(
        "--cfs-path",
        help=f"Dir where CF examples get dumped/loaded. Defaults to {LOG_DIR}/cf_data.",
        default=Path(LOG_DIR, "cf_data"),
        type=Path
    )

    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )

    parser.add_argument(
        "--use-cam",
        action='store_true',
        default=False
    )

    subparsers = parser.add_subparsers(dest='subcommand')

    # pred command parsing
    pred_cmd = subparsers.add_parser(
        'pred',
        help='preds NG CF samples'
    )
    pred_cmd.add_argument(
        "--src-class",
        default=None,
        type=lambda x: int(x) if x is not None else x
    )
    pred_cmd.add_argument(
        "--dst-class",
        default=None,
        type=lambda x: int(x) if x is not None else x
    )

    # results command parsing
    result_cmd = subparsers.add_parser(
        'results',
        help='prints and saves results, plots, '
    )
    result_cmd.add_argument(
        "-k", "--n_neighbors",
        default=5,
        type=int
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

    args = parser.parse_args()

    logging.basicConfig(
        filename=Path(LOG_DIR, f'{args.subcommand}_log.txt'),
        filemode='w',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=args.log_level)

    if args.subcommand == 'pred':
        if args.src_class is not None and args.dst_class is not None:
            predictNGCFSamplesClassPair(args)
            if args.use_cam:
                genCAMMixCounterfactualsClassPair(args)
        else:
            predictNGCFSamples(args)
            if args.use_cam:
                genCAMMixCounterfactuals(args)
                # genCAMMixCounterfactualsWithDecorator(args)
    if args.subcommand == 'results':
        computeResults(args)

    # elif args.subcommand == 'results':
    #     getResults(args)
    else:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        # test_NGCounterfactual(args)
        # test_CrossClassDecorator(args)
