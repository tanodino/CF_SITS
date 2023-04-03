"""
Script to run experiments using the method Native-guide (NG)
Paper: Delaney et al, 2021 : https://arxiv.org/abs/2009.13211  
Code: https://github.com/e-delaney/Instance-Based_CFE_TSC

Method Description: 
This method looks for the query's nearest-unlike-neighbor (NUN), then replaces a portion of the query with the corresponding parts from nearest neighbor. Class activation map (CAM) is used to choose the parts.

For a given query sample with predicted class C
1. Select a pool of training samples with class != C
2. Search them for NUN to the query sample
3. Put the NUN sample through the classifier in order to retrieve its CAM
4. Use CAM to select the most influential portions from the NUN to substitute into the query.


Implementation notes:
- The original code is spread across multiple notebooks and scripts. The relevant portions have been transposed in this script to facilitate our experiments.
- Their code proceeds the CAM-based sample mixing until the source class
 probability falls bellow 0.5. This is insufficient for the multi-class case.
 we have this changed it to "until source class probability is no longer the maximal probability".
- The original code seems to use true labels of the training samples when forming the pool for Nearest-unlike-neighbor (NUN) search. This neglects wether training sample is correctly classified. He have thus used predicted classes instead.


"""
import logging
import os
from pathlib import Path
from glob import glob
from dataclasses import dataclass
import sys
from typing import Any
from tqdm import tqdm
import numpy as np
from scipy import signal as sig
import torch

from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset
from cfsits_tools.viz import produceResults

from cfsits_tools.cli import getBasicParser
from cfsits_tools.data import loadAllDataNpy, VALID_SPLITS, npyData2DataLoader
from cfsits_tools.utils import loadPreds
from cfsits_tools.metrics import metricsReport
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import ClfPredProba, loadWeights, ClfPrediction, getCurrentDevice, ClfPrediction

NGCAMCF_FNAME = f"NGCAMCFs.npy"
NGCF_FNAME = f"NGCFs_IDX.npy"
SLICE_FNAME = '_slice_{n:02d}.npy'
NSLICES = 10


def trainCFmodel(args, fullData=None):
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    y_pred = loadPreds(args.model_name, args.split, args.year)
    cfModel = NativeGuideSearch(metric='dtw')
    if not args.dry_run:
        train_pred = loadPreds(args.model_name, 'train', args.year)
        cfModel.fit(fullData["train"].X, train_pred)
    return cfModel


def predictNGSamples(args, fullData=None):
    # Prepare save of CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCF_FNAME
    if args.slice is not None:
        output_fname += SLICE_FNAME.format(n=args.slice)
    out_path = Path(base_dir, output_fname)

    # Avoid overwriting existing results
    try:
        cfs_idx = loadCFs(args.cfs_path, args.split, slice=args.slice)
        return cfs_idx
    except FileNotFoundError:
        pass

    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']

    # Load preds
    y_pred = loadPreds(args.model_name, args.split, args.year)

    # create CF model
    cfModel = trainCFmodel(args, fullData)

    if args.slice is not None:
        bs = X.shape[0]//NSLICES
        i = list(range(0, X.shape[0], bs))[args.slice]
        slice_idx = np.arange(i, i+bs)
        X = X[i:i+bs]
        y_pred = y_pred[i:i+bs]

    if not args.dry_run:
        cfs_idx = cfModel.searchNG(
            X, y_pred, return_indices=True)
    else:
        cfs_idx = np.zeros(X.shape[0])

    if not args.dry_run:
        np.save(out_path, cfs_idx, allow_pickle=True)


def genCAMMixCounterfactuals(args, fullData=None):
    # Load NGCAM CFs if they exist
    base_dir = Path(args.cfs_path, args.split)
    try:
        dataCF = loadCFs(args.cfs_path, args.split,
                         use_cam=True, slice=args.slice)
        return dataCF
    except FileNotFoundError:
        pass

    # prepare to save CFs
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = NGCAMCF_FNAME
    if args.slice is not None:
        output_fname += SLICE_FNAME.format(n=args.slice)
    out_path = Path(base_dir, output_fname)

    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    y_pred = loadPreds(args.model_name, args.split, args.year)
    n_classes = fullData['n_classes']

    # slice data 
    if args.slice is not None:
        bs = X.shape[0]//NSLICES
        i = list(range(0, X.shape[0], bs))[args.slice]
        X = X[i:i+bs]
        y_pred = y_pred[i:i+bs]

    # Load NG cf idx
    cf_idx = loadCFs(args.cfs_path, args.split,
                     use_cam=False, slice=args.slice)
    assert cf_idx.shape[0] == X.shape[0]

    # Find NG CF samples and preds from idx
    ng_cfs = fullData['train'].X[cf_idx]
    y_pred_train = loadPreds(args.model_name, 'train', args.year)
    ng_preds = y_pred_train[cf_idx]

    # Load model
    model = S2Classif(n_class=n_classes, dropout_rate=.5)
    model.to(getCurrentDevice())
    loadWeights(model, args.model_name)

    # create cam mixer and mix
    mixer = CAMMixer(model)
    if not args.dry_run:
        cam_cfs = mixer.mix(X, ng_cfs, y_pred)
        assert np.any(cam_cfs != ng_cfs)
    else:
        cam_cfs = ng_cfs

    # save CFs
    if not args.dry_run:
        np.save(out_path, cam_cfs, allow_pickle=True)
    logging.info(f"NGCAMCfs saved to {out_path}")
    return cam_cfs


def loadCFs(cfs_path, split, use_cam=False, slice=None):
    # Load NG CFs
    base_dir = Path(cfs_path, split)
    input_fname = NGCAMCF_FNAME if use_cam else NGCF_FNAME
    if slice is not None:
        input_fname += SLICE_FNAME.format(n=slice)
    try:
        in_path = Path(base_dir, input_fname)
        logging.info(f"Loading {in_path} (size:{in_path.stat().st_size})")
        cf_idx = np.load(in_path, allow_pickle=True)[np.newaxis][0]
    except FileNotFoundError:
        if slice is None:
            logging.info(f"file {input_fname} not found, "
                         "trying to load info from NGCF slice files...")
            cf_idx = joinNGCFFiles(base_dir, use_cam)
            if cf_idx is None:
                raise FileNotFoundError("No *_slice_*.npy files found.")
            logging.info("loading done")
        else:
            raise FileNotFoundError()
    return cf_idx


def joinNGCFFiles(root_dir, use_cam=False):
    pat = 'NGCAM*_slice_*.npy' if use_cam else 'NGCF*_slice_*.npy'
    pat = os.path.join(root_dir, pat)
    files = glob(pat)
    files.sort()
    if files:
        arr = [np.load(fpath) for fpath in files]
        arr = np.concatenate(arr, axis=0)
        if len(files) != NSLICES:
            logging.warning(
                f"Only {len(files)} out of {NSLICES} slices avaliable")
        return arr
    else:
        return None


def computeResults(args):
    # Load data
    fullData = loadAllDataNpy(year=args.year, squeeze=True)
    X, y_true = fullData[args.split]
    n_classes = fullData['n_classes']
    y_pred = loadPreds(args.model_name, args.split, args.year)
    y_pred_train = loadPreds(args.model_name, 'train', args.year)

    # Load model
    model = S2Classif(n_class=n_classes, dropout_rate=.5)
    model.to(getCurrentDevice())
    loadWeights(model, args.model_name)

    # load cfs
    cf_idx = loadCFs(args.cfs_path, args.split, use_cam=False)
    if args.use_cam:
        dataCF = loadCFs(args.cfs_path, args.split, use_cam=True)
    else:
        dataCF = fullData['train'].X[cf_idx]

    # compute model preds for dataCF
    y_predCF = ClfPrediction(
        model, npyData2DataLoader(dataCF, batch_size=2048))

    logging.info(
        f"{args.split} samples that went to the class of the NG: "
        f"{np.sum(fullData['train'].y[cf_idx] == y_predCF)}"
        f" out of {y_predCF.shape[0]}")

    # filter for correct preds
    is_correct = y_true == y_pred
    X = X[is_correct]
    y_true = y_true[is_correct]
    y_pred = y_pred[is_correct]
    dataCF = dataCF[is_correct]
    y_predCF = y_predCF[is_correct]


    try:
        # Load train CFs and do preds
        NG_idx_train = loadCFs(args.cfs_path, 'train', use_cam=False)
        if args.use_cam:
            dataCF_train = loadCFs(args.cfs_path, 'train', use_cam=True)
        else:
            dataCF_train = fullData['train'].X[NG_idx_train]
        y_predCF_train = ClfPrediction(
            model, npyData2DataLoader(dataCF_train, batch_size=2048))
        logging.info(
            "Number of train samples that went to the class of the NG: "
            f"{np.sum(fullData['train'].y[NG_idx_train] == y_predCF_train)}"
            f" out of {y_predCF_train.shape[0]}")
    except FileNotFoundError:
        dataCF_train = None
        logging.warning(
            "Counterfactual examples for training data not found. "
            "Stability metrics cannot be computed. To generate this data, "
            "call this script with the following arguments: "
            "--split=train pred --use-cam")

    metricsReport(
        X=X,
        Xcf=dataCF,
        y_pred_cf=y_predCF,
        X_train=fullData['train'].X,
        y_pred_train=y_pred_train,
        Xcf_train=dataCF_train,
        k=args.n_neighbors,
        ifX=fullData['train'].X,
        model=model,
    )

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


@dataclass  # using dataclass decorator to simplify constructor declaration
class NativeGuideSearch():
    metric: str = 'dtw'
    subarray_length: int = 1

    def fit(self, X, y_pred):
        n_classes = len(np.unique(y_pred))
        self.n_classes_ = n_classes
        self.explainer_ = []
        for pred_label in np.unique(y_pred):
            knn = KNeighborsTimeSeriesClassifier(
                n_neighbors=1, metric=self.metric, verbose=2)
            X_not_label = X[y_pred != pred_label]
            y_not_label = y_pred[y_pred != pred_label]
            # train NN model used for NG retrieval
            knn.fit(to_time_series_dataset(X_not_label), y_not_label)
            original_indexes = np.where(y_pred != pred_label)[0]
            self.explainer_.append((knn, original_indexes))

        return self

    def searchNG(self, X, y_pred, return_indices=False):
        if return_indices:
            output = np.zeros((X.shape[0]), dtype=int)
        else:
            output = X.copy()
        # find ng samples
        for i, sample in tqdm(enumerate(X), desc='samples', total=X.shape[0]):
            sample = X[i]
            pred_label = y_pred[i]
            nn, og_idx = self.explainer_[pred_label]
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

    def mix(self, X, X_ngs, y_pred):
        mix_cfs = X.copy()
        labels = np.unique(y_pred.astype(int))
        for source_label in labels:
            is_source = y_pred == source_label
            # mix original and ng sample via CAM
            mix_cfs[is_source] = self._mix_query_ng_arrays_via_cam(
                X[is_source],  X_ngs[is_source], source_label)

        return mix_cfs

    def _mix_query_ng_arrays_via_cam(self, X, ng_samples, source_label):
        # mix original and ng sample via CAM
        cam_weights = getCAM(ng_samples, source_label, self.model)
        X_cfs = []
        for i, query in tqdm(enumerate(X), desc='mix samples', total=X.shape[0]):
            X_cf = self._mix_one_pair_via_cam(
                source_label,
                query,
                ng_samples[i],
                cam_weights[i])
            X_cfs.append(X_cf)
        X_cfs = np.stack(X_cfs, axis=0)
        return X_cfs

    def _mix_one_pair_via_cam(
            self,
            source_class,
            query,
            ng_sample,
            cam_weights_for_ng_sample):
        subarray_length = self.initial_subarray_length

        # initialize probability target to min value so we enter the loop
        prob_source = 1
        max_prob = 1

        # Note: subaray_length gets increased at each iteration,
        # so it is checked against array lenght here to guarantee the loop ends
        while (prob_source >= max_prob
               and subarray_length <= len(ng_sample)):
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
            prob_source = probas.squeeze()[source_class]
            max_prob = probas.squeeze().max()

            subarray_length += 1

        return X_example

    def _findSubarray(self, array, k):
        # used to find the maximum contigious subarray of length k in the explanation weight vector

        # Iterate to find all the sub-arrays
        n = len(array)
        sub_arr = [array[i:i+k] for i in range(n-k+1)]
        sub_arr = np.stack(sub_arr, axis=0)
        # find array with largest value sum
        value_sum = sub_arr.sum(axis=1)
        most_influencial_array = sub_arr[np.argmax(value_sum)]
        return most_influencial_array


def native_guide_retrieval(
        query, pred_label, X_train, y_train,
        n_neighbors=1, distance='dtw'):

    X_not_label = X_train[y_train != pred_label]
    original_indexes = np.where(y_train != pred_label)[0]

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
    model.to(getCurrentDevice())
    loadWeights(model, args.model_name)
    # create NG model
    cfModel = NativeGuideSearch()
    X_train, y_train = fullData['train']
    cfModel.fit(*fullData['train'])
    # find one
    logging.debug("cf model ready, starting explain")
    cfModel.searchNG(fullData['test'].X[0:1], [1])


def test_CAM(args):
    fullData = loadAllDataNpy(squeeze=True)
    model = S2Classif(
        n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getCurrentDevice())
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
        help='preds NG CF examples'
    )
    pred_cmd.add_argument(
        "--slice",
        default=None,
        type=lambda x: int(x) if x is not None else x
    )
    pred_cmd.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )

    pred_cmd.add_argument(
        "--use-cam",
        action='store_true',
        default=False
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
    result_cmd.add_argument(
        "--use-cam",
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        datefmt='%Y/%m/%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                Path(LOG_DIR, f'{args.subcommand}_log.txt'), mode="w"
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if args.subcommand == 'pred':
        predictNGSamples(args)
        if args.use_cam:
            genCAMMixCounterfactuals(args)
    elif args.subcommand == 'results':
        computeResults(args)


