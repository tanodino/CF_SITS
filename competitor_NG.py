"""
Native-guide (NG)
      - Paper: https://arxiv.org/abs/2009.13211  \cite{Delaney2021}
      - Code: https://github.com/e-delaney/Instance-Based_CFE_TSC
      - Description: replace parts from nearest neighbor; use CAM activations to choose the parts.

"""
from collections import defaultdict
from functools import wraps
import os
from pathlib import Path
from typing import Any
import numpy as np
import logging
import re
from glob import glob
from dataclasses import dataclass
from scipy import signal as sig
import torch

from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from cfsits_tools import cli
from cfsits_tools.cli import getBasicParser
from cfsits_tools.data import dummyTrainTestData, loadAllDataNpy, VALID_SPLITS, npyData2DataLoader, loadSplitNpy
from cfsits_tools.model import S2Classif
from cfsits_tools.utils import ClfPredProba, loadWeights, savePklModel, loadPklModel, ClfPrediction, getDevice, ClfPrediction

from competitor_knn import getNoiserName


def predictNGCFSamples(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']

    # create CF model
    cfModel = NGCounterfactual(metric='dtw')
    if not args.dry_run:
        cfModel.fit(*fullData["train"])

    # Find and save NG samples as CFs for samples in X
    X_cfs_dict = defaultdict(dict)
    for src_label in range(n_classes):
        logging.info(f"Find NGCfs from class {src_label}...")
        is_src = y == src_label
        src_idx = np.where(is_src)[0]  # np.where returns a tuple!
        X_src = X[is_src]
        X_cfs_dict[src_label]['og_idx'] = src_idx
        possible_labels = set(range(n_classes)) - {src_label}
        for dst_label in possible_labels:
            logging.info(f"Find NGCfs to class {dst_label}...")
            dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)
            if not args.dry_run:
                cfs_idx = cfModel.explain(
                    X_src, dest_y, return_indices=True)
            else:
                cfs_idx = np.zeros(X_src.shape[0])
            X_cfs_dict[src_label][dst_label] = cfs_idx
            logging.info(f"Cfs to class {dst_label} done")

    # save CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCFs_IDX.npy"
    if not args.dry_run:
        np.save(Path(base_dir, output_fname), X_cfs_dict, allow_pickle=True)
        logging.info(f"NGCfs saved to {Path(base_dir, output_fname)}")
    return X_cfs_dict


def genCAMMixCounterfactuals(args, fullData=None):
    # Load data
    fullData = fullData or loadAllDataNpy(year=args.year, squeeze=True)
    X, y = fullData[args.split]
    n_classes = fullData['n_classes']
    # Load model
    model = S2Classif(n_class=len(np.unique(y)), dropout_rate=.5)
    model.to(getDevice())
    loadWeights(model, args.model_name)
    # create cam mixer
    mixer = CAMMixer(model)
    # Load NG CFs
    base_dir = Path(args.cfs_path, args.split)
    output_fname = f"NGCFs_IDX.npy"
    ng_cfs_dict = np.load(Path(base_dir, output_fname), allow_pickle=True)[np.newaxis][0]

    for src_label in fullData['classes']:
        og_idx = ng_cfs_dict[src_label]['og_idx']
        is_src = y == src_label
        src_idx = np.where(is_src)[0]  # np.where returns a tuple!
        X_src = X[is_src]
        possible_targets = set(fullData['classes']) - {src_label}
        for dst_label in possible_targets:
            dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)
            ng_cfs_idx = ng_cfs_dict[src_label][dst_label]
            ng_cfs = fullData['train'].X[ng_cfs_idx]
            cam_cfs = mixer.mix(X_src, ng_cfs, dest_y)
            ng_cfs_dict[src_label][dst_label] = cam_cfs

    # save CFs indices
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCAMCFs_IDX.npy"
    np.save(Path(base_dir, output_fname), ng_cfs_dict, allow_pickle=True)
    logging.info(f"NGCAMCfs saved to {Path(base_dir, output_fname)}")
    return ng_cfs_dict


def repeatCrossClass(split, fullData, mode, src_dst_dict=defaultdict(dict)):
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
            # loop over src-dst class pairs
            for src_label in fullData['classes']:
                # form src label mask
                is_src = y == src_label
                X_src = X[is_src]
                if mode == 'create':
                    src_idx = np.where(is_src)[0]  # np.where returns a tuple!
                    src_dst_dict[src_label]['og_idx'] = src_idx
                elif mode == 'transform':
                    src_idx = src_dst_dict[src_label]['og_idx']
                # iterate over all possible target classes
                possible_targets = set(fullData['classes']) - {src_label}
                for dst_label in possible_targets:
                    dest_y = dst_label * np.ones(X_src.shape[0], dtype=np.int32)
                    # call wrapped func and save results
                    if mode == 'create':
                        cfs_idx = func(X_src, dest_y, *fn_args, **fn_kwargs)
                        src_dst_dict[src_label][dst_label] = cfs_idx
                    elif mode == 'transform':
                        nn_cfs_idx = src_dst_dict[src_label][dst_label]
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
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCFs_IDX.npy"
    np.save(Path(base_dir, output_fname), X_cfs_dict, allow_pickle=True)
    logging.info(f"NGCfs saved to {Path(base_dir, output_fname)}")
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
    base_dir = Path(args.cfs_path, args.split)
    output_fname = f"NGCFs_IDX.npy"
    ng_cfs_dict = np.load(Path(base_dir, output_fname), allow_pickle=True)[np.newaxis][0]

    @repeatCrossClass(args.split, fullData, mode='transform', src_dst_dict=ng_cfs_dict)
    def genCAMCF(X_src=None, dest_y=None, ng_cfs=None):
        cam_cfs = mixer.mix(X_src, ng_cfs, dest_y)
        return cam_cfs

    X_cfs_dict = genCAMCF()

    # save CAM CFs 
    base_dir = Path(args.cfs_path, args.split)
    os.makedirs(base_dir, exist_ok=True)
    output_fname = f"NGCAMCFs.npy"
    np.save(Path(base_dir, output_fname), X_cfs_dict, allow_pickle=True)
    logging.info(f"NGCfs saved to {Path(base_dir, output_fname)}")
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
                n_neighbors=1, metric=self.metric)
            X_not_label = X[y != target_label]
            y_not_label = y[y != target_label]
            # train NN model used for NG retrieval
            knn.fit(to_time_series_dataset(X_not_label), y_not_label)
            original_indexes = np.where(y != target_label)[0]
            self.explainer_.append((knn, original_indexes))
        # self.explainer_ = KNeighborsTimeSeriesClassifier(
        #         n_neighbors=self.n_neighbors, metric=self.metric,
        #         n_jobs=8)
        # self.explainer_.fit(to_time_series_dataset(X), y)
        return self

    def explain(self, X, y, return_indices=False):
        # version with one knn per target class
        if return_indices:
            output = np.zeros((X.shape[0]), dtype=int)
        else:
            output = X.copy()
        # find ng samples
        labels = np.unique(y)
        for target_label in labels:
            to_label = y == target_label
            nn, og_idx = self.explainer_[target_label]
            ng_idx = nn.kneighbors(
                to_time_series_dataset(X[to_label, :]),
                return_distance=False)

            if return_indices:
                output[to_label] = og_idx[ng_idx.squeeze()]
            else:
                output[to_label] = nn._ts_fit.squeeze()[ng_idx]

        return output

    # def explain(self, X, target_class:int):
    #     # verion with global knn
    #     knn = self.explainer_
    #     # use native guide retrieval
    #     # knn model from tslearn expects X with shape
    #     # (n_samples, timesteps, channels)
    #     # both dist and ind have shape (n_queries, n_neighbors)
    #     dist, ind = knn.kneighbors(
    #         to_time_series_dataset(X),
    #         n_neighbors=knn.n_samples_fit_,
    #         return_distance=True)
    #     not_target_fit = knn._y != target_class
    #     closest_idx = np.argmin(dist[:, not_target_fit], axis=1)
    #     idx_for_fit_X = ind[:, not_target_fit][:, closest_idx]
    #     ng_samples = knn._fit_X[idx_for_fit_X]

    #     X_cfs = self._mix_query_ng_via_cam(X, ng_samples, target_class)

    #     return X_cfs

    # def explain_batch(self, X, y_true=None):
    #     knn = self.explainer_
    #     # both dist and ind have shape
    #     # (n_queries, n_neighbors)
    #     dist, ind = knn.kneighbors(
    #         to_time_series_dataset(X),
    #         n_neighbors=knn.n_samples_fit_,
    #         return_distance=True)

    #     X_cfs_dict = defaultdict(dict)
    #     for target_class in range(self.n_classes_):
    #         if y_true is not None:
    #             not_target = y_true != target_class
    #             this_X = X[not_target]
    #             this_dist = dist[not_target]
    #             this_ind = ind[not_target]
    #         else:
    #             this_X = X
    #             this_dist = dist
    #             this_ind = ind
    #         # find NG samples
    #         not_target_fit = knn._y != target_class
    #         closest_idx = np.argmin(this_dist[:, not_target_fit], axis=1)
    #         idx_for_fit_X = this_ind[:, not_target_fit][:, closest_idx]
    #         ng_samples = knn._fit_X[idx_for_fit_X]

    #         # mix original and ng sample via CAM
    #         this_X_cfs = self._mix_query_ng_via_cam(
    #             this_X, target_class,
    #             ng_samples)

    #         # save info
    #         if y_true is not None:
    #             this_y = y_true[not_target]
    #             possible_classes = set(range(self.n_classes_)) - {target_class}
    #             for source_class in possible_classes:
    #                 cfs = this_X_cfs[this_y == source_class]
    #                 X_cfs_dict[source_class][target_class] = cfs
    #         else:
    #             X_cfs_dict[target_class] = this_X_cfs

    #     return X_cfs_dict


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
            mix_cfs[is_target] = self._mix_query_ng_via_cam(
                X[is_target],  X_cfs[is_target], target_label)

        return mix_cfs

    def _mix_query_ng_via_cam(self, X, ng_samples, target_class):
        # mix original and ng sample via CAM
        cam_weights = getCAM(X, target_class, self.model)
        X_cfs = []
        for i, query in enumerate(X):
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
    fullData = dummyTrainTestData(n_samples_per_class=20, n_timestamps=24, n_classes=8, test_size=0.1)

    X_cfs_idx2 = predictNGCFSamples(args, fullData)
    X_cfs_idx = predictNGCFSamplesWithDecorator(args, fullData)

    X_cfs2 = genCAMMixCounterfactuals(args, fullData)
    X_cfs = genCAMMixCounterfactualsWithDecorator(args, fullData)

    

if __name__ == "__main__":
    LOG_DIR = os.path.join('logs', os.path.basename(
        os.path.splitext(__file__)[0]))

    parser = getBasicParser()
    parser.set_defaults(noiser_name='noiser_NGCAM')

    parser.add_argument(
        "--model-dir",
        help=f"Dir where classifier model is dumped/loaded. Defaults to {LOG_DIR}.",
        default=Path(LOG_DIR),
        type=Path
    )

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

    subparsers = parser.add_subparsers(dest='subcommand')

    # pred command parsing
    pred_cmd = subparsers.add_parser(
        'pred',
        help='preds NG CF samples'
    )
    # pred_cmd.add_argument(
    #     "--cfs-path",
    #     default=Path(LOG_DIR, "cf_data"),
    #     type=Path
    # )
    pred_cmd.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test"
    )

    # results command parsing
    result_cmd = subparsers.add_parser(
        'results',
        help='prints and saves results, plors, '
    )
    # result_cmd.add_argument(
    #     "--cfs-path",
    #     default=Path(LOG_DIR, "cf_data"),
    #     type=Path
    # )
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

    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s]%(levelname)s:%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=args.log_level)

    if args.subcommand == 'pred':
        predictNGCFSamples(args)
    # elif args.subcommand == 'results':
    #     getResults(args)
    else:
        logging.getLogger(__name__).setLevel(logging.INFO)
        # test_NGCounterfactual(args)
        test_CrossClassDecorator(args)
