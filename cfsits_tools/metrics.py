"""
date: 2023-03-28 12:07:47

Functions to compute metrics on the generated counterfactuals

Metrics:

Proximity:
L2 norm (X-Xcf)

Compactness:
(abs(X-Xcf) <= threshold).mean()

Plausability
Use Isolation forest to count outliers
"""
from collections import namedtuple
import numpy as np
import logging
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, normalized_mutual_info_score, precision_score, recall_score
from cfsits_tools.data import npyData2DataLoader
from cfsits_tools.utils import ClfPrediction

from cfsits_tools.viz import printConfMatrix


# log to same logger initialized by the main script calling this module
logger = logging.getLogger('__main__')


def validity(X, Xcf, model):
    y_pred = []
    for data in [X, Xcf]:
        dataloader = npyData2DataLoader(
            data[:, np.newaxis, :], batch_size=2048)
        y_pred.append(ClfPrediction(model, dataloader))
    return validity_from_pred(*tuple(y_pred))


def validity_from_pred(y_pred, y_pred_cf):
    return np.mean(y_pred != y_pred_cf)


def proximity(X, Xcf, order=2):
    return np.linalg.norm(X-Xcf, axis=1, ord=order)


def relative_proximity(X, Xcf, y_cf_pred, nnX, nny, order=2):
    num =  np.linalg.norm(X-Xcf, axis=1, ord=order)
    classes = np.unique(y_cf_pred)
    # Xnun = nearest neighbor with class != ypred
    Xnun = np.zeros_like(X)
    for dst in classes:
        nnX_unlike = nnX[nny != dst]
        nn = NearestNeighbors().fit(nnX_unlike)
        nn_idx = nn.kneighbors(X[y_cf_pred==dst], n_neighbors=1, return_distance=False).squeeze()
        Xnun[y_cf_pred==dst] = nnX_unlike[nn_idx]
    den = np.linalg.norm(X-Xnun, axis=1, ord=order)
    return num/den


def compactness(X, Xcf, threshold=1e-4):
    return (np.abs(X-Xcf) <= threshold).mean(axis=1)


def plausibility(X, Xcf, X_ref=None, estimator=None):
    X_ref = X_ref if X_ref is not None else X
    if not isinstance(estimator, BaseEstimator):
        estimator = IsolationForest(n_estimators=300).fit(X_ref)
    ratios = plausibility_ratios(Xcf, estimator)
    # printOtherIFmetrics(X, Xcf, outlier_estimator)
    return ratios.inlier


def plausibility_ratios(X, outlier_estimator):
    ratiotup = namedtuple("ratiotup", ["inlier", "outlier", "total"])
    preds = _to_0_1(outlier_estimator.predict(X))
    n_outlier, n_inlier = np.bincount(preds, minlength=2)
    n_total = X.shape[0]
    ratios = ratiotup(
        inlier=n_inlier/n_total,
        outlier=n_outlier/n_total,
        total=n_total
    )
    return ratios


def printOtherIFmetrics(X, Xcf, outlier_estimator):
    logger.info("ISOLATION FOREST RESULTS:")
    pred_orig = _to_0_1(outlier_estimator.predict(X))
    pred_cf = _to_0_1(outlier_estimator.predict(Xcf))
    logger.info(f'Real data [outliers, inliers]: {np.bincount(pred_orig)}')
    logger.info(f'CF data   [outliers, inliers]: {np.bincount(pred_cf)}')

    # Metrics
    inlier_recall = recall_score(pred_orig, pred_cf)
    outlier_precision = precision_score(pred_orig, pred_cf)
    logger.info(f"Inlier recall: {inlier_recall}")
    logger.info(f"Outlier precision: {outlier_precision}")
    logger.info("NMI score: %f" %
                 (normalized_mutual_info_score(pred_orig, pred_cf)))
    logger.info("Agreement in/outlier btw real and cf data (accuracy): %f" %
                 (accuracy_score(pred_orig, pred_cf)))
    logger.info(
        "Confusion matrix: (isolation forest prediction on original data vs. IF prediction on CF)")
    cm = confusion_matrix(pred_orig, pred_cf)
    printConfMatrix(cm)


def _to_0_1(pred):
    """sklearn predicts in/outliers with values {+1,-1}
    This remaps them to {1,0}"""
    pred += 1
    pred[np.where(pred == 2)] = 1
    pred = pred.astype("int")
    return pred


def stability(
        X, Xcf, estimator=None, 
        k=None, nnX=None, nnXcf=None, 
        target_class=None):
    """
    Stability (aka robustness) from Ates 2020 .
    In the definition, nearest neighbors are takend from the training set.
    In the analysis, they choose single source-target pair of classes (for both query and neighbors).
    They say they take only the correctly classified samples.
    Remains unclear:
    - nn search restricted to the chosen source class ?
    - 
    """
    if estimator is None:
        assert nnX is not None, "estimator and nnX cannot both be none"
        assert nnXcf is not None, "estimator and nnXcf cannot both be none"
        assert k is not None, "estimator and k cannot both be none"
        estimator = NearestNeighbors(n_neighbors=k).fit(nnX)
    else:
        k = estimator.n_neighbors
        # Prepare neighborhood samples and their counterfactuals
        nnX = estimator._fit_X
        nnXcf = estimator._fit_X_cfs

    if target_class is not None and nnXcf.ndim > nnX.ndim:
        dest_y = target_class
        if isinstance(dest_y, np.ndarray):
            dest_y = target_class[0]
        nnXcf = nnXcf[:, dest_y, ...]
    elif  nnXcf.ndim > nnX.ndim:
        raise ValueError("Must inform a target class when choice is possible")
    assert nnXcf.shape == nnX.shape, f"{nnXcf.shape} != {nnX.shape}"

    nn_idx = estimator.kneighbors(
        X, n_neighbors=k, return_distance=False)
    metric = np.empty((X.shape[0], k))
    for i, x in enumerate(X):
        for j, nn_i in enumerate(nn_idx[i]):
            num = np.linalg.norm(Xcf[i] - nnXcf[nn_i])
            den = np.linalg.norm(X[i] - nnX[nn_i])
            eps = np.finfo(X.dtype).resolution
            num = np.nanmax((eps,num))
            den = np.nanmax((eps,den))
            metric[i, j] = num / den

    metric = np.nanmax(metric, axis=1)
    return metric

def applyIF(clf, x_test):
    pred = clf.predict(x_test) + 1
    pred[np.where(pred == 2)] = 1
    pred = pred.astype("int")
    return pred


def metricsReport(X, Xcf, y_pred=None, y_pred_cf=None, ifX=None,
                  X_train=None, y_pred_train=None, Xcf_train=None, k=5, 
                  Xcf_train_target_class=None, Xcf_target_class=None, model=None):
    metrics_dict = {}
    # Proximity
    for order in [1, 2, np.inf]:
        proximity_avg = np.mean(proximity(X, Xcf, order=order))
        metrics_dict[f'proximity_L{order}'] = proximity_avg
        logger.info(f"avg proximity @ norm-{order}: {proximity_avg:0.4f}")

    # Relative proximity
    for order in [1, 2, np.inf]:
        rel_prox_avg = np.mean(
            relative_proximity(X, Xcf, y_pred_cf, X_train, y_pred_train, order=order))
        metrics_dict[f'rel_proximity_L{order}'] = rel_prox_avg
        logger.info(f"avg rel proximity @ norm-{order}: {rel_prox_avg:0.4f}")

    # Plausibility
    outlier_estimator = IsolationForest(n_estimators=300).fit(ifX)
    plausibility_avg = np.mean(plausibility(
        X, Xcf, estimator=outlier_estimator))
    metrics_dict[f'plausibility'] = plausibility_avg
    logger.info(f"avg plausibility: {plausibility_avg:0.4f}")

    # Validity
    if model is None and y_pred is not None:
        validity_avg = np.mean(validity_from_pred(y_pred, y_pred_cf))
    elif model is not None:
        validity_avg = np.mean(validity(X, Xcf, model))
    metrics_dict[f'validity'] = validity_avg
    logger.info(f"avg validity: {validity_avg:0.4f}")

    # Compactness
    for threshold in [1e-2, 1e-4, 1e-8]: #  1e-3, 1e-4, 1e-8
        c = compactness(X, Xcf, threshold=threshold)
        compactness_avg = np.mean(c)
        compactness_std = np.std(c)
        exp = np.log10(threshold)
        metrics_dict[f'compactness_1e{exp:g}'] = compactness_avg
        logger.info(f"avg compactness @ threshold={threshold:0.1e}:"
                    f" {compactness_avg:0.4f} (+-{compactness_std:0.4f})"
                    f" (min: {c.min():0.4f}, max: {c.max():0.4f})")


    # Stability
    if (X_train is None or Xcf_train is None):
        logger.warning('Counterfactual examples for trainint data are required to compute stability metric, but were missing. Skipping computation')
    else:
        if Xcf_target_class is not None:
            result = np.NaN * np.ones((X.shape[0]))
            n_classes = len(np.unique(Xcf_target_class))
            for dst in range(n_classes):
                result[Xcf_target_class == dst] = stability(
                    X[Xcf_target_class == dst], 
                    Xcf[Xcf_target_class == dst],
                    k=k, 
                    nnX=X_train[Xcf_train_target_class == dst], 
                    nnXcf=Xcf_train[Xcf_train_target_class == dst])
            stability_avg = np.mean(result)

        else:
            stability_avg = np.mean(
                stability(X=X, Xcf=Xcf, k=k, nnX=X_train, nnXcf=Xcf_train))
        metrics_dict['stability'] = stability_avg
        logger.info(f"avg stability: {stability_avg:0.4f}")
        return metrics_dict