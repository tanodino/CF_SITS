"""
author : tdrumond
date: 2023-03-28 12:07:47

Functions to compute metrics on the generated counterfactuals

Metrics:

Proximity:
L2 norm (X-Xcf)

Compactness:
((X-Xcf) > threshold).sum()

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


def validity(X, Xcf, model):
    y_pred = []
    for data in [X, Xcf]:
        dataloader = npyData2DataLoader(
            data[:, np.newaxis, :], batch_size=2048)
        y_pred.append(ClfPrediction(model, dataloader))
    return validity_from_pred(*tuple(y_pred))


def validity_from_pred(y_pred, y_pred_cf):
    return np.mean(y_pred != y_pred_cf)

def proximity(X, Xcf):
    return np.linalg.norm(X-Xcf, axis=1)


def compactness(X, Xcf, threshold=0.01):
    return ((X-Xcf) > threshold).sum()


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
    logging.info("ISOLATION FOREST RESULTS:")
    pred_orig = _to_0_1(outlier_estimator.predict(X))
    pred_cf = _to_0_1(outlier_estimator.predict(Xcf))
    logging.info(f'Real data [outliers, inliers]: {np.bincount(pred_orig)}')
    logging.info(f'CF data   [outliers, inliers]: {np.bincount(pred_cf)}')

    # Metrics
    inlier_recall = recall_score(pred_orig, pred_cf)
    outlier_precision = precision_score(pred_orig, pred_cf)
    logging.info(f"Inlier recall: {inlier_recall}")
    logging.info(f"Outlier precision: {outlier_precision}")
    logging.info("NMI score: %f" %
                 (normalized_mutual_info_score(pred_orig, pred_cf)))
    logging.info("Agreement in/outlier btw real and cf data (accuracy): %f" %
                 (accuracy_score(pred_orig, pred_cf)))
    logging.info(
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


def stability(X, Xcf, estimator, k=None, target_class=None):
    """
    Stability (aka robustness) from Ates 2020 .
    In the definition, nearest neighbors are takend from the training set.
    In the analysis, they choose single source-target pair of classes (for both query and neighbors).
    They say they take only the correctly classified samples.
    Remains unclear:
    - nn search restricted to the chosen source class ?
    - 
    """
    eps = np.finfo(X.dtype).resolution
    k = estimator.n_neighbors
    # Prepare neighborhood samples and their counterfactuals
    nnX = estimator._fit_X
    nnXcf = estimator._fit_X_cfs

    if target_class is not None:
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
            metric[i, j] = (
                (np.linalg.norm(Xcf[i] - nnXcf[nn_i]))
                / (eps + np.linalg.norm(X[i] - nnX[nn_i]))
                )

    metric = np.nanmax(metric, axis=1)
    return metric
