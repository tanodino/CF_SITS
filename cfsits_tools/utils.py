import json
import os
from pathlib import Path
import time
import logging
import pickle as pkl
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from cfsits_tools.data import loadAllDataNpy, npyData2DataLoader
from cfsits_tools.model import S2Classif



DEFAULT_MODEL_DIR = "models"
# log to same logger initialized by the main script calling this module
logger = logging.getLogger('__main__')


def setSeed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def getCurrentDevice() -> str:
    """Get current cuda device if available, or fallback to cpu"""
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available() else "cpu")
    return device


def setFreeDevice():
    """
    Globally set a free gpu as device (if available).
    Take random if all show 0 ocupation.
    """
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        mem_aloc = np.array([torch.cuda.memory_allocated(i)
                            for i in range(count)])
        if np.all(mem_aloc == 0):
            dev_ix = random.randrange(0, count)
        else:
            dev_ix = np.argmin(mem_aloc)
        device = torch.device(f"cuda:{dev_ix}")
        torch.cuda.set_device(device)


def sendToDevice(torchObjList, device=None):
    """Send list of objects to device

    Args:
        torchObjList (Iterable): a list(-like) object containing multiple torch objects.
        device (str or TorchDevice, optional): Destination device. If None, the current device is taken with getCurrentDevice. Defaults to None.
    """
    device = device or getCurrentDevice()
    for obj in torchObjList:
        obj.to(device)


def loadWeights(model, file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    model.load_state_dict(torch.load(file_path))
    logger.info(f"Loaded weights from {file_path}")
    return file_path


def saveWeights(model, file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    torch.save(model.state_dict(), file_path)
    logger.info(f"Weights saved to {file_path}")
    return file_path


def freezeModel(model):
    for p in model.parameters():
        p.requires_grad = False


def savePkl(object, file_path):
    with open(file_path, "wb") as f:
        pkl.dump(object, f)


def loadPkl(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(f)


def savePklModel(model, file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    savePkl(model, file_path)


def loadPklModel(file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    return loadPkl(file_path)


def ClfPrediction(model, data_x, device=None):
    device = device or getCurrentDevice()
    pred_all = []
    model.eval()
    for x in data_x:
        x = x[0]
        x = x.to(device)
        pred = model(x)
        pred_all.append((pred.argmax(1)).cpu().detach().numpy())
    return np.concatenate(pred_all, axis=0)


def ClfPredProba(model, data_x, device=None):
    device = device or getCurrentDevice()
    pred_all = []
    model.eval()
    for x in data_x:
        x = x[0]
        x = x.to(device)
        logits = model(x).squeeze(1)
        probas = F.softmax(logits, dim=1)
        pred_all.append(probas.cpu().detach().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all


def evaluate(model, data_xy, device=None):
    """Computes model's F1 score on given data"""
    device = device or getCurrentDevice()
    labels = []
    pred_tot = []
    model.eval()
    for x, y in data_xy:
        labels.append(y.cpu().detach().numpy())
        x = x.to(device)
        pred = model(x)
        pred_tot.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
    labels = np.concatenate(labels, axis=0)
    pred_tot = np.concatenate(pred_tot, axis=0)
    return f1_score(labels, pred_tot, average="weighted")


def generateCF(noiser, loader, device=None):
    device = device or getCurrentDevice()
    dataCF = []
    noiser.eval()
    for x_batch in loader:
        x_batch = x_batch[0]
        x_batch = x_batch.to(device)
        to_add = noiser(x_batch)
        # print(x_batch.shape)
        # print(to_add.shape)
        dataCF.append((x_batch+to_add).cpu().detach().numpy())
    return np.concatenate(dataCF, axis=0)


def predictionAndCF(model, noiser, data, device=None):
    """from ExtractCF.py, more complete than the one from generateCF.py"""
    device = device or getCurrentDevice()
    labels = []
    pred_tot = []
    dataCF = []
    pred_CF = []
    noise_CF = []
    model.eval()
    noiser.eval()
    for x in data:
        x = x[0]
        x = x.to(device)
        pred = model(x)
        to_add = noiser(x)
        pred_cf = model(x+to_add)
        dataCF.append((x+to_add).cpu().detach().numpy())
        pred_tot.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        pred_CF.append(np.argmax(pred_cf.cpu().detach().numpy(), axis=1))
        noise_CF.append(np.squeeze(to_add.cpu().detach().numpy()))
    pred_tot = np.concatenate(pred_tot, axis=0)
    pred_CF = np.concatenate(pred_CF, axis=0)
    dataCF = np.concatenate(dataCF, axis=0)
    noiseCF = np.concatenate(noise_CF, axis=0)
    return pred_tot, pred_CF, dataCF, noiseCF


# def predictionAndCF(model, noiser, data, device):
#     """from generateCF.py"""
#     labels = []
#     pred_tot = []
#     dataCF = []
#     pred_CF = []
#     model.eval()
#     noiser.eval()
#     for x in data:
#         x = x[0]
#         x = x.to(device)
#         pred = model(x)
#         to_add = noiser(x)
#         pred_cf = model(x+to_add)
#         dataCF.append((x+to_add).cpu().detach().numpy())
#         pred_tot.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
#         pred_CF.append(np.argmax(pred_cf.cpu().detach().numpy(), axis=1))

#     pred_tot = np.concatenate(pred_tot, axis=0)
#     pred_CF = np.concatenate(pred_CF, axis=0)
#     return pred_tot, pred_CF, np.concatenate(dataCF, axis=0)


def generateOrigAndAdd(model, noiser, train, device=None):
    device = device or getCurrentDevice()
    data = []
    dataCF = []
    prediction_cf = []
    prediction = []
    orig_label = []
    model.eval()
    noiser.eval()
    for x_batch, y_batch in train:
        x_batch = x_batch.to(device)
        to_add = noiser(x_batch)
        pred_cf = model(x_batch+to_add)
        pred = model(x_batch)
        data.append(x_batch.cpu().detach().numpy())
        dataCF.append(x_batch.cpu().detach().numpy() +
                      to_add.cpu().detach().numpy())
        prediction_cf.append(np.argmax(pred_cf.cpu().detach().numpy(), axis=1))
        prediction.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        orig_label.append(y_batch.cpu().detach().numpy())
    return np.concatenate(data, axis=0), np.concatenate(dataCF, axis=0), np.concatenate(prediction, axis=0), np.concatenate(prediction_cf, axis=0), np.concatenate(orig_label, axis=0)


def computeOrig2pred(orig_label, pred):
    classes = np.unique(orig_label)
    n_classes = len(classes)
    hashOrig2Pred = {}
    for v in classes:
        idx = np.where(orig_label == v)[0]
        hashOrig2Pred[v] = np.bincount(pred[idx], minlength=n_classes)
    return hashOrig2Pred


def loadPreds(model_name, split, year):
    base_dir = Path("data")
    out_name = f'y_pred_{split}_{year}_{model_name}.npy'
    out_path = Path(base_dir, out_name)
    if not out_path.exists():
        # compute predictions if they do not exist
        # load data
        fullData = loadAllDataNpy(year=year)
        data = npyData2DataLoader(fullData[split].X, batch_size=1048)
        # load model
        setFreeDevice()
        model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
        model.to(getCurrentDevice())
        loadWeights(model, model_name)
        y_pred = ClfPrediction(model, data)
        return y_pred
    else:
        return np.load(out_path)


def savePreds(model_name, split, year):
    base_dir = Path("data")
    out_name = f'y_pred_{split}_{year}_{model_name}.npy'
    out_path = Path(base_dir, out_name)
    print(out_path)

    fullData = loadAllDataNpy(year=year, squeeze=False)
    # Load model
    setFreeDevice()
    model = S2Classif(n_class=fullData['n_classes'], dropout_rate=.5)
    model.to(getCurrentDevice())
    loadWeights(model, model_name)
    y_pred = ClfPrediction(
        model, npyData2DataLoader(fullData[split].X, batch_size=2048))
    np.save(out_path, y_pred)

