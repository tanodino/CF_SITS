import os
import sys
import time
import pickle as pkl

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

DEFAULT_MODEL_DIR = "models"

def setSeed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def getDevice() -> str:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def sendToDevice(torchObjList, device=None):
    device = device or getDevice()
    for obj in torchObjList:
        obj.to(device)


def loadModel(model, file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    model.load_state_dict(torch.load(file_path))


def freezeModel(model):
    for p in model.parameters():
        p.requires_grad = False

def savePklModel(model, file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    with open(file_path, "wb") as f:
        pkl.dump(model, f)

def loadPklModel(file_path, model_dir=None):
    model_dir = model_dir or DEFAULT_MODEL_DIR
    file_path = os.path.join(model_dir, file_path)
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    with open(file_path, "rb") as f:
        return pkl.load(f)

def trainModel(model, train, valid, n_epochs, loss_ce, optimizer, path_file, device):
    # XXX on main_multi.py, this is called at each epoch begining
    # Which one is correct?
    model.train()
    best_validation = 0
    for e in range(n_epochs):
        loss_acc = []
        for x_batch, y_batch in train:
            model.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_ce(pred, y_batch.long())
            loss.backward()
            optimizer.step()
            loss_acc.append(loss.cpu().detach().numpy())

        print("epoch %d with loss %f" % (e, np.mean(loss_acc)))
        score_valid = evaluate(model, valid, device)
        print("\t val on VALIDATION %f" % score_valid)
        if score_valid > best_validation:
            best_validation = score_valid
            torch.save(model.state_dict(), path_file)
            print("\t\t BEST VALID %f" % score_valid)

        sys.stdout.flush()


def torchModelPredict(model, data_x):
    device = getDevice()
    model.to(device)
    return ClfPrediction(model, data_x, device)


def ClfPrediction(model, data_x, device):
    pred_all = []
    model.eval()
    for x in data_x:
        x = x[0]
        x = x.to(device)
        pred = model(x)
        pred_all.append((pred.argmax(1)).cpu().detach().numpy())
    return np.concatenate(pred_all, axis=0)


def evaluate(model, data_xy, device):
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


def generateCF(noiser, loader, device):
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


def predictionAndCF(model, noiser, data, device):
    """from ExtractCF.py, more complete than the one from generateCF.py"""
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
    return pred_tot, pred_CF, np.concatenate(dataCF, axis=0), np.concatenate(noise_CF, axis=0)


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


def generateOrigAndAdd(model, noiser, train, device):
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


def applyIF(clf, x_test):
    pred = clf.predict(x_test) + 1
    pred[np.where(pred == 2)] = 1
    pred = pred.astype("int")
    return pred
