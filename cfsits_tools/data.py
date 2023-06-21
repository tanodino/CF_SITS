from collections import namedtuple
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEFAULT_DATA_DIR = "data"

VALID_SPLITS = {'train', 'valid', 'test'}

#Define datatuple type
datatuple = namedtuple("datatuple",["X","y"])

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:, 2, :]
    nir = x_train[:, 3, :]
    temp_data = (nir - red) / ((nir + red) + eps)
    return np.expand_dims(temp_data, 1)

def _validate_split(split):
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split should be one of {VALID_SPLITS} and not {split}")


def _fname(x_or_y, split, year=2020):
    return f"{x_or_y}_{split}_{year}.npy"


def loadOneNpy(
        x_or_y, split, data_path=None, year=2020, ndvi=True, squeeze=False, 
        ch_first=True, y_float=False):
    data_path = data_path or DEFAULT_DATA_DIR
    path = Path(data_path, _fname(x_or_y, split, year))
    data = np.load(path)
    if x_or_y == 'x':
        if ch_first:
            data = np.moveaxis(data, (0, 1, 2), (0, 2, 1))
        if ndvi:
            data = extractNDVI(data)
        if squeeze:
            data = np.squeeze(data)
    elif x_or_y == 'y':
        if y_float:
            data = data - 1.0
        else:
            data = data.astype('int')
            data -= 1
    return data


def loadSplitNpy(split,  **load_kwargs):
    _validate_split(split)
    X = loadOneNpy('x', split, **load_kwargs)
    y = loadOneNpy('y', split, **load_kwargs)
    return datatuple(X, y)


def dummyTrainTestData(n_samples_per_class=100, n_timestamps=5, n_classes=3, 
                       test_size=0.2, seed=0):
    from tslearn.generators import random_walk_blobs
    from sklearn.model_selection import train_test_split
    X, y = random_walk_blobs(
        n_samples_per_class, n_timestamps, 
        d=1, n_blobs=n_classes, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X[:, :, 0], y, 
        test_size=test_size, random_state=seed, 
        stratify=y)
    fullData = dict(classes=list(range(n_classes)), n_classes=n_classes)
    fullData['train'] = datatuple(X_train, y_train)
    fullData['test'] = datatuple(X_test, y_test)
    return fullData


def loadAllDataNpy(**load_kwargs):
    data = {split: loadSplitNpy(split, **load_kwargs)
           for split in VALID_SPLITS}
    # get y_train to compute n_classes
    _, y = data['train']

    classes = np.unique(y)
    data.update(n_classes=len(classes), classes=classes)
    return data


def npyData2Dataset(X, y=None):
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    X = torch.Tensor(X)
    if y is not None:
        y = torch.Tensor(y)
        dataset = TensorDataset(X, y)
    else:
        dataset = TensorDataset(X)
    return dataset


def npyData2DataLoader(X, y=None, **loader_kwargs):
    dataset = npyData2Dataset(X, y)
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader


def loadOneDataset(split, **load_kwargs):
    X, y = loadSplitNpy(split, **load_kwargs)
    return npyData2Dataset(X, y)


def loadOneDataLoader(split, data_path=None, year=2020, **loader_kwargs):
    dataset = loadOneDataset(split, data_path, year)
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader


def loadAllDataloaders(data_path=None, year=2020, **loader_kwargs):
    dataloaders = SimpleNamespace(
        **{split: loadOneDataLoader(
            split,
            data_path,
            year,
            **loader_kwargs)
           for split in VALID_SPLITS}
    )
    return dataloaders


def load_UCR_dataset(name, split):
    UCR_DIR = "UCRArchive_2018"
    data_path = data_path or DEFAULT_DATA_DIR
    path = Path(
        data_path, UCR_DIR,
        name, f"{name}_{split}.tsv"
    )
    data = np.loadtxt(path, delimiter='\t')
    X, y = data[:, 1:], data[:, 0].astype('int')
    labels = np.unique(y)
    if labels[0] == 1:
        y -= 1
    elif np.all(labels == [-1, 1]):
        y = (y+1)//2
    elif np.all(labels == [3, 4, 5, 6, 7, 8]) :
        y -= 3

    return datatuple(X, y)

