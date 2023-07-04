from collections import namedtuple
import os
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEFAULT_DATA_DIR = "data"
UCR_DIR = "UCRArchive_2018"

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
    if split.lower() not in VALID_SPLITS:
        raise ValueError(
            f"split should be one of {VALID_SPLITS} and not {split}")


def _fname(x_or_y, split, year=2020):
    return f"{x_or_y}_{split}_{year}.npy"


def loadOneNpy(
        x_or_y, split, data_path=None, year=2020, ndvi=True, squeeze=False, 
        ch_first=True, y_float=False):
    _validate_split(split)
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
    """Converts a dataset in numpy format to TensorDataset (from torch.utils.data).
    A 3rd dimension (channels) is added if the given X array is 2D.

    Args:
        X (ndarray): Array containing data samples. If 2D, should have shape (n_samples x n_timesteps). If 3D, should have shape (n_samples x 1 x n_timesteps)
        y (ndarray, optional): _description_. Defaults to None.

    Returns:
        TensorDataset: torch object containing both X and y in torch.Tensor form. 
    """
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
    """Converts a dataset in numpy format to a DataLoader (from torch.utils.data).
    A 3rd dimension (channels) is added if the given X array is 2D.

    Args:
        X (ndarray): Array containing data samples. If 2D, should have shape (n_samples x n_timesteps). If 3D, should have shape (n_samples x 1 x n_timesteps)
        y (ndarray, optional): _description_. Defaults to None.
        loader_kwargs: any keyword arguments taken by DataLoader (from torch.utils.data) such as shuffle, batch_size, etc.

    Returns:
        DataLoader: torch object containing both X and y, that can be iterated by batch.
    """
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


def _loadUCRtsv(name, train_or_test, data_path=None):
    data_path = data_path or DEFAULT_DATA_DIR
    path = Path(
        data_path, UCR_DIR,
        name, f"{name}_{train_or_test.upper()}.tsv"
    )
    # load data from tsv file
    data = np.loadtxt(path, delimiter='\t')
    # labels are in the first column
    X, y = data[:, 1:], data[:, 0].astype('int')
    # not all datasets have labels in the same standard
    # bellow they get standardized to be in range(0, n_classes)
    labels = np.unique(y)
    if labels[0] == 1:
        y -= 1
    elif np.all(labels == [-1, 1]):
        y = (y+1)//2
    elif np.all(labels == [3, 4, 5, 6, 7, 8]) :
        y -= 3
    return datatuple(X, y)


def load_UCR_dataset(name, split, data_path=None):
    _validate_split(split)
    # if split is valid, need to load train data file
    file_split = 'TRAIN' if split.lower() == 'valid' else split.upper()
    X, y = _loadUCRtsv(name, file_split, data_path)

    # min-max-normalize data in X
    # if current split is test, need to load train data to get max and min 
    X_train = (_loadUCRtsv(name, 'train', data_path).X
                  if split.lower() == 'test'
                  else X)
    X = (X-X_train.min())/(X_train.max()-X_train.min())

    # if split is train or valid, need to split TRAIN file in two parts before returning the result
    if  split.lower() == 'train' or split.lower() == 'valid':
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.25, random_state=123)
        train_idx, valid_idx = splitter.split(X, y)
        if split.lower() == 'train':
            return datatuple(X[train_idx], y[train_idx])
        elif split.lower() == 'valid':
            return datatuple(X[valid_idx], y[valid_idx])
    else:
        return datatuple(X, y)


def list_UCR_datasets(data_path=None):
    data_path = data_path or DEFAULT_DATA_DIR
    datasets = os.listdir(os.path.join(data_path, UCR_DIR))
    # Datasets within the following folder also exist in the root folder
    # The root folder versions are raw with missing data and variable length
    # thus being unsuitable to our experiments.
    exclude = ['Missing_value_and_variable_length_datasets_adjusted']
    exclude += os.listdir(os.path.join(data_path, UCR_DIR, exclude[0]))
    datasets = list(filter(lambda x: x not in exclude, datasets))
    datasets
    return datasets