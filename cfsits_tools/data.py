from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEFAULT_DATA_DIR = "data"

VALID_SPLITS = {'train', 'valid', 'test'}



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


def loadOneNpy(x_or_y, split, data_path=None, year=2020, ndvi=True):
    data_path = data_path or DEFAULT_DATA_DIR
    path = Path(data_path, _fname(x_or_y, split, year))
    data = np.load(path)
    if x_or_y == 'x':
        data = np.moveaxis(data, (0, 1, 2), (0, 2, 1))
        if ndvi:
            data = extractNDVI(data)
    elif x_or_y == 'y':
        data = data - 1.0
    return data


def loadSplitNpy(split, data_path=None, year=2020, return_dict=False):
    _validate_split(split)
    X = loadOneNpy('x', split, data_path, year)
    y = loadOneNpy('y', split, data_path, year)
    if return_dict:
        return dict(X=X, y=y)
    else:
        return X, y


def loadAllDataNpy(data_path=None, year=2020, return_dict=False):
    data = {split: loadSplitNpy(split, data_path, year, return_dict)
           for split in VALID_SPLITS}
    # get y_train to compute n_classes
    if return_dict:
        y = data['train']['y']
    else:
        _, y = data['train']
    classes = np.unique(y)
    data.update(n_classes=len(classes), classes=classes)
    return data


def npyData2Dataset(X, y=None):
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


def loadOneDataset(split, data_path=None, year=2020):
    X, y = loadSplitNpy(split, data_path, year)
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
