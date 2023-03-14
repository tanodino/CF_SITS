#KOUMBIA
import numpy as np
#import tensorflow as tf
import torch
import torch.nn as nn
import os
import sys
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import time
from sklearn.manifold import TSNE

from torch.utils.data import TensorDataset, DataLoader

import time
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cfsits_tools.model import S2Classif, MLPClassif
from cfsits_tools.utils import trainModel
from cfsits_tools.data import loadSplitNpy, extractNDVI

#torch.save(model.state_dict(), PATH)

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()

MODEL_DIR = 'models'
DATA_DIR = 'data'

def main(argv):
    year = 2020#int(argv[1])
    
    x_train, y_train = loadSplitNpy('train', DATA_DIR, year)
    x_valid, y_valid = loadSplitNpy('valid', DATA_DIR, year)
    x_test, y_test = loadSplitNpy('test', DATA_DIR, year)
    


    n_classes = len(np.unique(y_train))
    print(x_train.shape)



    n_timestamps = x_train.shape[-1]
    
    x_train = torch.Tensor(x_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)
    
    x_valid = torch.Tensor(x_valid) # transform to torch tensor
    y_valid = torch.Tensor(y_valid)

    train_dataset = TensorDataset(x_train, y_train) # create your datset
    #test_dataset = TensorDataset(x_test, y_test) # create your datset
    valid_dataset = TensorDataset(x_valid, y_valid) # create your datset


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)
    #test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate=.5)
    #model = MLPClassif(n_classes, dropout_rate=.5)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    
    loss_ce = nn.CrossEntropyLoss().to(device)
    n_epochs = 1000
    file_path = "model_weights_tempCNN"
    #file_path = "model_weights"
    file_path = os.path.join(MODEL_DIR, file_path)
    trainModel(model, train_dataloader, valid_dataloader, n_epochs, loss_ce, optimizer, file_path, device)    


if __name__ == "__main__":
   main(sys.argv)
