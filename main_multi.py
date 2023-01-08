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
from sklearn.metrics import confusion_matrix

import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

from main_regDiego import discriminator_loss, generator_loss, prediction, generateOrigAndAdd, extractNDVI

def trainModelClassif(model, train, valid, n_epochs, loss_ce, optimizer, path_file, device):
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
            loss_acc.append( loss.cpu().detach().numpy() )
        
        print("epoch %d with loss %f"%(e, np.mean(loss_acc)))
        score_valid = prediction(model, valid, device)
        print("\t val on VALIDATION %f"%score_valid)
        if score_valid > best_validation:
            best_validation = score_valid
            torch.save(model.state_dict(), path_file)
            print("\t\t BEST VALID %f"%score_valid)
        
        sys.stdout.flush()

def trainClassif(train_dataset,valid_dataset):
    n_classes = len(np.unique(train_dataset.tensors[1]))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)
    #test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate=.5)
    #model = MLPClassif(n_classes, dropout_rate=.5)
    model.to(device)
    
    # Train if not already done
    file_path = "model_weights_tempCNN_Multi"    
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)       
        loss_ce = nn.CrossEntropyLoss().to(device)
        n_epochs = 1000
        trainModelClassif(model, train_dataloader, valid_dataloader, n_epochs, loss_ce, optimizer, file_path, device)

    return file_path


def trainModelNoise(model, noiser, discr, train, n_epochs, n_classes, optimizer, optimizerD, loss_bce, n_timestamps, device, path_file):
    model.eval()
    noiser.train()
    discr.train()
    #torch.autograd.set_detect_anomaly(True)
    id_temps = np.array(range(n_timestamps))
    id_temps = torch.Tensor(id_temps).to(device)
    for e in range(n_epochs):
        loss_discr = []
        loss_acc = []
        loss_cl = []
        loss_generator = []
        loss_uni = []

        t_avg_all = []
        non_zeros = [] # just to track sparsity
        loss_reg_L1 = []
        loss_reg_L2 = []

        for x_batch, y_batch in train:
            noiser.zero_grad()
            discr.zero_grad()
            optimizerD.zero_grad()
            optimizer.zero_grad()
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            to_add = noiser(x_batch)
            n_batch = x_batch.shape[0]
 
            x_cf = x_batch+to_add
            pred_cl = model(x_cf)

            # Classification loss
            prob_cl = torch.nn.functional.softmax(pred_cl,dim=1)
            y_ohe = F.one_hot(y_batch.long(), num_classes=n_classes)
            prob = torch.sum(prob_cl * y_ohe,dim=1)
            loss_classif = torch.mean( -torch.log( 1. - prob + torch.finfo(torch.float32).eps ) )

            #Time-contiguity Regularizer
            # Compute central times
            to_add_abs = torch.abs(to_add)
            #1) Shared accross variables: max absolute sum over variables
            # _, t_avg = torch.max(torch.sum(to_add_abs,dim=1,keepdim=True),dim=-1,keepdim=True)
            #2) One per variable:
            _, t_avg = torch.max(to_add_abs,dim=-1,keepdim=True)
            #3) Top-k centroids

            # Compute distance to t_avg
            #1) linear
            # diff = (t_avg - id_temps)
            #2) circular
            diff = torch.minimum(torch.remainder(t_avg - id_temps, n_timestamps),
                                torch.remainder(id_temps - t_avg, n_timestamps))

            # Weights
            # 1) Quadratic distance from \tilde{t}
            weights = torch.square(diff)
            # 2) Absolute distance
            # weights = torch.abs(diff)
            # 3) Log barrier window
            # max_w = n_timestamps // 6
            # eps = torch.Tensor([torch.finfo(torch.float32).eps]).to(device)
            # eps = torch.Tensor([1e-32]).to(device)
            # weights = -torch.log(torch.max(1 - torch.div(torch.abs(diff),max_w), eps.expand_as(diff)))

            uni_reg = torch.sum( weights * to_add_abs) / n_batch

            # Group-Lasso
            # 1) Flexible location (superposing groups)
            # g_w = n_timestamps // 6 # group width
            # groups = [(k + np.arange(g_w))%n_timestamps for k in range(n_timestamps)]
            # group_reg = 0
            # for group in groups:
            #     group_reg += torch.sum(torch.norm(to_add[...,group], dim=-1)) / n_batch
            # uni_reg = group_reg
            # 2) Fixed location
            # n_groups = 6
            # g_w = n_timestamps // n_groups # group width
            # group_reg = torch.sum(torch.norm(to_add.unflatten(-1,(n_groups,g_w)), dim=-1)) / n_batch
            # uni_reg = group_reg

            # Adversarial part
            real_output = discr( x_batch ).view(-1)
            fake_output = discr( x_batch + to_add.detach() ).view(-1)
            # Discriminator            
            loss_d = discriminator_loss(real_output, fake_output, loss_bce, device)
            loss_d.backward()
            # if e % 5 == 0: # Update discriminator only every 5 iterations
            #     loss_d.backward()
            optimizerD.step()
            # Generator
            fake_output_2 = discr( x_cf )
            fake_output_2 = torch.squeeze(fake_output_2)
            loss_g = generator_loss(fake_output_2, loss_bce, device)
            
            loss = 1.*loss_classif + 0.8*loss_g + .2*uni_reg # One t-tilde per variable - square dist
            # loss = 1.*loss_classif + 2*loss_g + .05*uni_reg # One t-tilde per variable - log barrier
            # loss = 1.*loss_classif + 0.8*loss_g + 5.*group_reg # Group Lasso - overlapping
            # loss = 1.*loss_classif + 0.4*loss_g + 10.*group_reg # Group Lasso - overlapping
            loss.backward()
            optimizer.step()


            # Tracking loss evolution
            loss_acc.append( loss.cpu().detach().numpy() ) # Composite loss
            loss_discr.append( loss_d.cpu().detach().numpy())
            loss_generator.append(loss_g.cpu().detach().numpy())
            loss_cl.append(loss_classif.cpu().detach().numpy() )
            loss_uni.append(uni_reg.cpu().detach().numpy())

            # Tracking other indicators
            reg_L1 = torch.sum( torch.abs(torch.squeeze(to_add))) / n_batch
            reg_L2 = torch.sum( torch.square(torch.squeeze(to_add))) / n_batch
            reg_L0 = torch.count_nonzero(torch.squeeze(to_add)).float() / n_batch

            non_zeros.append(reg_L0.cpu().detach().numpy())
            loss_reg_L1.append( reg_L1.cpu().detach().numpy())
            loss_reg_L2.append( reg_L2.cpu().detach().numpy())            
            t_avg_all.append(t_avg.cpu().detach().numpy())


        print("epoch %d with Noiser loss %f (l_GEN %f l_CL %f and reg_UNI %f) and Discr Loss %f. Average perturbation norm: L1=%f, L2=%f, L0=%.1f"%(e, np.mean(loss_acc), np.mean(loss_generator), np.mean(loss_cl), np.mean(loss_uni), np.mean(loss_discr), np.mean(loss_reg_L1), np.mean(loss_reg_L2), np.mean(non_zeros)))

        # --- Some result inspection ---
        data, dataCF, pred, pred_cf, orig_label = generateOrigAndAdd(model, noiser, train, device)
        #print("F1 SCORE original model %f"%f1_score(orig_label, pred,average="weighted"))

        subset_idx = np.where(pred == orig_label)[0]
        cm = confusion_matrix(pred[subset_idx], pred_cf[subset_idx])
        print(cm)

        number_of_changes = len( np.where(pred[subset_idx] != pred_cf[subset_idx])[0] )
        print("NUMBER OF CHANGED PREDICTION : %d over %d, original size is %d"%(number_of_changes, pred[subset_idx].shape[0], pred.shape[0]))
        
        idx_list = np.where(pred != pred_cf)[0]
        idx_list = shuffle(idx_list)
        idx = idx_list[0]
        sample = np.squeeze( data[idx] )
        sampleCF = np.squeeze( dataCF[idx] )
        ex_cl = pred[idx]
        ex_cfcl = pred_cf[idx]

        #Central time histogram
        plt.clf()
        t_avg_all = np.concatenate(t_avg_all,axis=0)
        plt.hist(t_avg_all.squeeze(), bins=np.concatenate(([-.5],np.arange(n_timestamps))))
        plt.savefig("epoch_%d_t_avg_hist.jpg"%(e) )

        plt.clf()
        # plt.plot(np.arange(len(sample)), sample,'b')
        # plt.plot(np.arange(len(sampleCF)), sampleCF,'r')
        plt.plot(np.arange(n_timestamps), sample.transpose(-1,-2),'b')
        plt.plot(np.arange(n_timestamps), sampleCF.transpose(-1,-2),'r')
        plt.savefig("epoch_%d_from_cl_%d_2cl_%d.jpg"%(e, ex_cl, ex_cfcl) )
        # plt.waitforbuttonpress(0) # this will wait for indefinite time
        #plt.close(fig)

        # Perturbation colormap
        plt.clf()
        perturbation = sampleCF-sample
        limit = np.abs(perturbation).max()
        plt.imshow(perturbation, cmap='seismic', vmin=-limit, vmax=limit) #aspect='auto', extent=[]
        plt.colorbar(orientation='horizontal')
        plt.title('Perturbation')
        plt.savefig("epoch_%d_from_cl_%d_2cl_%d_cmap.jpg"%(e, ex_cl, ex_cfcl) )

        torch.save(noiser.state_dict(), path_file)
        sys.stdout.flush()

def trainNoiser(train_dataset, file_path):
    n_timestamps = train_dataset.tensors[0].shape[-1]
    n_var = train_dataset.tensors[0].shape[-2]
    n_classes = len(np.unique(train_dataset.tensors[1]))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps*n_var, .3, n_var)
    discr = Discr(.2)
    model.to(device)
    noiser.to(device)
    discr.to(device)
    
    # Load Classifier weights and freeze
    model.load_state_dict(torch.load(file_path))
    for p in model.parameters():
        p.requires_grad = False

    # Train Noiser
    path_file_noiser = "noiser_weights_Multi"
    optimizer = torch.optim.Adam(noiser.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizerD = torch.optim.Adam(discr.parameters(), lr=0.0001, weight_decay=1e-4)        
    loss_bce = nn.BCELoss().to(device)
    n_epochs = 100*n_var
    trainModelNoise(model, noiser, discr, train_dataloader, n_epochs, n_classes, optimizer, optimizerD, loss_bce, n_timestamps, device, path_file_noiser)
    

def main(argv):
    year = 2020#int(argv[1])

    torch.manual_seed(0)
    print('\n=========\nManual seed activated for reproducibility\n=========')

    x_train = np.load("x_train_%d.npy"%year)
    x_valid = np.load("x_valid_%d.npy"%year)
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    x_valid = np.moveaxis(x_valid,(0,1,2),(0,2,1))

    y_train = np.load("y_train_%d.npy"%year)-1.
    y_valid = np.load("y_valid_%d.npy"%year)-1.

    # Commented so that we have multivariate time series (4 channels)
    # x_train = extractNDVI(x_train)
    # x_valid = extractNDVI(x_valid)
    
    x_train = torch.Tensor(x_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)
    
    x_valid = torch.Tensor(x_valid) # transform to torch tensor
    y_valid = torch.Tensor(y_valid)

    train_dataset = TensorDataset(x_train, y_train) # create your datset
    #test_dataset = TensorDataset(x_test, y_test) # create your datset
    valid_dataset = TensorDataset(x_valid, y_valid) # create your datset

    file_path = trainClassif(train_dataset,valid_dataset)
    trainNoiser(train_dataset,file_path)

    #print( model.parameters() )    


if __name__ == "__main__":
   main(sys.argv)
