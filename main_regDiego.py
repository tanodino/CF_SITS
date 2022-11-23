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

from model import MLPClassif, MLPBranch, Noiser, Discr, S2Classif

#torch.save(model.state_dict(), PATH)

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()

def discriminator_loss(real_output, fake_output, loss_bce, device):
    #real_labels = torch.full((real_output.shape[0],), 1., dtype=torch.float, device=device)
    #fake_labels = torch.full((fake_output.shape[0],), 0., dtype=torch.float, device=device)
    real_labels = torch.Tensor([1.] * real_output.shape[0]).to(device)
    fake_labels = torch.Tensor([0.] * fake_output.shape[0]).to(device)
    #print("real_output ",real_output)
    #print("fake_output ",fake_output)

    real_loss = loss_bce(real_output, real_labels)
    fake_loss = loss_bce(fake_output, fake_labels)
    total_loss = (real_loss + fake_loss ) /2
    return total_loss


def generator_loss(fake_output, loss_bce, device):
    #real_labels = torch.full((fake_output.shape[0],), 1., dtype=torch.float, device=device)
    real_labels = torch.Tensor([1.] * fake_output.shape[0]).to(device)
    return loss_bce(fake_output, real_labels)


def prediction(model, valid, device):
    labels = []
    pred_tot = []
    model.eval()
    for x, y in valid:
        labels.append( y.cpu().detach().numpy() )
        x = x.to(device)
        pred = model(x)
        pred_tot.append( np.argmax( pred.cpu().detach().numpy() ,axis=1) )
    labels = np.concatenate(labels, axis=0)
    pred_tot = np.concatenate(pred_tot, axis=0)
    return f1_score(labels,pred_tot,average="weighted")

def generateOrigAndAdd(model, noiser, train, device):
    data = []
    dataCF = []
    prediction = []
    orig_label = []
    model.eval()
    noiser.eval()
    for x_batch, y_batch in train:
        x_batch = x_batch.to(device)
        to_add = noiser(x_batch)
        pred = model(x_batch+to_add)
        data.append(x_batch.cpu().detach().numpy())
        dataCF.append(x_batch.cpu().detach().numpy()+ to_add.cpu().detach().numpy())
        prediction.append( np.argmax( pred.cpu().detach().numpy(),axis=1) )
        orig_label.append( y_batch.cpu().detach().numpy() )
    return np.concatenate(data,axis=0), np.concatenate(dataCF,axis=0), np.concatenate(prediction,axis=0), np.concatenate(orig_label,axis=0)

def computeOrig2pred(orig_label, pred):
    classes = np.unique(orig_label)
    n_classes = len( classes )
    hashOrig2Pred = {}
    for v in classes:
        idx = np.where(orig_label == v)[0]
        hashOrig2Pred[v] = np.bincount( pred[idx], minlength=n_classes )
    return hashOrig2Pred

def trainModelNoise(model, noiser, discr, train, n_epochs, n_classes, optimizer, optimizerD, loss_bce, n_timestamps, device, path_file):
    model.eval()
    noiser.train()
    discr.train()
    #torch.autograd.set_detect_anomaly(True)
    id_temps = np.array(range(n_timestamps))
    id_temps = torch.Tensor(id_temps).to(device)
    for e in range(n_epochs):
        loss_acc = []
        loss_discr = []
        loss_reg_L1 = []
        loss_reg_L2 = []
        loss_reg_entro = []
        loss_cl = []
        loss_reg_tv = []
        loss_generator = []
        loss_uni = []

        for x_batch, y_batch in train:
            noiser.zero_grad()
            discr.zero_grad()
            optimizerD.zero_grad()
            optimizer.zero_grad()
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            to_add = noiser(x_batch)
 
            x_cf = x_batch+to_add
            pred_cl = model(x_cf)

            '''
            print("x_batch ",x_batch)
            print("to_add ", to_add)
            print("x_cf ", x_cf)
            '''

            prob_cl = torch.nn.functional.softmax(pred_cl,dim=1)
            y_ohe = F.one_hot(y_batch.long(), num_classes=n_classes)
            prob = torch.sum(prob_cl * y_ohe,dim=1)
            loss_classif = torch.mean( -torch.log( 1. - prob + torch.finfo(torch.float32).eps ) )
            #loss = 

            #Entropy regularizer
            reg_entro = torch.mean( torch.sum( torch.special.entr(prob_cl), dim=1) )
            loss = (loss_classif) + reg_entro

            #Unimodal Regularizer
            to_add_abs = torch.abs(np.squeeze(to_add) )
            weighted = to_add_abs * id_temps
            t_avg = torch.sum( weighted, dim=1) / torch.sum(to_add_abs, dim=1)
            diff = (torch.unsqueeze(t_avg,1) - id_temps)            
            uni_reg = .05*torch.mean( torch.sum( torch.square(diff) * to_add_abs, dim=1) )
            loss+=uni_reg
            
            #Total Variation Regularizer L1
            reg_tv = torch.mean( torch.sum( torch.abs( torch.squeeze(to_add[:,:,1:] - to_add[:,:,:-1]) ),dim=1) )
            loss += 3.*reg_tv

            #Total Variation Regularizer L2
            #reg_tv = torch.mean( torch.sum( torch.square( torch.squeeze(to_add[:,:,1:] - to_add[:,:,:-1]) ),dim=1) )
            #loss += reg_tv

            #L1 regularizer
            #reg_L1 = torch.mean( torch.sum( torch.abs(torch.squeeze(to_add)), dim=1) )
            #loss += 5.*reg_L1


            #L2 Regularization
            reg_L2 = torch.mean( torch.sum( torch.square(torch.squeeze(to_add)), dim=1) )
            loss += 5.*reg_L2
            
            #magnitude, _ = torch.max( torch.abs( torch.squeeze(x_cf) - torch.squeeze(x_batch) ), dim=1)
            #reg_sim = torch.mean( magnitude )
            #loss+=reg_sim
            '''
            loss_d = 0.0
            
            '''
            #discr.zero_grad()
            real_output = discr( x_batch ).view(-1)
            fake_output = discr( x_batch + to_add.detach() ).view(-1)

            
            loss_d = discriminator_loss(real_output, fake_output, loss_bce, device)
            loss_d.backward()
            optimizerD.step()
            #print("=========")

            #optimizer.zero_grad()
            fake_output_2 = discr( x_cf )
            fake_output_2 = torch.squeeze(fake_output_2)
            loss_g = generator_loss(fake_output_2, loss_bce, device)

            #loss_g = .5*loss_g 
            loss += .2*loss_g
            
            loss.backward()
            optimizer.step()
            loss_acc.append( loss.cpu().detach().numpy() )
            #loss_discr.append( 0 )
            loss_discr.append( loss_d.cpu().detach().numpy())
            
            loss_cl.append(loss_classif.cpu().detach().numpy() )
            loss_reg_entro.append(reg_entro.cpu().detach().numpy() )
            loss_reg_tv.append( reg_tv.cpu().detach().numpy())
            #loss_reg_tv.append( 0 )
            loss_reg_L2.append( reg_L2.cpu().detach().numpy())
            #loss_reg_L2.append( 0 )
            loss_reg_L1.append( 0 )
            #loss_reg_L1.append(reg_L1.cpu().detach().numpy() )
            loss_generator.append(loss_g.cpu().detach().numpy())
            #loss_generator.append( 0 )
            loss_uni.append(uni_reg.cpu().detach().numpy())



        print("epoch %d with Gen loss %f (l_GEN %f l_CL %f and reg_L1 %f and l_entro %f and l_TV %f and reg_L2 %f and reg_UNI %f) and Discr Loss %f"%(e, np.mean(loss_acc), np.mean(loss_generator), np.mean(loss_cl), np.mean(loss_reg_L1), np.mean(loss_reg_entro), np.mean(loss_reg_tv), np.mean(loss_reg_L2), np.mean(loss_uni), np.mean(loss_discr)))
        data, dataCF, pred, orig_label = generateOrigAndAdd(model, noiser, train, device)
        hashOrig2Pred = computeOrig2pred(orig_label, pred)
        for k in hashOrig2Pred.keys():
            print("\t ",k," -> ",hashOrig2Pred[k])
        print("========")
        idx = np.random.randint(low=0, high=data.shape[0],size=1)
        sample = np.squeeze( data[idx] )
        sampleCF = np.squeeze( dataCF[idx] )
        ex_cl = orig_label[idx]
        ex_cfcl = pred[idx]

        
        plt.clf()
        plt.plot(np.arange(len(sample)), sample,'b')
        plt.plot(np.arange(len(sampleCF)), sampleCF,'r')
        plt.savefig("epoch_%d_from_cl_%d_2cl_%d.jpg"%(e, ex_cl, ex_cfcl) )
        #plt.waitforbuttonpress(0) # this will wait for indefinite time
        #plt.close(fig)
        #exit()
        torch.save(noiser.state_dict(), path_file)
        sys.stdout.flush()


def trainModel(model, train, valid, n_epochs, loss_ce, optimizer, path_file, device):
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
        if score_valid > best_validation:
            best_validation = score_valid
            torch.save(model.state_dict(), path_file)
            print("\t\t BEST VALID %f"%score_valid)
        
        sys.stdout.flush()

def extractNDVI(x_train):
    eps = np.finfo(np.float32).eps
    red = x_train[:,2,:]
    nir = x_train[:,3,:]
    temp_data = (nir - red ) / ( (nir + red) + eps )
    return np.expand_dims(temp_data, 1)

def main(argv):
    year = 2020#int(argv[1])

    x_train = np.load("x_train_%d.npy"%year)
    x_valid = np.load("x_valid_%d.npy"%year)
    x_train = np.moveaxis(x_train,(0,1,2),(0,2,1))
    x_valid = np.moveaxis(x_valid,(0,1,2),(0,2,1))

    print(x_train.shape)
    
    y_train = np.load("y_train_%d.npy"%year)-1.
    y_valid = np.load("y_valid_%d.npy"%year)-1.

    n_classes = len(np.unique(y_train))

    '''
    class_id = 0
    idx = np.where(y_train == class_id)[0]
    x_train = x_train[idx]
    y_train = y_train[idx]
    '''
    print(x_train.shape)
    #exit()

    x_train = extractNDVI(x_train)
    x_valid = extractNDVI(x_valid)

    n_timestamps = x_train.shape[-1]
    
    
    
    x_train = torch.Tensor(x_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)
    
    x_valid = torch.Tensor(x_valid) # transform to torch tensor
    y_valid = torch.Tensor(y_valid)

    train_dataset = TensorDataset(x_train, y_train) # create your datset
    #test_dataset = TensorDataset(x_test, y_test) # create your datset
    valid_dataset = TensorDataset(x_valid, y_valid) # create your datset


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)
    #test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model1 = MLPBranch(.5)
    #model = MLPClassif(n_classes, .5)
    model = S2Classif(n_classes, dropout_rate = .5)
    noiser = Noiser(n_timestamps, .3)
    discr = Discr(.2)
    model.to(device)
    noiser.to(device)
    discr.to(device)
    #model1.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    
    optimizer = torch.optim.Adam(noiser.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizerD = torch.optim.Adam(discr.parameters(), lr=0.0001, weight_decay=1e-4)
    
    loss_ce = nn.CrossEntropyLoss().to(device)
    loss_bce = nn.BCELoss().to(device)
    n_epochs = 1000
    #file_path = "model_weights"
    file_path = "model_weights_tempCNN"
    #file_path = "model_weights"
    model.load_state_dict(torch.load(file_path))
    for p in model.parameters():
        p.requires_grad = False

    path_file_noiser = "noiser_weights_UNI"
    #trainModel(model, train_dataloader, valid_dataloader, n_epochs, loss_ce, optimizer, file_path, device)
    trainModelNoise(model, noiser, discr, train_dataloader, n_epochs, n_classes, optimizer, optimizerD, loss_bce, n_timestamps, device, path_file_noiser)
    
    #print( model.parameters() )
    #exit()
    '''
    import random
    
    for p in model.parameters():
        #if random.uniform(0, 1) > .5:
        p.requires_grad = True
        #print("\tFALSE")
    '''
    


if __name__ == "__main__":
   main(sys.argv)
