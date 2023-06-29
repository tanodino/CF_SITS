# KOUMBIA
import logging
from pathlib import Path
import numpy as np
# import tensorflow as tf
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
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cfsits_tools import utils
from cfsits_tools import cli
from cfsits_tools import log


from cfsits_tools.model import Inception, MLPClassif, MLPBranch, Noiser, Discr, S2Classif
from cfsits_tools.loss import discriminator_loss, generator_loss
from cfsits_tools.utils import generateOrigAndAdd, ClfPrediction
from cfsits_tools.data import DEFAULT_DATA_DIR, load_UCR_dataset, loadSplitNpy, npyData2DataLoader
# torch.save(model.state_dict(), PATH)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

MODEL_DIR = utils.DEFAULT_MODEL_DIR
DATA_DIR = DEFAULT_DATA_DIR


def trainModelNoise(model, noiser, discr, train, n_epochs, n_classes, reg_gen, reg_uni, optimizer, optimizerD, loss_bce, n_timestamps, device, path_file, loss_cl_type="log", margin=0.1, do_plots=False):
    model.eval()
    noiser.train()
    discr.train()
    # torch.autograd.set_detect_anomaly(True)
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
        t_avg_all = []
        non_zeros = []  # just to track sparsity

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

            prob_cl = torch.nn.functional.softmax(pred_cl, dim=1)
            y_ohe = F.one_hot(y_batch.long(), num_classes=n_classes)
            prob = torch.sum(prob_cl * y_ohe, dim=1)

            if loss_cl_type == "log":
                loss_classif = torch.mean(-torch.log(1. -
                                          prob + torch.finfo(torch.float32).eps))
            else:  # margin loss
                max_other = torch.max(prob_cl * (1-y_ohe))
                loss_classif = torch.mean(-torch.log(1. - torch.maximum(
                    prob + margin - max_other, torch.tensor(0)) + torch.finfo(torch.float32).eps))

            # Entropy regularizer
            reg_entro = torch.mean(
                torch.sum(torch.special.entr(prob_cl), dim=1))

            # Unimodal Regularizer
            # Compute central time
            to_add_abs = torch.abs(np.squeeze(to_add))
            # 1) center of mass
            # weighted = to_add_abs * id_temps
            # t_avg = torch.sum( weighted, dim=1) / torch.sum(to_add_abs, dim=1)
            # 2) max absolute value
            _, t_avg = torch.max(to_add_abs, dim=1)
            # 3) convolution (possibly circular)
            # sigma, w = 1, 5
            # filter = torch.exp( - (torch.arange(-w,w+1,dtype=torch.float, device=device))**2 / (2*sigma ** 2))
            # filter /= filter.sum()
            # filter = filter[None].expand(1, -1, -1) # circ. conv requires 3D tensors
            # conv = F.conv1d(F.pad(torch.unsqueeze(to_add_abs,1), (w,w), "circular"), filter)
            # t_avg = torch.squeeze(conv.argmax(-1).float())

            # Compute distance to t_avg
            t_avg = torch.unsqueeze(t_avg, 1)
            # 1) linear
            # diff = (t_avg - id_temps)
            # 2) circular
            diff = torch.minimum(torch.remainder(t_avg - id_temps, n_timestamps),
                                 torch.remainder(id_temps - t_avg, n_timestamps))

            uni_reg = torch.mean(
                torch.sum(torch.square(diff) * to_add_abs, dim=1))
            # uni_reg = torch.mean( torch.sum( to_add_abs, dim=1) )
            # uni_reg = torch.mean( torch.sum( torch.abs(diff) * to_add_abs, dim=1) )

            # Multimodal Regularizer
            # 1) Group reg
            # regulariser = 0
            # for k in range(1,n_timestamps+1):
            #    weight = torch.pow(0.5, torch.flip(torch.arange(k),[0])).to(device).expand_as(to_add[...,:k])
            #    regulariser += (torch.norm(weight*to_add[...,:k]) + torch.norm(weight*to_add[...,-k:]))
            # multi_reg = torch.mean( torch.sum(regulariser))
            # 2) Modes centered at top-k abs value
            # k_modes = 2
            # to_add_abs = torch.abs(np.squeeze(to_add) )
            # _, t_avg = torch.topk(to_add_abs,k_modes,dim=1)
            # diff = torch.abs(torch.unsqueeze(t_avg, dim=2) - id_temps.expand(1,1,-1))
            # diff,_ = torch.min(diff,dim=1) # closest centroid
            # multi_reg = torch.mean( torch.sum( torch.square(diff) * to_add_abs, dim=1) )

            # Total Variation Regularizer L1
            reg_tv = torch.mean(torch.sum(torch.abs(torch.squeeze(
                to_add[:, :, 1:] - to_add[:, :, :-1])), dim=1))

            # Total Variation Regularizer L2
            # reg_tv = torch.mean( torch.sum( torch.square( torch.squeeze(to_add[:,:,1:] - to_add[:,:,:-1]) ),dim=1) )
            # loss += reg_tv

            # L1 regularizer
            reg_L1 = torch.mean(
                torch.sum(torch.abs(torch.squeeze(to_add)), dim=1))
            # loss += 5.*reg_L1

            # L2 Regularization
            reg_L2 = torch.mean(
                torch.sum(torch.square(torch.squeeze(to_add)), dim=1))
            # loss +=

            # L0 norm (to track sparsity)
            L0 = torch.mean(torch.count_nonzero(
                torch.squeeze(to_add), dim=1).float())

            # magnitude, _ = torch.max( torch.abs( torch.squeeze(x_cf) - torch.squeeze(x_batch) ), dim=1)
            # reg_sim = torch.mean( magnitude )
            # loss+=reg_sim
            '''
            loss_d = 0.0
            
            '''
            # discr.zero_grad()
            real_output = discr(x_batch).view(-1)
            fake_output = discr(x_batch + to_add.detach()).view(-1)

            loss_d = discriminator_loss(
                real_output, fake_output, loss_bce, device)
            loss_d.backward()
            optimizerD.step()
            # print("=========")

            # optimizer.zero_grad()
            fake_output_2 = discr(x_cf)
            fake_output_2 = torch.squeeze(fake_output_2)
            loss_g = generator_loss(fake_output_2, loss_bce, device)

            # loss_g = .5*loss_g

            # loss = 4*loss_classif + .1*uni_reg + .01*reg_L2 + .01*reg_tv + loss_g

            # .5*loss_g + .05*uni_reg #+ .05*reg_L2 + .05*reg_tv
            loss = 1.*loss_classif + reg_gen*loss_g + reg_uni*uni_reg
            loss.backward()
            optimizer.step()

            loss_acc.append(loss.cpu().detach().numpy())
            # loss_discr.append( 0 )
            loss_discr.append(loss_d.cpu().detach().numpy())

            loss_cl.append(loss_classif.cpu().detach().numpy())
            loss_reg_entro.append(0)
            # loss_reg_entro.append(reg_entro.cpu().detach().numpy() )
            loss_reg_tv.append(reg_tv.cpu().detach().numpy())
            # loss_reg_tv.append( 0 )
            loss_reg_L1.append(reg_L1.cpu().detach().numpy())
            # loss_reg_L1.append( 0 )
            loss_reg_L2.append(reg_L2.cpu().detach().numpy())
            # loss_reg_L2.append( 0 )
            loss_generator.append(loss_g.cpu().detach().numpy())
            # loss_generator.append( 0 )
            loss_uni.append(uni_reg.cpu().detach().numpy())
            t_avg_all.append(t_avg.cpu().detach().numpy())
            non_zeros.append(L0.cpu().detach().numpy())

        logger.info("epoch %d with Gen loss %f (l_GEN %f l_CL %f and reg_L1 %f and l_TV %f and reg_L2 %f and reg_UNI %f) and Discr Loss %f and L0 %.1f" % (e, np.mean(loss_acc), np.mean(
            loss_generator), np.mean(loss_cl), np.mean(loss_reg_L1), np.mean(loss_reg_tv), np.mean(loss_reg_L2), np.mean(loss_uni), np.mean(loss_discr), np.mean(non_zeros)))
        data, dataCF, pred, pred_cf, orig_label = generateOrigAndAdd(
            model, noiser, train, device)
        # print("F1 SCORE original model %f"%f1_score(orig_label, pred,average="weighted"))
        # exit()
        '''
        hashOrig2Pred = computeOrig2pred(orig_label, pred)
        for k in hashOrig2Pred.keys():
            print("\t ",k," -> ",hashOrig2Pred[k])
        print("========")
        '''
        subset_idx = np.where(pred == orig_label)[0]

        cm = confusion_matrix(pred[subset_idx], pred_cf[subset_idx])
        print(cm)

        number_of_changes = len(
            np.where(pred[subset_idx] != pred_cf[subset_idx])[0])
        logger.info("NUMER OF CHANGED PREDICTIONS : %d over %d, original size is %d" % (
            number_of_changes, pred[subset_idx].shape[0], pred.shape[0]))

        idx_list = np.where(pred != pred_cf)[0]
        idx_list = shuffle(idx_list)
        idx = idx_list[0]
        sample = np.squeeze(data[idx])
        sampleCF = np.squeeze(dataCF[idx])
        ex_cl = pred[idx]
        ex_cfcl = pred_cf[idx]

        # Central time histogram
        if do_plots:
            plt.clf()
            t_avg_all = np.concatenate(t_avg_all, axis=0)
            plt.hist(t_avg_all.squeeze(), bins=np.concatenate(
                ([-.5], np.arange(n_timestamps))))
            plt.savefig(os.path.join(args.img_path,
                        "epoch_%d_t_avg_hist.jpg" % (e)))

            plt.clf()
            plt.plot(np.arange(len(sample)), sample, 'b')
            plt.plot(np.arange(len(sampleCF)), sampleCF, 'r')
            plt.savefig(os.path.join(
                args.img_path, "epoch_%d_from_cl_%d_2cl_%d.jpg" % (e, ex_cl, ex_cfcl)))
            # plt.waitforbuttonpress(0) # this will wait for indefinite time
            # plt.close(fig)
            # exit()
        utils.saveWeights(noiser, path_file)

        sys.stdout.flush()


def launchTraining(args):
    # Load data
    if args.dataset == 'colza':
        x_train, y_train = loadSplitNpy(
            'train', data_path=DATA_DIR, year=args.year)
        x_valid, y_valid = loadSplitNpy(
            'valid', data_path=DATA_DIR, year=args.year)
    else:  # load UCR dataset
        x_train, y_train = load_UCR_dataset(args.dataset, split='train')
        x_valid, y_valid = load_UCR_dataset(args.dataset, split='valid')

    n_classes = len(np.unique(y_train))
    logger.info(f'x_train shape: {x_train.shape}')

    n_timestamps = x_train.shape[-1]

    # Classification model
    if args.model_arch == 'TempCNN':
        model = S2Classif(n_classes, dropout_rate=args.dropout_rate)
    elif args.model_arch == 'Inception':
        model = Inception(n_classes)
    elif args.model_arch == 'MLP':
        model = MLPClassif(n_classes, dropout_rate=args.dropout_rate)

    # noiser model
    noiser = Noiser(
        out_dim=n_timestamps,
        dropout_rate=args.dropout_noiser,
        shrink=args.shrink,
        base_arch=args.noiser_arch)

    # Discriminator model
    discr = Discr(args.dropout_discr, encoder=args.discr_arch)

    # device setup
    utils.setFreeDevice()
    device = utils.getCurrentDevice()
    model.to(device)
    noiser.to(device)
    discr.to(device)

    # Load and freeze classification model
    utils.loadWeights(model, args.model_name)
    utils.freezeModel(model)

    # compute y_pred
    y_pred_train = ClfPrediction(
        model, npyData2DataLoader(x_train, batch_size=2048), device)
    y_pred_train = torch.Tensor(y_pred_train)
    y_pred_valid = ClfPrediction(
        model, npyData2DataLoader(x_valid, batch_size=2048), device)
    y_pred_valid = torch.Tensor(y_pred_valid)

    # Prepare dataloaders for training
    train_dataloader = npyData2DataLoader(
        x_train, y_pred_train,  shuffle=True, batch_size=args.batch_size)
    valid_dataloader = npyData2DataLoader(
        x_valid, y_pred_valid,  shuffle=False, batch_size=64)

    optimizer = torch.optim.Adam(
        noiser.parameters(),
        lr=args.learning_rate,  # see default value at cli.py
        weight_decay=args.weight_decay  # default value at cli.py
    )
    optimizerD = torch.optim.Adam(
        discr.parameters(),
        lr=args.learning_rate,  # see default value at cli.py
        weight_decay=args.weight_decay  # default value at cli.py
    )

    # loss_ce = nn.CrossEntropyLoss().to(device)
    loss_bce = nn.BCELoss().to(device)

    trainModelNoise(
        model, noiser, discr, train_dataloader, args.epochs, n_classes, args.reg_gen, args.reg_uni,
        optimizer, optimizerD, loss_bce, n_timestamps, device, args.noiser_name, args.loss_cl_type, args.margin, args.do_plots)


if __name__ == "__main__":
    parser = cli.getBasicParser()
    parser = cli.addNoiserArguments(parser)
    parser = cli.addTrainingArguments(parser)
    parser.set_defaults(epochs=100)
    parser.add_argument(
        '--img-path',
        default=log.getLogdir(__file__)
        help='Directory in which plots and figures are saved'
    )
    parser.add_argument(
        '--do-plots',
        action='store_true',
        default=False,
        help='Runs plotting functions and writes results to IMG_PATH'
    )

    args = parser.parse_args()
    # logging set up
    logger = log.setupLogger(__file__, parser)

    logger.info(f"Setting manual seed={args.seed} for reproducibility")
    utils.setSeed(args.seed)

    launchTraining(args)

    path_file_noiser = os.path.join(MODEL_DIR, args.noiser_name)
    log.saveCopyWithParams(path_file_noiser)
