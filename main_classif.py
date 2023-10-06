# KOUMBIA
import numpy as np
import torch
import torch.nn as nn
import os
import sys

from cfsits_tools import utils, cli
from cfsits_tools import log
from cfsits_tools.log import saveCopyWithParams, setupLogger
from cfsits_tools.model import Inception, S2Classif, MLPClassif
from cfsits_tools.data import DEFAULT_DATA_DIR, load_UCR_dataset, loadSplitNpy, extractNDVI, npyData2DataLoader
from cfsits_tools.utils import evaluate, getCurrentDevice, saveWeights


MODEL_DIR = utils.DEFAULT_MODEL_DIR
DATA_DIR = DEFAULT_DATA_DIR


def trainModel(model, train, valid, loss_ce, optimizer, device, args):
    model.train()
    best_validation = 0
    for e in range(args.epochs):
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

        logger.info("epoch %d with loss %f" % (e, np.mean(loss_acc)))
        score_valid = utils.evaluate(model, valid, device)
        logger.info("\t val on VALIDATION %f" % score_valid)
        if score_valid > best_validation:
            best_validation = score_valid
            log.saveWeightsAndParams(model, args.model_name, args)
            logger.info("\t\t BEST VALID %f" % score_valid)

        sys.stdout.flush()


def launchTraining(args):
    # Load data
    if args.dataset == "koumbia":
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

    train_dataloader = npyData2DataLoader(
        x_train, y_train,
        shuffle=True, batch_size=args.batch_size)
    valid_dataloader = npyData2DataLoader(
        x_valid, y_valid,
        shuffle=False, batch_size=3*args.batch_size)

    # device setup
    utils.setFreeDevice()
    device = utils.getCurrentDevice()

    if args.model_arch == 'TempCNN':
        model = S2Classif(n_classes, dropout_rate=args.dropout_rate)
    elif args.model_arch == 'Inception':
        model = Inception(n_classes)
    elif args.model_arch == 'MLP':
        model = MLPClassif(n_classes, dropout_rate=args.dropout_rate)

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    loss_ce = nn.CrossEntropyLoss().to(device)

    trainModel(model, train_dataloader, valid_dataloader,
               loss_ce, optimizer, device, args)


if __name__ == "__main__":
    parser = cli.getBasicParser(
        description="Trains a classification model on a chosen dataset. Several hyperparameters can be chosen via cli arguments. Model weights and metadata get saved to a dedicated directory within logs/main_classif.")
    parser = cli.addClfLoadArguments(parser)
    parser = cli.addClfTrainArguments(parser)
    parser = cli.addOptimArguments(parser)
    parser.set_defaults(learning_rate=0.00001,
                        weight_decay=1e-4, epochs=1000, batch_size=16)
    args = parser.parse_args()
    # logging set up
    logger = log.setupLogger(__file__, parser)

    logger.info(f"Setting manual seed={args.seed} for reproducibility")
    utils.setSeed(args.seed)

    launchTraining(args)

