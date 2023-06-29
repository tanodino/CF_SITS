from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from cfsits_tools.data import list_UCR_datasets
from cfsits_tools.log import numericLogLevel


def getBasicParser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        default='colza',
        choices=['colza']+list_UCR_datasets()
    )
    parser.add_argument(
        "-y","--year",
        default=2020,
        type=int,
        help='Year of the colza data. Only 2020 available now. Only used when dataset=colza.'
    )
    parser.add_argument(
        "--model-name",
        default="model_weights",
        help='Name of the file containing classifier model weights'
    )
    parser.add_argument(
        "--noiser-name",
        default="noiser_weights",
        help='Name of the file containing noiser model weights'
    )
    parser.add_argument(
        "--seed",
        default=0,
        help='Seed for random generators.'
    )
    parser.add_argument(
        '--log-level',
        type=numericLogLevel,
        default=numericLogLevel('info'),
        help='Set logging level for script (debug, info, warning, error). '
    )
    parser.add_argument(
        '--logfile',
        default='log.txt',
        help='Choose logging file name.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without executing computationally long actions. Useful for testing.'
    )

    return parser


def addTrainingArguments(parser):
    train_args = parser.add_argument_group('Training args')
    train_args.add_argument(
        "--learning-rate",
        default=1e-4
    )
    train_args.add_argument(
        "--weight-decay",
        default=1e-4
    )
    train_args.add_argument(
        "--epochs",
        default=1000
    )
    train_args.add_argument(
        "--batch-size",
        default=128,
        help="Batch size used for training."
    )
    return parser

def addClfModelArguments(parser):
    model_args = parser.add_argument_group('Classification model args')
    model_args.add_argument(
        "--dropout-rate",
        default=0.5,
        help="Dropout rate of the classification model."
    )
    model_args.add_argument(
        "--model-arch",
        default='TempCNN',
        help='Classification model architecture.'
    )
    return parser



def addNoiserArguments(parser):
    noiser_args = parser.add_argument_group('Noiser training args')
    noiser_args.add_argument(
        "--discr-arch",
        default='TempCNN',
        help='Discriminator model architecture.'
    )
    noiser_args.add_argument(
        '--noiser-arch',
        default='MLP',
        help='Noiser (generator) model architecture.'
    )
    noiser_args.add_argument(
        '--shrink',
        action='store_true',
        default=False,
        help='When training, uses softshrink in the noiser model. On inference, uses a noiser model trained with softshrink.'
    )
    noiser_args.add_argument(
        '--reg-gen',
        default=0.5,
        help='Weight of the GAN generator loss.'
    )
    noiser_args.add_argument(
        '--reg-uni',
        default=0.05,
        help='Weight of the unimodal L1 regularization loss.'
    )
    noiser_args.add_argument(
        '--loss-cl-type',
        default='log',
        choices=['log' 'margin'],
        help="Class-swapping loss type. Choose between 'log' or 'margin'."
    )
    noiser_args.add_argument(
        '--margin',
        default=0.1,
        help='Margin term, used only if --loss-cl-type=margin is passed.'
    )
    noiser_args.add_argument(
        "--dropout-noiser",
        default=0.3,
        help="Dropout rate of the noiser (generator) model."
    )
    noiser_args.add_argument(
        "--dropout-discr",
        default=0.2,
        help="Dropout rate of the discriminator model."
    )
    return parser




