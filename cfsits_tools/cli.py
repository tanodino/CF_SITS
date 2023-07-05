from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from cfsits_tools.data import VALID_SPLITS, list_UCR_datasets
from cfsits_tools.log import numericLogLevel


def getBasicParser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="koumbia",
        choices=["koumbia"] + list_UCR_datasets()
    )
    parser.add_argument(
        "-y","--year",
        default=2020,
        type=int,
        help='Year of the colza data. Only 2020 available now. Only used when dataset=koumbia.'
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
        '--dry-run',
        action='store_true',
        help='Run without executing computationally long actions. Useful for testing.'
    )

    return parser


def addClfLoadArguments(parser):
    parser.add_argument(
        "--model-name",
        default="model_weights",
        help='Name of the file containing classifier model weights.'
    )
    parser.add_argument(
        "--model-arch",
        default='Inception',
        help='Classification model architecture.'
    )
    return parser


def addNoiserLoadArguments(parser):
    parser.add_argument(
        "--noiser-name",
        default="noiser_weights",
        help='Name of the file containing noiser model weights.'
    )
    parser.add_argument(
        '--noiser-arch',
        default='MLP',
        help='Noiser (generator) model architecture.'
    )

    def str2bool(s: str) -> bool:
        if s in {'True', 'False'}: return eval(s) 
        else: raise TypeError(f"Invalid boolean value {s}")

    parser.add_argument(
        '--shrink',
        type=str2bool,
        default=True,
        help='Uses softshrink in the noiser model.'
    )
    return parser

def addOptimArguments(parser):
    train_args = parser.add_argument_group('Optimization args')
    train_args.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    train_args.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    train_args.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )
    train_args.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for training."
    )
    return parser

def addClfTrainArguments(parser):
    model_args = parser.add_argument_group('Classification model training args')
    model_args.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate of the classification model."
    )
    return parser


def addNoiserTrainArguments(parser):
    noiser_args = parser.add_argument_group('Noiser training args')
    noiser_args.add_argument(
        "--discr-arch",
        default='TempCNN',
        help='Discriminator model architecture.'
    )
    noiser_args.add_argument(
        '--reg-gen',
        type=float,
        default=0.5,
        help='Weight of the GAN generator loss.'
    )
    noiser_args.add_argument(
        '--reg-uni',
        type=float,
        default=691.2,
        help='Weight of the unimodal L1 regularization loss.'
    )
    noiser_args.add_argument(
        '--loss-cl-type',
        default='margin',
        choices=['log', 'margin'],
        help="Class-swapping loss type. Choose between 'log' or 'margin'."
    )
    noiser_args.add_argument(
        '--margin',
        type=float,
        default=0.1,
        help='Margin term, used only if --loss-cl-type=margin is passed.'
    )
    noiser_args.add_argument(
        "--dropout-noiser",
        type=float,
        default=0.3,
        help="Dropout rate of the noiser (generator) model."
    )
    noiser_args.add_argument(
        "--dropout-discr",
        type=float,
        default=0.2,
        help="Dropout rate of the discriminator model."
    )
    return parser


def addInferenceParams(parser):
    group = parser.add_argument_group('Inference params')
    group.add_argument(
        '--split',
        choices=VALID_SPLITS,
        default='test',
        help='Data partition used to compute results.'
    )
    return parser

