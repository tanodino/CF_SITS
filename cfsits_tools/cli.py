from argparse import ArgumentParser
import logging

def getBasicParser():
    parser = ArgumentParser()
    parser.add_argument(
        "-y","--year",
        default=2020,
        type=int,
        )
    parser.add_argument(
        "--model-name",
        default="model_weights_tempCNN"
        )
    parser.add_argument(
        "--noiser-name",
        default="noiser_weights_paper"
    )
    parser.add_argument(
        "--seed",
        default=0
    )

    parser.add_argument(
        '--log-level',
        type=numericLogLevel,
        default=numericLogLevel('info')
    )
    parser.add_argument(
        '--dry-run',
        action='store_true'
    )

    return parser


def numericLogLevel(loglevel:str):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level

def addTrainArguments(parser):
    train_args = parser.add_argument_group('training args')
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
        default=1e-4
    )
    train_args.add_argument(
        "--batch-size",
        default=128
    )
    train_args.add_argument(
        "--dropout-rate",
        default=0.5
    )
    return parser