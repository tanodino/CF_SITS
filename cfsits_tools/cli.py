from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

def getBasicParser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-y","--year",
        default=2020,
        type=int,
        help='Year of the data. Only 2020 available now.'
        )
    parser.add_argument(
        "--model-name",
        default="model_weights_tempCNN",
        help='Name of the file containing classifier model weights'
        )
    parser.add_argument(
        "--noiser-name",
        default="noiser_weights_paper",
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
        '--dry-run',
        action='store_true',
        help='Run without executing computationally long actions. Useful for testing.'
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