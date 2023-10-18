import hashlib
import json
import logging
import os
from pathlib import Path
import re
from shutil import copy2
import sys
from typing import Optional

from cfsits_tools import utils

logger = logging.getLogger('__main__')

IGNORE_ARGS = ['do_plots']
PARAMS_FILE_EXT = '.params.json'

def numericLogLevel(loglevel:str) -> int:
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level


def getScriptName(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def _check_parser_or_args(parser, args):
    if args is None and parser is not None:
        args = parser.parse_args()
    elif args is None and parser is None:
        raise RuntimeError("You must provide at least one of {args, parser}.")
    return args

def createLogdir(filename, parser=None, args=None):
    args = _check_parser_or_args(parser, args)
    script_name = getScriptName(filename)
    subdir = getSuffixWithParameters(parser, args) + getParamHashSuffix(args)
    log_dir = os.path.join('logs', script_name, subdir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setupLogger(filename, parser=None, args=None):
    script_name = getScriptName(filename)
    log_dir = createLogdir(filename, parser, args)
    logger = logging.getLogger('__main__')
    # parse args
    args = _check_parser_or_args(parser, args)

    fname = 'log.txt'

    log_filepath = os.path.join(log_dir, fname)

    handlers = [
        logging.FileHandler(log_filepath, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(
            # fmt="%(funcName)-12.12s| %(levelname)-5.5s: %(message)s"
            fmt="%(asctime)s | %(funcName)-12.12s | %(levelname)-5.5s: %(message)s",
            datefmt='%Y-%m-%d-%H:%M:%S'
        ))
        logger.addHandler(h)
        from time import gmtime, strftime

    logger.setLevel(args.log_level)

    now = strftime("%a, %d %b %Y %H:%M:%S +0000 (GMT)", gmtime())
    logger.info(f'Running {script_name} @ {now}')
    logger.info(f'Parameters: {json.dumps(vars(args), sort_keys=True)}')
    logger.info(f'Logging to {log_dir}')
    return logger

def getLogdir() -> str :
    log_dir, log_file = os.path.split(logger.handlers[0].baseFilename)
    return log_dir


def _removeIgnoreArgs(args) -> dict:
    args_d = vars(args)
    # Remove args to ignore
    for key in IGNORE_ARGS:
        if key in args_d:
            del args_d[key]
    return args_d


def getNonDefaultArgs(parser, args=None) -> dict:
    if args is None:
        args = parser.parse_args()
    args_d = _removeIgnoreArgs(args)
    # Adapted from https://stackoverflow.com/questions/44542605/python-how-to-get-all-default-values-from-argparse
    # To get all defaults:
    # XXX DO NOT USE SUBPARSERS WITH THIS
    # subparsers mess up with the internal parser dict containing defaults
    # it becomes empty so we can't retrieve default values anymore
    # see https://stackoverflow.com/q/43688450
    all_defaults = {key: parser.get_default(key) for key in args_d}
    # Get non default by comparing with all_defaults
    non_default_args = dict(filter(
        lambda kv: args_d[kv[0]] != all_defaults[kv[0]], args_d.items()))
    return non_default_args


def getSuffixWithParameters(parser=None, args=None):
    args = _check_parser_or_args(parser, args)
    # it is only possible to retrieve default values if parser is passed. 
    # if not, function provides a suffix based on all params
    if parser is not None:
        args = getNonDefaultArgs(parser, args)
    param_list = [f"{key.replace('_','-')}_{val}" for key, val in args.items()]
    param_list.sort()
    # if all params are default, param list is empty
    # add default as the suffix
    # truncate so that filenames do not get too big
    # unix has a filename limit of 255 utf-8 chars
    suffix =  ('__'.join(param_list) or 'default')[:200]

    return suffix


def includeParamSuffix(file_path, parser):
    # get current root name and extension of the given file_path
    root, ext = os.path.splitext(file_path)
    # prepare name of the copy including non default params as a suffix
    copy_path = root + '___' +getSuffixWithParameters(parser)

    return copy_path, ext


def getParamHashSuffix(args):
    args_d = _removeIgnoreArgs(args)
    # serialized version of args
    json_args = json.dumps(args_d, sort_keys=True)
    # take first 7 hex figures of md5 hash as signature
    param_hash = hashlib.md5(
        json_args.encode('utf-8'), usedforsecurity=False
    ).hexdigest()[:7]
    suffix = f'__md5_{param_hash}'
    return suffix


def includeParamHashSuffix(copy_path, args):
    copy_path += getParamHashSuffix(args)
    return copy_path


def saveJson(fname, object, root_dir=None):
    root_dir = root_dir or getLogdir()
    logger = logging.getLogger('__main__')
    fname = fname if fname.endswith('.json') else fname + '.json'
    path = os.path.join(root_dir, fname)
    with open(path, 'w') as fp:
        json.dump(object, fp)
    logger.info(f'Saved to {path} file.')


def saveMetrics(metrics_dict, root_dir=None):
    saveJson('metrics', metrics_dict, root_dir)


def saveParams(basefile, args):
    """
    Save complete set of params to a json file with same basename + .params.json extension
    Keys in the dictionary get sorted before saving.
    """
    logger = logging.getLogger('__main__')
    all_params_file = basefile + '.params.json'
    with open(all_params_file, 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True)
    logger.info(f'Full param set of {basefile} saved at {all_params_file}')


def loadParams(basefile):
    logger = logging.getLogger('__main__')
    all_params_file = basefile + '.params.json'
    logger.info(f'Loading full param set of {basefile} from {all_params_file}')
    with open(all_params_file) as fp:
        args_d = json.load(fp)
    return args_d


def saveCopyWithParams(file_path, parser):
    args = parser.parse_args()
    # prepare name of the copy including non default params as a suffix
    # get current root name and extension of the given file_path
    root, ext = os.path.splitext(file_path)
    # prepare name of the copy including non default params as a suffix
    copy_path = root + '___' +getSuffixWithParameters(parser)

    # add unique hash to fname
    copy_path += getParamHashSuffix(args) + ext

    # now that file path is set
    # save copy and params file
    copy2(file_path, copy_path)
    logger.info(f'Saved copy of {file_path} at {copy_path}')
    saveParams(copy_path, args)


def copy2Logdir(file_path):
    log_dir = getLogdir()
    copy2(file_path, log_dir)
    logger.info(f'Saved copy of {file_path} at {log_dir}')
    param_file = file_path + PARAMS_FILE_EXT
    if os.path.exists(param_file):
        copy2(param_file, log_dir)
        logger.info(f'Saved copy of {param_file} at {log_dir}')


def saveWeightsAndParams(model, file_name, args, root_dir=None):
    """Saves model weights and params to current logdir 
    (or rood_dir if given)
    """
    root_dir = root_dir or getLogdir()
    file_path = utils.saveWeights(model, file_name, root_dir)
    saveParams(file_path, args)


def loadWeightsAndParams(model, file_name, root_dir=None):
    """ Loads model weights and params from current logdir
    (or rood_dir if given)
    """
    root_dir = root_dir or getLogdir()
    file_path = utils.loadWeights(model, file_name, root_dir)
    return loadParams(file_path)


def loadModelMatchingDataset(model, model_fname, dataset_name, logs_dir) -> Optional[dict]:
    trained_models = os.listdir(logs_dir)
    logger.info(f"Looking for classification model trained on {dataset_name}...")
    for folder in trained_models:
        model_path = os.path.join(logs_dir, folder, model_fname)
        # First check if dataset name is in the folder name
        # keyword appears either at the start or after double underscore
        res = re.match(r'(?:\A|__)dataset_(\w+)__',folder)
        if res is not None and res.group(1) == dataset_name:
            return loadWeightsAndParams(model, model_path, root_dir='.')
        # If no match was found maybe the info is only in the params file 
        elif res is None:
            # load params file to check for dataset
            args_d = loadParams(model_path)
            if args_d['dataset'] == dataset_name:
                # Note: set model dir to empty string so that nothing is pre-pended to model_path
                utils.loadWeights(model, model_path, model_dir='.')
                return args_d
    msg = f"No model trained on {dataset_name} was found at {logs_dir}."
    logger.info(msg)
    raise RuntimeError(msg)


def loadClfMatchingDataset(model, model_fname, dataset_name):
    logs_dir = 'logs/main_classif'
    return loadModelMatchingDataset(model, model_fname, dataset_name, logs_dir)


def loadNoiserMatchingDataset(model, model_fname, dataset_name):
    logs_dir = 'logs/main_noiser'
    return loadModelMatchingDataset(model, model_fname, dataset_name, logs_dir)
