import hashlib
import json
import logging
import os
from pathlib import Path
from shutil import copy2
import sys


def numericLogLevel(loglevel:str):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level


def getScriptName(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def getLogdir(filename, parser):
    args = parser.parse_args()
    script_name = getScriptName(filename)
    subdir = getSuffixWithParameters(parser) + getParamHashSuffix(args)
    log_dir = os.path.join('logs', script_name, subdir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setupLogger(filename, parser):
    script_name = getScriptName(filename)
    log_dir = getLogdir(filename, parser)
    logger = logging.getLogger('__main__')
    # parse args
    args = parser.parse_args()

    log_filepath = os.path.join(log_dir, args.logfile)

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
    return logger


def getNonDefaultArgs(parser) -> dict:
    args = parser.parse_args()
    args_d = vars(args)
    # Adapted from https://stackoverflow.com/questions/44542605/python-how-to-get-all-default-values-from-argparse
    # To get all defaults:
    all_defaults = {key: parser.get_default(key) for key in args_d}
    # Get non default by comparing with all_defaults
    non_default_args = dict(filter(
        lambda kv: args_d[kv[0]] != all_defaults[kv[0]], args_d.items()))
    return non_default_args


def getSuffixWithParameters(parser):
    args = getNonDefaultArgs(parser)
    param_list = [f"{key.replace('_','-')}_{val}" for key, val in args.items()]
    param_list.sort()
    # if all params are default, param list is empty
    # add default as the suffix
    suffix =  ('__'.join(param_list) or 'default')
    return suffix


def includeParamSuffix(file_path, parser):
    # get current root name and extension of the given file_path
    root, ext = os.path.splitext(file_path)
    # prepare name of the copy including non default params as a suffix
    copy_path = root + '___' +getSuffixWithParameters(parser)

    return copy_path, ext


def getParamHashSuffix(args):
    # serialized version of args
    json_args = json.dumps(vars(args), sort_keys=True)
    # take first 7 hex figures of md5 hash as signature
    param_hash = hashlib.md5(
        json_args.encode('utf-8'), usedforsecurity=False
    ).hexdigest()[:7]
    suffix = f'__md5_{param_hash}'
    return suffix

def includeParamHashSuffix(copy_path, args):
    copy_path += getParamHashSuffix(args)
    return copy_path

def saveCopyWithParams(file_path, parser):
    args = parser.parse_args()

    # prepare name of the copy including non default params as a suffix
    copy_path, ext = includeParamSuffix(file_path, parser)
    # add unique hash to fname
    copy_path = includeParamHashSuffix(copy_path, args)
    # save complete set of params to a json file with same basename + .params.json extension
    all_params_file = copy_path + '.params.json'

    # now that file paths are set
    # save copy and params file
    copy2(file_path, copy_path+ext)
    with open(all_params_file, 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True)

