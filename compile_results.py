""" Reads all results from main_noiser together in a pandas dataframe

Designed to be run in interactive mode.

"""
#%%
import os
import numpy as np
import pandas as pd

def load_as_record(path):
    with open(path) as fp:
        return pd.read_json(f'[{fp.read()}]', orient='records')

base_path = os.path.join('logs', 'main_noiser')
available_noisers = list(filter(
    lambda _: os.path.isdir(os.path.join(base_path,_)), 
    os.listdir(base_path)
))
results = []
for folder in available_noisers:
    params_path = os.path.join(base_path, folder, 'noiser_weights.params.json')
    metrics_path = os.path.join(base_path, folder, 'metrics.json')
    params = load_as_record(params_path)
    metrics = load_as_record(metrics_path)
    row = pd.concat([params, metrics], axis=1)
    results.append(row)
results = pd.concat(results, axis=0)
results
#%% drop epochs=1 runs (usuly test runs)
results.query("epochs > 10")
results.epochs
#%% save to csv table
from time import gmtime, strftime
fname = 'results_compiled_' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '.csv'
results.to_csv(os.path.join(base_path, fname))
#%% Drop not useful columns
keep = ['dataset', 'shrink', 'loss_cl_type', 'margin', 'reg_gen', 'reg_uni', 'validity', 'proximity_L2', 'compactness_1e-2', 'plausibility_L2', 'sensitivity']

results.filter(regex='|'.join(keep))
#%%