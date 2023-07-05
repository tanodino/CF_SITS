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
results = pd.concat(results, axis=0, ignore_index=True)
results
#%% drop epochs=1 runs (usuly test runs)
results = results.query("epochs > 10")
#%% save to csv table
from time import gmtime, strftime
fname = 'results_compiled_' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '.csv'
results.to_csv(os.path.join(base_path, fname), index=False)

#%% add compromise metric
results['plausibility_x_validity'] = results['plausibility'] * results['validity']
#%% Drop not useful columns
keep = [
    'dataset', 'discr_arch',
    'shrink', 'loss_cl_type', 'margin', 'reg_gen', 'reg_uni', 
    ]
metrics = {
    'validity': 'high', # higher is better
    'rel_proximity_L2': 'low', # lower is better
    'proximity_L2': 'low', # lower is better
    'compactness_1e-2' : 'high', # Higher is better
    'stability': 'low', # Lower is better
    'plausibility': 'high',  # Higher is better
    'plausibility_x_validity': 'high'
    }
table = results.filter(regex='|'.join(keep+list(metrics.keys())))


table
#%%
# pretty formatting
def highlight(styler):
    styler.format('{:.5g}', subset=['margin', 'reg_gen', 'reg_uni'])
    styler.format('{:3.1%}',
                  subset=['validity', 'plausibility',
                          'plausibility_x_validity', 'compactness_1e-2'])
    styler.format('{:.3g}', subset=['stability', 'proximity_L2', 'rel_proximity_L2'])
    for name in metrics:
        if metrics[name] == 'low':
            styler.highlight_min(subset=[name], color='green', axis=0)
        elif metrics[name] =='high':
            styler.highlight_max(subset=[name], color='green', axis=0)
    return styler


display(table.query("discr_arch=='TempCNN'").style.pipe(highlight))
display(table.query("discr_arch=='Inception'").style.pipe(highlight))

# %%
