""" Reads all results from main_noiser together in a pandas dataframe

Designed to be run in interactive mode.

For the plots to run make sure you have these packages installed:
conda install seaborn nbformat plotly

"""
#%% First read all available results
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import display, HTML, Markdown

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
    try:
        params_path = os.path.join(base_path, folder, 'noiser_weights.params.json')
        metrics_path = os.path.join(base_path, folder, 'metrics.json')
        params = load_as_record(params_path)
        metrics = load_as_record(metrics_path)
        row = pd.concat([params, metrics], axis=1)
        results.append(row)
    except FileNotFoundError:
        pass
results = pd.concat(results, axis=0, ignore_index=True)
# results
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
    'loss_cl_type', 
    'shrink', 
    'margin', 'reg_gen', 'reg_uni', 
    ]
metrics = {
    'rel_proximity_L2': 'low', # lower is better
    'proximity_L2': 'low', # lower is better
    'stability': 'low', # Lower is better
    'compactness_1e-2' : 'high', # Higher is better
    'compactness_1e-8' : 'high', # Higher is better
    'plausibility': 'high',  # Higher is better
    'validity': 'high', # higher is better
    'plausibility_x_validity': 'high'
    }
table = results.loc[:,keep+list(metrics.keys())].sort_values(keep)


table
#%%
# pretty formatting
def highlight(styler):
    styler.format('{:.5g}', subset=['margin', 'reg_gen', 'reg_uni'])
    styler.format('{:3.1%}',
                  subset=['validity', 'plausibility',
                          'plausibility_x_validity', 
                          'compactness_1e-2', 'compactness_1e-8'])
    styler.format('{:.3g}', subset=['stability', 'proximity_L2', 'rel_proximity_L2'])
    cmap_l='YlGn_r'
    cmap_h = cmap_l.split("_")[0] + ("_r" if not cmap_l.endswith("_r") else "")
    for name in metrics:
        if metrics[name] == 'low':
            styler.background_gradient(cmap=cmap_l, axis=0, subset=[name])
            styler.highlight_min(subset=[name], color='light green', axis=0)
        elif metrics[name] =='high':
            styler.background_gradient(cmap=cmap_h, axis=0, subset=[name])
            styler.highlight_max(subset=[name], color='light green', axis=0)
    return styler

#%% Example viz with query
# display(table.query("discr_arch=='TempCNN'").style.pipe(highlight))
# display(table.query("discr_arch=='Inception'").style.pipe(highlight))

# %% Example viz with groupby 
# to separate groups of attributes in multiple tables

# [display(df.style.pipe(highlight)
#          .set_caption(f"TempCNN, shrink={val}")
#          )
#  for val, df in table.query("discr_arch=='TempCNN'").groupby(['shrink'])]

# [display(df.style.pipe(highlight)
#          .set_caption(f"TempCNN, {val}")
#          )
#     for val, df in table.groupby(['discr_arch', 'loss_cl_type'])]
#%% A function to facilitate creating tables like the examples above
def pprint_table(table, query=None, groupby=None, sort=None):
    caption = ""
    if sort:
        table = table.sort_values(sort)
    if query:
        table = table.query(query)
        caption += query
    if groupby:
        caption += ", "
        for val, df in table.groupby(groupby):
            try:
                this_caption = caption + ", ".join(
                    [f"{k}={v}" for k, v in zip(groupby, val)]
                )
            except TypeError:
                this_caption = caption + f"{groupby[0]}={val}"

            display(df.drop(columns=groupby).style.pipe(highlight)
                    .set_caption(this_caption))
    else:
        display(table.style.pipe(highlight))
#%%
pprint_table(table, "discr_arch=='TempCNN'", ['shrink','loss_cl_type'])
# pprint_table(table, "discr_arch=='TempCNN' & shrink==True", ['loss_cl_type'])
# pprint_table(table, "discr_arch=='Inception'", ['shrink','loss_cl_type'])


#%% Margin sweep
# table to justify the choice of margin = 0.1
display(Markdown("# Margin sweep"))
display(HTML('This table results form a margin sweep with default regularization values.'))
query="discr_arch=='Inception' & shrink==True & loss_cl_type=='margin'& reg_gen==0.0002 & reg_uni==691.2"
pprint_table(table, query)
df = table.query(query)
fig = px.line(data_frame=df,
        x='margin',
        y=['proximity_L2', 'stability', 'compactness_1e-8', 'plausibility', 'validity'], 
        title=query,
        markers=True,
        log_x=True,
        width=1100, height=520,
        # line_dash_sequence=['dot', 'dash'],
        )
fig.show()
display(HTML(
    'We observe plausibility increase then decrease with increasing margin (up to 1, zoom the plot between 0.001 and 1), signaling 0.1 as a good compromise.'
    ))
#%%
# display(HTML(
#     'These tables show all reg values explored in each of the four (loss/shrink) scenarios.'
# ))
# pprint_table(table, "discr_arch=='Inception' & margin==0.1", ['shrink','loss_cl_type'])
#%%
display(Markdown(
"""## Metrics vs regularizations, loss type and softshrink use
These plots show metrics vs reg values explored in each of the four (loss/shrink) scenarios."""
))
display(Markdown(
"""
### Impact of shrink
- little difference in most scenarios
- makes stability worse when (loss=margin & low reg uni & high reg gen)
- expected to improve compactness, but the oposite was observed in some cases.
- same or better plausibility when loss=log; Same or worse when loss=margin.
- no difference or better validity

### Impact of reg_uni
- plausibility and compactness tend to improve with increasing reg_uni
- Validity is not much affected, except when reg_uni==100 & reg_gen >=0.1 (mid-high)
- proximity and stability seem to improve with increasing reg-uni up to 100, though current results are still incomplete.


"""
))
df = table.query("discr_arch=='Inception' & margin==0.1"
                 " & reg_gen!=0.5 & reg_gen!=0.0002"
                 )
fig = px.line(data_frame=df.sort_values(['reg_uni', 'reg_gen', 'loss_cl_type']),
        x='reg_uni',
        y=['compactness_1e-8','compactness_1e-2'], 
        facet_row="reg_gen",
        facet_col="loss_cl_type",
        line_dash='shrink',
        markers=True,
        log_x=True,
        width=1100, height=520,
        # line_dash_sequence=['dot', 'dash'],
        )
fig.show()

fig = px.line(data_frame=df.sort_values(['reg_uni', 'reg_gen', 'loss_cl_type']),
        x='reg_uni',
        y=['proximity_L2', 'stability', 'compactness_1e-8', 'plausibility', 'validity'], 
        facet_row="reg_gen",
        facet_col="loss_cl_type",
        line_dash='shrink',
        markers='shrink',
        log_x=True,
        width=1100, height=520,
        # line_dash_sequence=['dot', 'dash'],
        )
fig.show()


#%%
display(Markdown(
"""
### Impact of reg_gen
- reg_gen has less impact when reg uni is high (1000)
- lower reg gen seems to be better for all metrics
"""
))
df = table.query("discr_arch=='Inception' & margin==0.1"
                 " & reg_uni!=0.28 & reg_uni!=691.2"
                 )

fig = px.line(data_frame=df.sort_values(['reg_uni', 'reg_gen', 'loss_cl_type']),
        x='reg_gen',
        y=['proximity_L2', 'stability', 'compactness_1e-8', 'plausibility', 'validity'], 
        facet_row="reg_uni",
        facet_col="loss_cl_type",
        line_dash='shrink',
        markers='shrink',
        log_x=True,
        width=1100, height=780,
        )
fig.show()
#%%
display(Markdown(
"""
### Impact of margin loss
- Validity: mostly same. Margin is worse for (shrink=False & reg_gen >=0.01)
- Plausibility: differences appear as reg_uni >=10. Gain is inconsistent but can be large.
- Compactness: gains seem to follow those of plausibility.
- Proximity: small differences, inconsistent.
- Stability: inconsistent differences.
"""
))
df = table.query("discr_arch=='Inception' & margin==0.1"
                 " & reg_uni!=0.28 & reg_uni!=691.2"
                 )

fig = px.line(data_frame=df.sort_values(['reg_uni', 'reg_gen', 'loss_cl_type']),
        x='reg_gen',
        y=['proximity_L2', 'stability', 'compactness_1e-8', 'plausibility', 'validity'], 
        facet_row="reg_uni",
        facet_col="shrink",
        line_dash='loss_cl_type',
        markers=True,
        log_x=True,
        width=1100, height=780,
        )
fig.show()


#%% Examples of seaborn scatter plots
# df = table.query("discr_arch=='Inception' & margin==0.1")
# grid = sns.relplot(data=df,
#             x='reg_uni', y='reg_gen', 
#             col="loss_cl_type",
#             style="shrink",
#             size='validity',
#             hue='plausibility',
#             palette='flare',
#             height=4,
#             )
# grid.set(yscale='log', xscale='log')
# grid.figure.suptitle('higher is better')
# #%%
# df = table.query("discr_arch=='Inception' & margin==0.1")
# grid = sns.relplot(data=df,
#             x='reg_uni', y='reg_gen', 
#             col="loss_cl_type",
#             row="shrink",
#             size='stability',
#             hue='proximity_L2',
#             palette='flare_r',
#             height=4,
#             )
# grid.set(yscale='log', xscale='log')
# grid.figure.suptitle('lower is better')

# %%
