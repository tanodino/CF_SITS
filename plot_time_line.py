"""
Thanks to:
    - https://stackoverflow.com/questions/32619424/is-it-possible-to-plot-timelines-with-matplotlib/32626852
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random

import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt

def read_date(filename):
    dates = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        for l in lines:
            y = int(l[0:4])
            m = int(l[4:6])
            d = int(l[6::])
            dates.append( pd.to_datetime('%s-%s-%s' % (y, m, d)) )
    return dates

def gen_rand_date():
    year = random.randint(2010, 2018)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return pd.to_datetime('%s-%s-%s' % (year, month, day))


fileName = "dates.txt"
dates = read_date(fileName)
#print(ddd)
#exit()

random.seed(4)
n_events = random.randint(20, 50)

data = pd.DataFrame({"date": dates})

'''
data = pd.DataFrame({
    "feature": [random.randint(1, 10) for x in range(n_events)],
    "flag_special_event": [random.randint(1, 10) == 4 for x in range(n_events)],
    "event_type": [random.randint(0, 312) % 4 for x in range(n_events)],
    "date": [gen_rand_date() for _ in range(n_events)]
})
'''


event_colors = {
    0: '#ffc50f',
    1: '#f3452c',
    2: '#80cacc',
    3: '#8c9f0f',
    4: '#8060cb'}
'''
data['event_color'] = data['event_type'].apply(
    lambda x: event_colors[x]).tolist()

data = data[['date', 'event_type', 'event_color', 'feature', 'flag_special_event']]
data.head(10)
'''

fig, ax = plt.subplots(1, 1)

handles = []
'''
for label, color in event_colors.items():
    handle = mpatches.Patch(
        color=color, 
        label=str(label))

    handles.append(handle)
    
ax.legend(
    handles=handles, 
    ncol=len(handles))
'''
#Stile ticks
fig.autofmt_xdate()
fig.set_size_inches(18, 5)

#Hide y axis 
ax.yaxis.set_visible(False)
ax.get_yaxis().set_ticklabels([])

#Hide spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
#
#ax.xaxis.set_ticks_position('bottom')

#Leave left/right marging on x-axis
min_date = data['date'].min()
max_date = data['date'].max()
day_offset = pd.DateOffset(days=15)
print(day_offset)
#month_offset = pd.DateOffset(months=12)
#ax.set_xlim(min_date - month_offset, max_date + month_offset)
ax.set_xlim(min_date - day_offset, max_date + day_offset)

#Limit y-range
ax.set_ylim(.9, 1.4);

#Time line
timeline_y = 1
ax.axhline(
    timeline_y,
    linewidth=1,
    color='#CCCCCC')

'''
ax.annotate(
    'Timeline', 
    (min_date - day_offset, timeline_y),
    fontsize=15)
#fig
'''
plt.xticks(fontsize=20)
plt.savefig("prova.png")
#exit()

#Feature line
'''
feature_line_y = 1.2
ax.axhline(
    feature_line_y, 
    linewidth=1,
    c='#CCCCCC')

ax.annotate(
    'Feature', 
    (min_date - month_offset, feature_line_y),
    fontsize=15)

fig.set_size_inches(18, 4)
'''
dot_size = 500
dot_special_scale = 2

'''
s = (
    data['flag_special_event'].astype(int) + 1
) ** dot_special_scale * dot_size
'''

ax.scatter(
    data['date'].tolist(),
    [1] * data.shape[0],
    #c=data['event_color'].tolist(),
    s=500,
    marker='o',
    linewidth=1,
    alpha=.5)

plt.savefig("prova.png")

'''
for idx, row_data in data.iterrows():
    if row_data['flag_special_event'] == True:
        ax.annotate(
            'Spe. Ev.', 
            (row_data['date'], 1.12),
            rotation=60)
'''
