# %%

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

# %%
# load neurons

brain = pymaid.get_skids_by_annotation('mw brain neurons')
neurons = pymaid.get_neurons(brain)

# %%
# plot distribution of presynaptic sites per neuron

connectors = neurons.connectors.groupby(['neuron', 'type']).count()
presyn_counts = connectors.loc[(slice(None), 0), :].connector_id.values

fig, ax = plt.subplots(1,1)
sns.histplot(presyn_counts, ax=ax)
ax.set(xlabel='Total Presynaptic Sites', ylabel='Number of Neurons', alpha=0.1)
ax.set_ylim([0,250])

# add different percentiles
ax.axvline(x=np.median(presyn_counts), color='gray', alpha=0.25)
ax.axvline(x=np.quantile(presyn_counts, 0.95), color='gray', alpha=0.25)
ax.axvline(x=np.quantile(presyn_counts, 0.75), color='gray', alpha=0.25)
ax.axvline(x=np.quantile(presyn_counts, 0.25), color='gray', alpha=0.25)
ax.axvline(x=np.quantile(presyn_counts, 0.05), color='gray', alpha=0.25)

# add text for each percentile
ax.text(np.quantile(presyn_counts, 0.95)-10, 255, f"{np.quantile(presyn_counts, 0.95):0.0f}", alpha=0.25)
ax.text(np.quantile(presyn_counts, 0.75)-5, 255, f"{np.quantile(presyn_counts, 0.75):0.0f}", alpha=0.25)
ax.text(np.quantile(presyn_counts, 0.5)-5, 255, f"{np.quantile(presyn_counts, 0.5):0.0f}", alpha=0.5)
ax.text(np.quantile(presyn_counts, 0.25)-5, 255, f"{np.quantile(presyn_counts, 0.25):0.0f}", alpha=0.25)
ax.text(np.quantile(presyn_counts, 0.05)-5, 255, f"{np.quantile(presyn_counts, 0.05):0.0f}", alpha=0.25)

# save plot
plt.savefig('plots/presynaptic-counts_per-neuron.png', format='png', bbox_inches='tight')

# %%
