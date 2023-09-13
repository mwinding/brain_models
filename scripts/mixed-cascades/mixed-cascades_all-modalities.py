# %%
# generate cascades from each sensory modality

from data_settings import data_date, pairs_path
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import pickle

from contools import Promat, Adjacency_matrix, Celltype_Analyzer, Celltype, Cascade_Analyzer
from datetime import datetime

today_date = datetime.today().strftime('%Y-%m-%d')

# load adjacency matrix for cascades
subgraph = ['mw brain and inputs', 'mw brain accessory neurons']
adj_ad_NT = Promat.pull_adj(type_adj='ad', data_date=data_date, subgraph=subgraph)

# load NT identities and apply negative values to rows with GABA or Glutamate
NT = pd.read_csv('data/NT/preliminary-NTs_2023-09-13.csv', index_col=0)
for skid in adj_ad_NT.index:

    try:
        pred = NT.loc[skid, 'predictions']
        if(pred=='Acetylcholine'):
            continue
        if((pred=='GABA')|(pred=='Glutamate')):
            adj_ad_NT.loc[skid] = adj_ad_NT.loc[skid].apply(lambda x: -x if x != 0 else x) # change non-zero values to negative values

    except:
        print(f'skid {skid} has no prediction')

adj_ad_noNT = Promat.pull_adj(type_adj='ad', data_date=data_date, subgraph=subgraph)

# prep start and stop nodes
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [Celltype(name, Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_skids = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

# %%
# cascades from each sensory modality
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
p = 0.05
max_hops = 8
n_init = 1000
simultaneous = True
adj=adj_ad_NT
pairs = Promat.get_pairs(pairs_path=pairs_path)
source_names = order

# run cascades with inhibitory edges
input_hist_list_NT = Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = source_names, stop_skids=output_skids, 
                                                                    adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous, pairs=pairs, pairwise=True)

pickle.dump(input_hist_list_NT, open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_NTs_{today_date}.p', 'wb'))
#input_hist_list_NT = pickle.load(open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_NTs_{today_date}.p', 'rb'))

# run cascades with no inhibitory edges
adj=adj_ad_noNT
input_hist_list_noNT = Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = source_names, stop_skids=output_skids, 
                                                                    adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous, pairs=pairs, pairwise=True)

pickle.dump(input_hist_list_noNT, open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_no-NTs_{today_date}.p', 'wb'))
#input_hist_list_noNT = pickle.load(open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_no-NTs_{today_date}.p', 'rb'))

# %%
# generate mega DataFrame with all data, add Cascade_Analyzer objects, and pickle it
from joblib import Parallel, delayed
from tqdm import tqdm
pairs = Promat.get_pairs(pairs_path=pairs_path)

# save processed cascades based on adj_ad with inhibitory edges
names = [x.name for x in input_hist_list_NT]
skid_hit_hists = [x.skid_hit_hist for x in input_hist_list_NT]

all_data_df = pd.DataFrame([[x] for x in skid_hit_hists], index=names, columns=['skid_hit_hists'])

cascade_objs = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer)(name=all_data_df.index[i], hit_hist=all_data_df.iloc[i, 0], n_init=n_init, pairs=pairs, pairwise=True) for i in tqdm(range(len(all_data_df.index))))
all_data_df['cascade_objs'] = cascade_objs

pickle.dump(all_data_df, open(f'data/cascades/all-sensory-modalities_processed-cascades_{n_init}-n_init_NTs_{today_date}.p', 'wb'))

# save processed cascades based on adj_ad without inhibitory edges
names = [x.name for x in input_hist_list_noNT]
skid_hit_hists = [x.skid_hit_hist for x in input_hist_list_noNT]

all_data_no_NT_df = pd.DataFrame([[x] for x in skid_hit_hists], index=names, columns=['skid_hit_hists'])

cascade_objs = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer)(name=all_data_no_NT_df.index[i], hit_hist=all_data_no_NT_df.iloc[i, 0], n_init=n_init, pairs=pairs, pairwise=True) for i in tqdm(range(len(all_data_no_NT_df.index))))
all_data_no_NT_df['cascade_objs'] = cascade_objs

pickle.dump(all_data_no_NT_df, open(f'data/cascades/all-sensory-modalities_processed-cascades_{n_init}-n_init_no-NTs_{today_date}.p', 'wb'))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

modality = 'visual'
fig, ax = plt.subplots(1,1)
sns.heatmap(all_data_df.loc[modality, 'skid_hit_hists'].sort_values(by=[0,1,2,3,4,5,6,7,8], ascending=False), ax=ax)
plt.savefig(f'plots/cascade-mixed_{modality}.png', bbox_inches='tight')

fig, ax = plt.subplots(1,1)
sns.heatmap(all_data_no_NT_df.loc[modality, 'skid_hit_hists'].sort_values(by=[0,1,2,3,4,5,6,7,8], ascending=False))
plt.savefig(f'plots/cascade-excitatory_{modality}.png', bbox_inches='tight')

# %%
