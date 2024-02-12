# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: brian2
#     language: python
#     name: python3
# ---

# %%
from src.model import run_exp
import src.utils as utl

# %% [markdown]
# This workflow serves as an example of how to use the spiking neural network
# model to explore the connectome.
#
# It is helpful to be familiar with
# [basic python data structures](https://neuropsychology.github.io/NeuroKit/resources/learn_python.html)
# as well as
# [pandas DataFrames](https://pandas.pydata.org/docs/user_guide/10min.html). 
#
# # Loading the data
#
# ## Custom neuron names
# Custom neuron names used in the lab can be edited in this 
# [spread sheet in the OneDrive](https://maxplanckflorida-my.sharepoint.com/:x:/g/personal/murakamik_mpfi_org/EeX_NEJ2kaVMvcHdbHPZkPcBG9IwOMWwkEingWCFmnv_SA?e=azcslm).
# The shared document is intended as the reference for the entire lab.
# The file `flywire_ids_630.xlsx` in this git repository may be outdated.
#
# There are two types of spread sheets:
# - pairs: If the sister neurons are known, flywire IDs for each of them is entered in a separate column. The suffixes `_r` and `_l` will be automatically appended to the neuron names.
# - single: If the sister neuron is not known, 
# there is one flywire ID column. The name is used as is.
#
# Please make sure to follow the consenus in the lab about naming conventions when assigning names for neurons here.
#
# ## Using multiple CPUs
# The computational load can be distributed over multiple CPU threads.
# To choose the number of threads, set `n_procs` to a number not higher than available on your computer.
# More threads will speed up the computation, but also increase memory usage.

# %%
# set up environment
path_comp = '../data/2023_03_23_completeness_630_final.csv'
path_con = '../data/2023_03_23_connectivity_630_final.parquet'
path_res = '../results/example/'

config = {
    'path_comp'         : path_comp, 
    'path_con'          : path_con, 
    'path_res'          : path_res,     # store results here
    'n_proc'            : 4,            # number of CPUs to use, -1 uses all available CPU
}

# %%
# custon neuron names
path_name = '../data/flywire_ids_630.xlsx'

# sheet names of the xls file to include
sheets_pair = [ # pair of neurons (left+right per row)
    'stop',
    'walk',
    'walk_outputs',
    ]

sheets_single = [ # single neurons (one per row)
    # 'sugar', 
    # 'ovidn', 
    # 'bitter', 
    ]

name2flyid = utl.create_name_dict(path_name, path_comp, sheets_pair, sheets_single)

# %%
# lists of neuron groups
l_p9 = ['P9_l', 'P9_r']
l_bb = ['BB_r', 'BB_l']
l_cdn = ['P9-cDN1_l', 'P9-cDN1_r']

# %% [markdown]
# # run experiments

# %%
# P9 activation
instructions = [ 
    (0, 'stim', l_p9), 
    (1, 'end', [])
    ]

run_exp(exp_name='P9', exp_inst=instructions, name2flyid=name2flyid, **config, force_overwrite=True)

# %%
# more complex instuctions: 
# (i) activate P9 
# (ii) after some delay, activate BB 
# (iii) after some more delay, silence BB

# caveate
# - silenced neurons cannot be unsilenced
# - silenced neurons will still spike, if they also have a Poisson input (stim)

instructions = [ 
    (0, 'stim', l_p9), 
    (0.25, 'stim', l_bb),
    (0.75, 'slnc', l_bb),
    (1, 'end', []),
    ]
run_exp(exp_name='P9+BB_slnc', exp_inst=instructions, name2flyid=name2flyid, **config, force_overwrite=True)

# %%
# changing model parameters
from src.model import default_params as params
from brian2 import Hz

instructions = [ 
    (0, 'stim', l_p9), 
    (1, 'end', [])
    ]

params['r_poi'] = 250 * Hz

run_exp(exp_name='P9_ultra', exp_inst=instructions, name2flyid=name2flyid, **config, params=params, force_overwrite=True)

# %%
# use 2 different stimulation frequencies
# frequency for stim2 is controlled via params['r_poi2']

instructions = [ 
    (0, 'stim', l_p9), 
    (0, 'stim2', l_bb),
    (1, 'end', []),
    ]

run_exp(exp_name='P9+BB2', exp_inst=instructions, name2flyid=name2flyid, **config, force_overwrite=True)

# %% [markdown]
# # Process results
#
# The results from different simulations can be combined and visualized in different ways.
# Below, we show how to
# - save the average rate across trials and standard deviation
# - create spike raster plots
# - plot line plots and heat maps with firing rate changes throughout the simulation 

# %%
# choose experiments to load
outputs = [
    '../results/example/P9+BB_slnc.parquet',
    '../results/example/P9.parquet',
]

# load spike times, calculate rate + standard deviation
df_spkt = utl.load_exps(outputs)
df_rate, df_std = utl.get_rate(df_spkt, duration=1)

# convert IDs to custom names, if possible
df_rate = utl.rename_index(df_rate, name2flyid)
df_std = utl.rename_index(df_std, name2flyid)

# save as spreadsheets
utl.save_xls(df_rate, '../results/example/rate.xlsx')
utl.save_xls(df_std, '../results/example/rate_std.xlsx')

# %%
# raster plots
neu = l_p9 + l_bb + l_cdn
utl.plot_raster(df_spkt, neu, name2flyid=name2flyid, xlims=(0, 1))

# %%
# firing rates
utl.plot_rate(df_spkt, neu, xlims=(0, 1), name2flyid=name2flyid)

# %%
# select top 20 neurons
top20 = df_rate.sort_values(by='P9', ascending=False).head(20)
top20

# %%
# plot heatmap
utl.plot_rate_heatmap(df_spkt, top20.index, xlims=(0, 1), name2flyid=name2flyid, do_zscore=False)

# %% [markdown]
# # Graph representation
# The results from a simulation can be visualized as a graph. The nodes are neurons and the edges are synapse counts. Using a graph visualization software, the node size can be scaled to represent the firing rate and the edge size to the synapse count.
#
# Graph files are created in two steps:
# 1. The full connectome is loaded in a `DiGraph` object using [NetworkX](https://networkx.org/).
# 2. The subset of neurons active in a given experiment is selected from the full graph and written to disk.
#
# The resulting `*.gexf` file can be loaded for visualization with [Gephi](https://gephi.org/),
# or the `*.gml` file with [Cytoscape](https://cytoscape.org/).

# %%
# load connectome into graph
G = utl.get_full_graph(path_comp, path_con)
print(len(G.nodes))

# %%
# select subgraph based on simulation results
output = '../results/example/P9.parquet'
utl.write_graph(G, output, name2flyid=name2flyid)
