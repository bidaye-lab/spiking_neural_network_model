# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: brian2
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from src import (
    datahandler as dah,
    model as mod,
    graphs as gra,
    visualize as vis,
    analysis as ana,
)


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
# ## Connectome data
# To construct the spiking neural network model based on connectome data,
# we use two data structures:
# - `ds_ids: pd.Series`\
#     A pandas Series with index increasing from 0...N and neuron IDs.
#     The index is in the following refered to as the canonical ID, the values are the database IDs. Under the hood, `brian2` uses the canonical IDs to refer to neurons.
# - `df_con: pd.DataFrame`\
#     A pandas DataFrame with columns
#     - `pre: int` canonical ID of presynnaptic neuron
#     - `post: int` canonical ID of postsynaptic neuron
#     - `w: int` synnapse number including sign (positive for excitatory, negative for inhibitory)
#
# To create the model for other connectomes, the data needs to be loaded into these data structures.

# %%
# set up environment
path_comp = "../data/2023_03_23_completeness_630_final.csv"
path_con = "../data/2023_03_23_connectivity_630_final.parquet"
path_res = "../results/example/"

# load flywire connectome
ds_ids = dah.load_flywire_ids(path_comp)
df_con = dah.load_flywire_connectivity(path_con, ds_ids)

# %% [markdown]
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

# %%
# custon neuron names
path_name = "../data/flywire_ids_630.xlsx"

# sheet names of the xls file to include
sheets_pair = [  # pair of neurons (left+right per row)
    "stop",
    "walk",
    "walk_outputs",
]

sheets_single = [  # single neurons (one per row)
    # 'sugar',
    # 'ovidn',
    # 'bitter',
]

name2id = dah.create_name_dict(path_name, ds_ids, sheets_pair, sheets_single)

# %% [markdown]
# # run experiments
# ## Using custom neuron names
# Having defined intuitive custom neuron names allows us to use these instead of the complicated database IDs to refer to neurons
# in our simulations. Below, we collect some of the neurons we want to excite or silence, e.g., `l_p9 = ['P9_l', 'P9_r']`.
#
# If we use custom neuron names, we need to also pass the `name2id` dictionary to the `run_exp` function.
#
# ## Using multiple CPUs
# The computational load can be distributed over multiple CPU threads.
# To choose the number of threads, set `n_procs` to a number not higher than available on your computer.
# More threads will speed up the computation, but also increase memory usage.
#
# We collect these settings in the `run_exp_kw_args` dictionary to pass them more concisely to the `run_exp` function.

# %%
# lists of neuron groups
l_p9 = ["P9_l", "P9_r"]
l_bb = ["BB_r", "BB_l"]
l_cdn = ["P9-cDN1_l", "P9-cDN1_r"]

# settings to apply in all following simulations
run_exp_kw_args = {
    "ds_ids": ds_ids,  # neuron database IDs
    "df_con": df_con,  # connectivity data
    "path_res": path_res,  # store results here
    "n_proc": 4,  # number of CPUs to use, -1 uses all available CPU
    "name2id": name2id,  # dictionary to map neuron names to ids
    "force_overwrite": True,  # if true, overwrite existing results
}

# %%
# P9 activation
instructions = [(0, "stim", l_p9), (1, "end", [])]

mod.run_exp(exp_name="P9", exp_inst=instructions, **run_exp_kw_args)

# %%
# more complex instuctions:
# (i) activate P9
# (ii) after some delay, activate BB
# (iii) after some more delay, silence BB

# caveate
# - silenced neurons cannot be unsilenced
# - silenced neurons will still spike, if they also have a Poisson input (stim)

instructions = [
    (0, "stim", l_p9),
    (0.25, "stim", l_bb),
    (0.75, "slnc", l_bb),
    (1, "end", []),
]
mod.run_exp(exp_name="P9+BB_slnc", exp_inst=instructions, **run_exp_kw_args)

# %%
# changing model parameters
from src.model import default_params as params
from brian2 import Hz

instructions = [(0, "stim", l_p9), (1, "end", [])]

params["r_poi"] = 250 * Hz

mod.run_exp(exp_name="P9_ultra", exp_inst=instructions, **run_exp_kw_args)

# %%
# use 2 different stimulation frequencies
# frequency for stim2 is controlled via params['r_poi2']

instructions = [
    (0, "stim", l_p9),
    (0, "stim2", l_bb),
    (1, "end", []),
]

mod.run_exp(exp_name="P9+BB2", exp_inst=instructions, **run_exp_kw_args)

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
    "../results/example/P9+BB_slnc.parquet",
    "../results/example/P9.parquet",
]

# load spike times, calculate rate + standard deviation
df_spkt = ana.load_exps(outputs)
df_rate, df_std = ana.get_rate(df_spkt, duration=1)

# convert IDs to custom names, if possible
df_rate = ana.rename_index(df_rate, name2id)
df_std = ana.rename_index(df_std, name2id)

# save as spreadsheets
ana.save_xls(df_rate, "../results/example/rate.xlsx")
ana.save_xls(df_std, "../results/example/rate_std.xlsx")

# %% [markdown]
# ## Plots using spike times in `df_spkt`

# %%
# raster plots
neu = l_p9 + l_bb + l_cdn
vis.plot_raster(df_spkt, neu, name2id=name2id, xlims=(0, 1))

# %%
# firing rates
vis.plot_rate(df_spkt, neu, name2id=name2id, xlims=(0, 1))

# %%
# select top 20 neurons
top20 = df_rate.sort_values(by="P9", ascending=False).head(20)
top20

# %%
# plot heatmap
vis.plot_rate_heatmap(
    df_spkt, top20.index, xlims=(0, 1), name2id=name2id, do_zscore=False
)

# %% [markdown]
# ## Plots using average firing rates in `df_rate`

# %%
# select only neurons with custom names
named_neurons = df_rate.index.isin(name2id)
df_rate_named = df_rate.loc[named_neurons]

# plot firing rate matrix
vis.firing_rate_matrix(df_rate_named)

# %%
# subtract P9 firing rate from all neurons
df_change = df_rate.subtract(df_rate.loc[:, "P9"], axis=0)

# ignore absolute changes in firing rate smaller than some threshold
df_change = df_change[(df_change.abs() >= 5).any(axis=1)]

# plot difference matrix
vis.firing_rate_matrix(df_change, rate_change=True)

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
G = gra.get_full_graph(ds_ids, df_con)
print(len(G.nodes))

# %%
# select subgraph based on simulation results
output = "../results/example/P9.parquet"
gra.write_graph(G, output, name2id=name2id)
