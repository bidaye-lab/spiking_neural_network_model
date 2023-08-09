from pathlib import Path

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import matplotlib.pylab as plt
from matplotlib.colors import CenteredNorm
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['savefig.facecolor'] = 'w'

import pickle

try:
    import networkx as nx
except ImportError:
    nx = None


def print_connectivity(config, neu1, neu2, name2flyid=dict()):
    # TODO

    df = pd.read_parquet(config['path_con'])

    ds = df.loc[ (df.loc[:, 'Presynaptic_ID'] == name2flyid.get(neu1, neu1)) & (df.loc[:, 'Postsynaptic_ID'] == name2flyid.get(neu2, neu2))]
    ec = ds.loc[:, 'Excitatory x Connectivity']
    c = ec.item() if len(ec) else 0
    print(f'pre: {neu1} {name2flyid[neu1]} -> post: {neu2} {name2flyid[neu2]} = {c}')

def csv2parquet(csv):
    '''Convert .csv to .parquet file for compression

    Necessary for the connectivity data
    Output file is written as filename.parquet


    Parameters
    ----------
    csv : str
        Path to csv file
    '''

    p_csv = Path(csv)
    p_parquet = p_csv.with_suffix('.parquet')
    print('INFO: Reading {}'.format(p_csv))
    df = pd.read_csv(p_csv)
    print('INFO: Writing {}'.format(p_parquet))

    df.to_parquet(p_parquet, compression='brotli')


def load_xls(path_names, sheets_pair, sheets_single):
    '''Load XLS file containing neuron name to flywire ID mapping

    Parameters
    ----------
    path_names : str
        Path to XLS file
    sheets_pair : list
        List of sheet names containing neuron pairs
    sheets_single : list
        List of sheet names containing individual neurons

    Returns
    -------
    df_pair : pd.DataFrame
        Dataframe mirroring the pair sheets
    df_sing : pd.DataFrame
        Dataframe mirroring the single sheets
    '''
    
    # sheets with left/right pairs (name | ID left | ID right)
    
    if sheets_pair:
        d_pair = pd.read_excel( 
            path_names,
            sheet_name=sheets_pair,
            dtype={'ID_left': str, 'ID_right': str}
            )
        df_pair = pd.concat(d_pair, ignore_index=True).dropna(how='all')
    else:
        d_pair = dict()
        df_pair = pd.DataFrame(columns=['Name', 'ID'])

    # sheets with single neurons (name | ID)
    if sheets_single:
        d_single = pd.read_excel( 
            path_names,
            sheet_name=sheets_single,
            dtype={'ID': str}
            )
        df_single = pd.concat(d_single, ignore_index=True).dropna(how='all')
    else:
        d_single = dict()
        df_single = pd.DataFrame(columns=['Name', 'ID'])

    # print info
    print('INFO: Loaded sheets ...')
    for i in [*d_pair.keys(), *d_single.keys()]:
        print('      ... {}'.format(i))
    print()

    return df_pair, df_single

def check_unique(df_pair, df_single):
    '''Check for double definition of neuron names and flywire IDs.
    Prints warning if duplicates found

    Parameters
    ----------
    df_pair : pd.DataFrame
        Names and flywire IDs for neuron pairs
    df_single : pd.DataFrame
        Names and flywire IDs for single neurons
    '''

    # check names
    ds = pd.concat( (df_single.loc[:, 'Name'], df_pair.loc[:, 'Name']), ignore_index=True) # merge names from both dataframes
    dup = ds.loc[ ds.duplicated(keep=False )] # series with duplicate values
    if dup.empty:
        print('INFO: All names are unique')
    else:
        print('WARNING: Found duplicate names:')
        print(dup)
    print()

    # check flywire IDs
    ds = pd.concat( (df_single.loc[:, 'ID'], df_pair.loc[:, 'ID_left'], df_pair.loc[:, 'ID_right']), ignore_index=True) # merge IDs from both dataframes
    ds = ds.dropna() # igrone nan for IDs
    dup = ds.loc[ ds.duplicated(keep=False )] # series with duplicate values
    if dup.empty:
        print('INFO: All IDs are unique')
    else:
        print('WARNING: Found duplicate IDs:')
        ds_n = pd.concat( (df_single.loc[:, 'Name'], df_pair.loc[:, 'Name'], df_pair.loc[:, 'Name']), ignore_index=True) # series with names of same structure as ds
        print(pd.concat( (ds_n.loc[dup.index], dup), axis=1 ) ) # pring names and IDs
    print()

def assign(name2id, name, id):
    '''Assign custom name to flywire ID
    Prints warning if flywire ID could not be interpreted as integer and skips those

    Parameters
    ----------
    name2id : dict
        dictionary to which to assign to
    name : str
        custom neuron name
    id : str
        str to be interpreted as flywired ID
    '''
    try:
        name2id[name] = int(id)
    except ValueError:
        print('WARNING: Did not assign any flywire ID to name {:>15}: {} is not a valid integer'.format(name, id))


def check_ids(name2flyid, path_comp):
    '''Check if flywire IDs of custom names appear in completeness file

    Parameters
    ----------
    name2flyid : dict
        Mapping between custom names and flywire IDs
    path_comp : str
        Path to completeness dataframe
    '''

    # check if IDs are correct: if everything is correct, nothing is printed
    df_comp = pd.read_csv(path_comp, index_col=0) # df with all IDs
    ids_all = df_comp.index # all flywire ids

    warn = False
    for k, v in name2flyid.items():
        if not v in ids_all:
            print('ERROR: ID {} for neuron {:>15} not found. Please provide correct flywire ID'.format(str(v), k))
            warn = True
    if not warn:
        print('INFO: IDs appear to match with {}'.format(path_comp))
    print()
  

def create_name_dict(path_name, path_comp, sheets_pair, sheets_single):
    '''Generate dictionary mapping custom neuron names to flywire IDs

    Parameters
    ----------
    path_name : str
        Path to XLS files containing neuron names and flywire IDs
    path_comp : str
        Path to completeness dataframe
    sheets_pair : list
        List of sheet names containing neuron pairs
    sheets_single : list
        List of sheet names containing individual neurons

    Returns
    -------
    name2flyid : dict
        Mapping between custom neuron names to flywire IDs
    '''

    # load XLS file into dataframes
    df_pair, df_single = load_xls(path_name, sheets_pair, sheets_single)
    
    # check for duplicates
    check_unique(df_pair, df_single)

    # create dictionary mapping custom names to flywire IDs
    name2flyid = dict()

    for i in df_pair.index: # left/right pairs
        n, id_l, id_r = df_pair.loc[i, ['Name', 'ID_left', 'ID_right']]
        n_l, n_r = '{}_l'.format(n), '{}_r'.format(n)
        assign(name2flyid, n_l, id_l)
        assign(name2flyid, n_r, id_r)

    for i in df_single.index: # single neurons
        n, id = df_single.loc[i, ['Name', 'ID']]
        assign(name2flyid, n, id)    

    print( 'Declared {} names for neurons'.format(len(name2flyid)))
    print()

    # check if all flywire IDs appar in neurotransmitter data
    check_ids(name2flyid, path_comp)

    return name2flyid

def useful_mappings(name2flyid, path_comp):
    '''Generate other useful mappings between custom names, flywire IDs
    and canonical IDs (starting with 0, will be equivalent to brian neuron IDs)

    Parameters
    ----------
    name2flyid : dict
        Mapping between custom neuron names and flywire IDs
    path_comp : str
        Path to completeness dataframe

    Returns
    -------
    flyid2name : dict
        Inverted name2flyid dictionary
    flyid2i : dict
        Mapping between flywire IDs and canonical IDs
    i2flyid : dict
        Inverted flyid2i dictionary
    name2i : dict
        Mapping between custom neuron names and canonical IDs
    i2name : dict
        Inverted name2s dictionary
    name_flyid2i : dict
        Mapping of custom neuron names and flywire IDs to canonical IDs
    '''

    flyid2name = { j: i for i, j in name2flyid.items() } # flywire ID: custom name

    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe

    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: biran ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    name2i = {i: flyid2i[j] for i, j in name2flyid.items() } # custom name: brian ID
    i2name = {j: i for i, j in name2i.items() } # brian ID: custom name

    name_flyid2i = name2i | flyid2i # union of dicts

    return flyid2name, flyid2i, i2flyid, name2i, i2name, name_flyid2i


##########
# analysis
def load_exps(l_prq):
    '''Load simulation results from disk

    Parameters
    ----------
    l_prq : list
        List of pickle files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs, stim_ids = [], dict()
    for p in l_prq:

        # ensure path object
        p = Path(p)

        # load spike data from parquet file
        df = pd.read_parquet(p)
        df.loc[:, 't'] = df.loc[:, 't'].astype(float)
        dfs.append(df)

        # load pickle for metadata
        with open(p.with_suffix('.pickle'), 'rb') as f:
            pkl = pickle.load(f)

        # get stimulated neurons
        df_inst = pkl['df_inst']
        rows = df_inst.loc[:, 'mode'].str.startswith('stim')
        ids = [ i for j in df_inst.loc[rows].loc[:, 'id'] for i in j]
        stim_ids[pkl['exp_name']] = ids


    df = pd.concat(dfs)
    # TODO: attrs is experimental, find another way https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html#pandas.DataFrame.attrs
    df.attrs['stim_ids'] = stim_ids

    return df

def get_rate(df, duration):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    duration : float
        Trial duration in seconds

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name'):
        for f, df_f in df_e.groupby('flywire_id'):

            r = []
            for _, df_t in df_f.groupby('trial'):
                r.append(len(df_t) / duration)
            r = np.array(r)

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')

    df_rate = df_rate.fillna(0)
    df_std = df_std.fillna(0)

    return df_rate, df_std

def rename_index(df, name2flyid):
    '''Rename flywire IDs to custom neuron names in index
    Also sort index and columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with flywire IDs as index
    name2flyid : dict
        Mapping between custom neuron names and flywire IDs

    Returns
    -------
    df : pd.DataFrame
        Renamed and sorted dataframe
    '''

    # replace flywire IDs with custom names
    flyid2name = {v: k for k, v in name2flyid.items()}
    df = df.rename(index=flyid2name)

    # sort: str first (i.e. custom names), then int (i.e. flywire IDs)
    df.index = df.index.astype(str)
    df = df.loc[
        sorted(sorted(df.index.astype(str)), key=lambda x: (x[0].isdigit(), x)), 
        sorted(df.columns.sort_values(), key=lambda x: len(x.split('+')))
        ]
    
    return df

def save_xls(df, path):
    '''Save DataFrame as xls file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experiments as columns and neurons as indixes
    path : str
        Filename of the xls file
    '''

    print('INFO: saving {} experiments to {}'.format(len(df.columns), path))

    with pd.ExcelWriter(path, mode='w', engine='xlsxwriter') as w:

        # write to file
        df.to_excel(w, sheet_name='all_experiments')

        # formatting in the xlsx file
        wb = w.book

        # set floating point display precision here (excel format)
        fmt = wb.add_format({'num_format': '#,##0.0'}) 
        for _, ws in w.sheets.items():
            ws.set_column(1, 1, 10, fmt)
            ws.freeze_panes(1, 1)


##########
# plotting
def plot_raster(df_spkt, neu, name2flyid=dict(), xlims=(None, None), figsize=(), path=None):
    '''Plot raster plots for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of flywire IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2flyid`
        must be supplied
    name2flyid : dict, optional
        Mapping betwen custon neuron names and flywire IDs, by default dict()
    xlims : tuple, optional
        xlims for plot, by default (None, None)
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 3*n_neu, 2*n_exp
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axmat = plt.subplots(ncols=n_neu, nrows=n_exp, squeeze=False, figsize=(dx, dy))

    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):

        trl_max = df_exp.max()['trial'] # for axis limits

        gr_neu = df_exp.groupby('flywire_id')
        for j, n in enumerate(neu):
            ax = axmat[i,j]

            idx = name2flyid.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
            
                for trl, df_trl in df_neu.groupby('trial'):
                    t = df_trl.loc[:, 't']
                    ax.eventplot(t, lineoffset=trl, linewidths=.5)

            except KeyError:
                pass
            
            # formatting
            if j == 0:
                ax.set_ylabel(e)
            else:
                ax.set_yticklabels('')
                
            if i == 0:
                ax.set_title(n)

            ax.grid(None)
            ax.set_xlim(xlims)
            ax.set_ylim(-0.5, trl_max + 0.5)
           

    for ax in axmat[-1]:
        ax.set_xlabel('time [s]')
    fig.tight_layout()

    if path:
        fig.savefig(path)


def plot_rate(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, name2flyid=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of flywire IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2flyid`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    name2flyid : dict, optional
        Mapping betwen custon neuron names and flywire IDs, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp = len(exp)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, 4
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))
    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('flywire_id')

        df_bin = pd.DataFrame()

        for n in neu:
            idx = name2flyid.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
                gr_trl = df_neu.groupby('trial')

                for trl in range(n_trl):

                    try: 
                        df_trl = gr_trl.get_group(trl)
                        t = df_trl.loc[:, 't']
                    except KeyError:
                        t = []

                    y, _ = np.histogram(t, bins=bins)
                    y = gaussian_filter1d(y.astype(float), sigma=sigma, axis=0)
                    y *= 1e3
                    df = pd.DataFrame(data={
                        't' : bins[:-1],
                        'r': y,
                        'trl': trl,
                        'neu': n,
                    })
                    df_bin = pd.concat([df_bin, df], ignore_index=True)

            except KeyError:
                df = pd.DataFrame(data={
                    't' : bins[:-1],
                    'r': 0,
                    'neu': n,
                })
                df_bin = pd.concat([df_bin, df], ignore_index=True)

        if do_zscore:
            for n, df in df_bin.groupby('neu'):
                idx = df.index
                df_bin.loc[idx, 'r'] = zscore(df_bin.loc[idx, 'r'], ddof=1)

        sns.lineplot(data=df_bin, ax=ax, x='t', y='r', errorbar='sd', hue='neu')

        # formatting
        ax.legend()
        ax.set_title(e)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rate [Hz]')

    fig.tight_layout()
    if path:
        fig.savefig(path)

def plot_rate_heatmap(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, exclude_stim=False, color_range=(None, None), name2flyid=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons in a heatmap

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of flywire IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2flyid`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    exclude_stim : bool, optional
        If True, replace stimulated neurons with nan, by default False
    color_range : tuple, optional
        Values for min and max for the color map, by default (None, None)
    name2flyid : dict, optional
        Mapping betwen custon neuron names and flywire IDs, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)
    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, .25 * n_neu + 1
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))
    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    if do_zscore:
        cmap = 'coolwarm'
        norm = CenteredNorm()
    else: 
        cmap = 'viridis'
        norm = None

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('flywire_id')

        # stuff for excluding stim
        # TODO: make more pretty
        id_b = df_spkt.attrs['stim_ids'][e]
        b2f = pd.Series(df_exp.loc[:, 'flywire_id'].values, index=df_exp.loc[:, 'brian_id']).to_dict()
        id_f = [ b2f[i] for i in id_b ]

        Z = []
        for n in neu:
            idx = name2flyid.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
                t = df_neu.loc[:, 't']
            except KeyError:
                t = []

            z, _ = np.histogram(t, bins=bins)
            z = gaussian_filter1d(z.astype(float), sigma=sigma, axis=0)
            z = z / n_trl * 1e3
            if do_zscore:
                z = zscore(z, ddof=1)

            if exclude_stim and idx in id_f:
                z[:] = np.nan

            Z.append(z)

        Z = np.vstack(Z)
        x = bins[:-1]
        y = np.arange(n_neu)
        im = ax.pcolormesh(x, y, Z, cmap=cmap,  norm=norm, vmin=color_range[0], vmax=color_range[1])
        fig.colorbar(im, ax=ax, location='right', orientation='vertical')

        # TODO colorbar label and xlabel

        ax.set_yticks(y)
        ax.set_yticklabels(neu)

        # formatting
        ax.set_title(e)
        ax.set_xlabel('time [s]')

    fig.tight_layout()

    if path:
        fig.savefig(path)

########
# graphs

def get_full_graph(p_comp, p_con):
    '''Convert completeness and connectivity dataframes to networkx graph

    Parameters
    ----------
    p_comp : str
        Path to completeness.csv
    p_con : str
        Path to connectivity.parquet

    Returns
    -------
    G : networkx.DiGraph
        Graph containing all data in completeness and connectivity dataframes
    '''

    df_comp = pd.read_csv(p_comp)
    df_con = pd.read_parquet(p_con)

    G = nx.DiGraph()
    G.add_nodes_from(df_comp.loc[:, 'Unnamed: 0'])
    G.add_weighted_edges_from(df_con.loc[:, ['Presynaptic_ID', 'Postsynaptic_ID', 'Excitatory x Connectivity']].values)

    return G


def write_graph(G, p_prq, name2flyid=dict(), neurons=[]):
    '''Select active neurons from G and write subgraph to disk
    File can be visualized with gephi software.

    Parameters
    ----------
    G : networkx.DiGraph
        Complete connectome graph
    p_prq : str
        Path to parquet file of a single experiment
    name2flyid : dict, optional
        Mapping between custom names and flywire IDs, by default dict()
        If supplied, flywire IDs will be converted in the outuput graph
    neurons : list
        List of custom names/flywire IDs to include in the graph.
        Resulting graph will include these neurons instead of all neurons
        that are active in a `p_prq` experiment.
        If custom names are supplied, name2flyid need to be passed as well.

    '''


    p_prq = Path(p_prq)
    p_pkl = p_prq.with_suffix('.pickle')
    p_gexf = p_prq.with_suffix('.gexf')
    p_gexf_cust = p_prq.parent / (p_prq.with_suffix('').name + '_cust.gexf')

    # determine duration
    with open(p_pkl, 'rb') as f:
        df = pickle.load(f)['df_inst']
    dur = df.loc[df.loc[:, 'mode'] == 'end'].loc[:, 't'].item()

    # load spike times/rate
    df_spkt = load_exps([p_prq])
    df_rate, _ = get_rate(df_spkt, duration=dur)
    ds = df_rate.iloc[:, 0]

    if neurons:
        # select subgraph based on custom list
        ids = [ name2flyid[i] for i in neurons ]
        G_sub = G.subgraph(ids).copy()
        nx.set_node_attributes(G_sub, 0.0, 'rate')
        p_gexf = p_gexf_cust

    else:
        # select subgraph with active neurons
        G_sub = G.subgraph(ds.index).copy()
        ids = ds.index

    # assign rate to
    for i in ids:
        if i in ds.index:
            r = ds.loc[i]
            G_sub.nodes[i]['rate'] = r

    # convert flywire ids to names
    flyid2name = { int(j): i for i, j in name2flyid.items() }
    G_sub = nx.relabel_nodes(G_sub, lambda x: flyid2name.get(x, int(str(x)[9:])))

    # add attributes to nodes
    nx.set_node_attributes(G_sub, False, name='named')
    nx.set_node_attributes(G_sub, '?', name='hemisphere')
    for n in name2flyid.keys():
        if n in G_sub.nodes:
            G_sub.nodes[n]['named'] = True
            if n.endswith('_r'):
                G_sub.nodes[n]['hemisphere'] = 'R'
            elif n.endswith('_l'):
                G_sub.nodes[n]['hemisphere'] = 'L'

    # make edge weights absolute and add `sign` attribute
    for _, _, d in G_sub.edges(data=True):
        weight = abs(d['weight'])
        sign = -1 if d['weight'] < 0 else 1
        d['weight'] = int(weight)
        d['sign'] = sign

    print('INFO: writing graph file {}'.format(p_gexf))
    nx.write_gexf(G_sub, p_gexf)

