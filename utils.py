import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# load from disk
def load_dfs(path_comp, path_con):
    
    # load neuron data
    df_comp = pd.read_csv(path_comp, index_col = 0) # neuron ids and excitation type
    df_con = pd.read_csv(path_con, index_col = 0) # connectivity

    return df_comp, df_con

def load_dicts(path_name):

    # load name mappings
    with open(path_name, 'rb') as f:
        flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i = pickle.load(f)

    return flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i 

def load_names(path_name):

    path_name = Path(path_name)
    flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i = load_dicts(path_name)
    names = [ *name2flyid.keys() ]
    return names

def load_exps(path_res, exp_type):
    '''Load simulation results from disk

    Parameters
    ----------
    path_res : str
        Path to results folder
    exp_type : str
        type of the experiment (coac | dly)

    Returns
    -------
    exps : dict
        data for all experiments of 'exp_type' in 'path_res'
    '''
    # wildcard matching all outputs of 'exp_type'
    pkl_glob = Path(path_res).glob('{}*.pickle'.format(exp_type))

    # initilize dicts
    exps = dict()

    # cycle through all experiments
    for p in pkl_glob:

        # load metadata from pickle
        with open(p, 'rb') as f:
            pkl = pickle.load(f) 
        n = pkl['exp_name']
        exps[n] = pkl

        # load spike times from feather
        f = p.with_suffix('.feather')
        exps[n]['data'] = pd.read_feather(f)

    return exps

# dataframe handling
def collect_spikes(exps):

    # collect spike times of all experiments in one dataframe
    df = pd.DataFrame()

    for e in exps: # cycle through experiments
        df_exp = exps[e]['data'] # df for individual experiment
        
        #  sum up spike events
        df_exp = df_exp.dropna(how='all') # select only rows containing spikes
        ds = df_exp.apply(lambda x: [i for i in x], axis=1) # concatenate spk times, ignore nan
        ds.name = e # dataseries name will convert to dataframe column
        df = pd.concat([df, ds], axis=1)
    
    return df

def rename_neurons(df, i2flyid, flyid2name):

    # rename brian ids to flywire/custom names
    df = df.rename(index=i2flyid).rename(index=flyid2name) # (1) flywire ids and (2) custom names
    df.index = df.index.astype(str) # represent flywire IDs as str, not int
    
    # sort indixes: named first
    idx_n = [ i for i in flyid2name.values() if i in df.index] # named neurons
    idx_i = [ i for i in df.index if i not in flyid2name.values() ] # flywire ids
    idx_i.sort() # sort flywire ids
    df = df.loc[idx_n + idx_i, :] # first named neurons, then flywire ids

    return df
    
def calculate_rates(df, n_runs=30, t_sim=1):
    
    def get_count(x):
        n = 0
        for i in x:
            if i is not None:
                n += len(i)
        n 
        return n

    df_count = df.applymap(get_count, na_action='ignore') # count spike events
    df_rate = df_count / ( n_runs * t_sim )

    df_rate = df_rate.fillna(0) # replace nan with 0, necessary for differences later

    return df_rate

def save_all_xls(df, file):
    # save as xlsx
    with pd.ExcelWriter(file, mode='w', engine='xlsxwriter') as w:
        df.to_excel(w, sheet_name='all_experiments')

        # formatting in the xlsx file
        wb = w.book
        # set floating point display precision here (excel format)
        fmt = wb.add_format({'num_format': '#,##0.0'}) 
        for _, ws in w.sheets.items():
            ws.set_column(1, 1, 10, fmt)
            ws.freeze_panes(1, 1)

# plotting
def plot_raster(df, exps, neurons, xlims=(0, 2), path=None):
    n_exp, n_neu = len(exps), len(neurons)
    fig, axmat = plt.subplots(ncols=n_neu, nrows=n_exp, squeeze=False, figsize=(3*n_neu, 2*n_exp))
    for i, e in enumerate(exps):
        for j, n in enumerate(neurons):
            ax = axmat[i,j]
            if j == 0:
                ax.set_ylabel(e)
            if i == 0:
                ax.set_title(n)
            trials = df.loc[n, e]
            if type(trials) == list:
                for k, t in enumerate(trials):
                    if str(type(t)) != str(type(None)):
                        ax.eventplot(t, lineoffset=k, linewidths=.5)
                ax.set_ylim(-0.5, len(trials)-0.5)
            ax.grid(None)
            ax.set_xlim(xlims)
            ax.set_yticklabels('')
    for ax in axmat[-1]:
        ax.set_xlabel('time [s]')

    fig.tight_layout()

    if path:
        fig.savefig(path)