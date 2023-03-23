import pandas as pd
import numpy as np

import pickle

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
    dfs_pair = pd.read_excel( 
        path_names,
        sheet_name=sheets_pair,
        dtype={'ID_left': str, 'ID_right': str}
        )
    df_pair = pd.concat(dfs_pair, ignore_index=True).dropna(how='all')

    # sheets with single neurons (name | ID)
    dfs_single = pd.read_excel( 
        path_names,
        sheet_name=sheets_single,
        dtype={'ID': str}
        )
    df_single = pd.concat(dfs_single, ignore_index=True).dropna(how='all')

    # print info
    print('INFO: Loaded sheets ...')
    for i in [*dfs_pair.keys(), *dfs_single.keys()]:
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
        print('WARNING: Could not assign ID {} to name {}'.format(id, name))


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
            print('WARNING: ID {} for neuron {} not found'.format(str(v), k))
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
def load_exps(l_pkl):
    '''Load simulation results from disk

    Parameters
    ----------
    l_pkl : list
        List of pickle files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pkl:
        # load metadata from pickle
        with open(p, 'rb') as f:
            pkl = pickle.load(f)
            df = pkl['spk_ts']
            df.loc[:, 't'] = df.loc[:, 't'].astype(float)
            dfs.append(df)

    df = pd.concat(dfs)

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