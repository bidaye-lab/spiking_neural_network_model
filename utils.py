import pandas as pd
from pathlib import Path

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
        Path(path_names),
        sheet_name=sheets_pair,
        dtype={'ID_left': str, 'ID_right': str}
        )
    df_pair = pd.concat(dfs_pair, ignore_index=True).dropna(how='all')

    # sheets with single neurons (name | ID)
    dfs_single = pd.read_excel( 
        Path(path_names),
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
    df_comp = pd.read_csv(Path(path_comp), index_col=0) # df with all IDs
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