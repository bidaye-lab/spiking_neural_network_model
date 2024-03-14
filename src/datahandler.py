from pathlib import Path
import pandas as pd

def load_flywire_ids(path_comp):
    '''Load database IDs from completeness file

    Parameters
    ----------
    path_comp : path-like
        Path to completeness file (CSV format)

    Returns
    -------
    ds : pd.Series
        Series of all database IDs

    Raises
    ------
    ValueError
        If database IDs are not unique
    '''
    
    ds = pd.read_csv(Path(path_comp)).loc[:, 'Unnamed: 0']
    if not ds.is_unique:
        raise ValueError(f'Database IDs in {path_comp} are not unique')

    return ds

def load_flywire_connectivity(path_con, ds_ids):
    '''Load connectivity data from flywire connectivity file

    Parameters
    ----------
    path_con : path-like
        Path to connectivity file (parquet format)
    ds_ids : pd.Series
        Series of all database IDs

    Returns
    -------
    df : pd.DataFrame
        Dataframe with columns "pre", "post", "w"
    '''

    df = pd.read_parquet(
        Path(path_con),
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"],
    )
    df.columns = ["pre", "post", "w"]

    id2i = {v: k for k, v in ds_ids.to_dict().items()}
    df.loc[:, 'pre'] = df.loc[:, 'pre'].map(id2i)
    df.loc[:, 'post'] = df.loc[:, 'post'].map(id2i)

    return df


def print_connectivity(ds_ids, df_con, neu1, neu2):
    '''Query connectivity between two neurons

    Parameters
    ----------
    ds_ids : pd.Series
        Series of all database IDs
    df_con : pd.DataFrame
        Dataframe with connectivity information in canonical IDs
    neu1 : int
        Database ID of first neuron
    neu2 : int
        Database ID of second neuron
    '''

    id2i = {v: k for k, v in ds_ids.to_dict().items()}

    ds = df_con.loc[ (df_con.loc[:, 'pre'] == id2i[neu1]) & (df_con.loc[:, 'post'] == id2i[neu2])]
    ec = ds.loc[:, 'w']
    c = ec.item() if len(ec) else 0
    print(f'pre: {neu1} -> post: {neu2} = {c}')

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
    '''Load XLS file containing neuron name to database ID mapping

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
            Path(path_names),
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
            Path(path_names),
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
    '''Check for double definition of neuron names and database IDs.
    Prints warning if duplicates found

    Parameters
    ----------
    df_pair : pd.DataFrame
        Names and database IDs for neuron pairs
    df_single : pd.DataFrame
        Names and database IDs for single neurons
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

    # check database IDs
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
    '''Assign custom name to database ID
    Prints warning if database ID could not be interpreted as integer and skips those

    Parameters
    ----------
    name2id : dict
        dictionary to which to assign to
    name : str
        custom neuron name
    id : str
        str to be interpreted as database ID
    '''
    try:
        name2id[name] = int(id)
    except ValueError:
        print('WARNING: Did not assign any database ID to name {:>15}: {} is not a valid integer'.format(name, id))


def check_ids(name2id, ds_ids):
    '''Check if database IDs of custom names appear in completeness file

    Parameters
    ----------
    name2id : dict
        Mapping between custom names and database IDs
    ds_ids : pd.Series
        Series of all database IDs
    '''

    # check if IDs are correct: if everything is correct, nothing is printed
    warn = False
    for k, v in name2id.copy().items():
        if not v in ds_ids.values:
            print('ERROR: ID {} for neuron {:>15} not found. Please provide correct database ID. Removing neuron'.format(str(v), k))
            name2id.pop(k)
            warn = True
    if not warn:
        print('INFO: all IDs in `name2id` appear to match with database IDs')
    print()


def create_name_dict(path_name, ds_ids, sheets_pair, sheets_single):
    '''Generate dictionary mapping custom neuron names to database IDs

    Parameters
    ----------
    path_name : str
        Path to XLS files containing neuron names and database IDs
    ds_ids : pd.Series
        Series of all database IDs
    sheets_pair : list
        List of sheet names containing neuron pairs
    sheets_single : list
        List of sheet names containing individual neurons

    Returns
    -------
    name2id : dict
        Mapping between custom neuron names to database IDs
    '''

    # load XLS file into dataframes
    df_pair, df_single = load_xls(path_name, sheets_pair, sheets_single)
    
    # check for duplicates
    check_unique(df_pair, df_single)

    # create dictionary mapping custom names to database IDs
    name2id = dict()

    for i in df_pair.index: # left/right pairs
        n, id_l, id_r = df_pair.loc[i, ['Name', 'ID_left', 'ID_right']]
        n_l, n_r = '{}_l'.format(n), '{}_r'.format(n)
        assign(name2id, n_l, id_l)
        assign(name2id, n_r, id_r)

    for i in df_single.index: # single neurons
        n, id = df_single.loc[i, ['Name', 'ID']]
        assign(name2id, n, id)    

    print( 'Declared {} names for neurons'.format(len(name2id)))
    print()

    # check if all database IDs appar in neurotransmitter data
    check_ids(name2id, ds_ids)

    return name2id

def useful_mappings(name2id, ds_ids):
    '''Generate other useful mappings between custom names, database IDs
    and canonical IDs (starting with 0, will be equivalent to brian neuron IDs)

    Parameters
    ----------
    name2id : dict
        Mapping between custom neuron names and database IDs
    ds_ids : pd.Series
        Series of all database IDs

    Returns
    -------
    id2name : dict
        Inverted name2id dictionary
    id2i : dict
        Mapping between database IDs and canonical IDs
    i2id : dict
        Inverted id2i dictionary
    name2i : dict
        Mapping between custom neuron names and canonical IDs
    i2name : dict
        Inverted name2s dictionary
    name_or_id2i : dict
        Mapping of custom neuron names and database IDs to canonical IDs
    '''

    id2name = { j: i for i, j in name2id.items() } # database ID: custom name

    i2id = ds_ids.to_dict() # canonical (i.e. brian) ID: database ID
    id2i = {j: i for i, j in i2id.items()} # brian ID: database ID

    name2i = {i: id2i[j] for i, j in name2id.items() } # custom name: brian ID
    i2name = {j: i for i, j in name2i.items() } # brian ID: custom name

    name_or_id2i = name2i | id2i # union of dicts

    return id2name, id2i, i2id, name2i, i2name, name_or_id2i
