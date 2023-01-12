import pandas as pd
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