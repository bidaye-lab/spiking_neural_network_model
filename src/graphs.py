from pathlib import Path
import pickle

import pandas as pd
import networkx as nx

from src.analysis import load_exps, get_rate

def get_full_graph(ds_ids, df_con):
    '''Create directd graph from connectome

    Nodes are values in `ds_ids`, edges are `w` column in `df_con`

    Parameters
    ----------
    ds_ids : pd.Series
        Series of all database IDs
    df_con : pd.DataFrame
        Dataframe with connectivity data in canonical IDs

    Returns
    -------
    G : networkx.DiGraph
        Connectome graph
    '''



    # graph with canonical IDs as nodes
    G = nx.DiGraph()
    G.add_nodes_from(ds_ids.index)

    # connect nodes
    G.add_weighted_edges_from(df_con.loc[:, ['pre', 'post', 'w']].values)

    # rename nodes from canonical IDs to database
    nx.relabel_nodes(G, ds_ids.to_dict(), copy=False)

    return G


def write_graph(G, p_prq, name2id=dict(), neurons=[], suffix='gexf', dur=0):
    '''Select active neurons from G and write subgraph to disk
    File can be visualized with gephi software (gexf format) or cytoscape (gml format).

    Parameters
    ----------
    G : networkx.DiGraph
        Complete connectome graph
    p_prq : str
        Path to parquet file of a single experiment
    name2id : dict, optional
        Mapping between custom names and database IDs, by default dict()
        If supplied, database IDs will be converted in the outuput graph
    neurons : list
        List of custom names/database IDs to include in the graph.
        Resulting graph will include these neurons instead of all neurons
        that are active in a `p_prq` experiment.
        If custom names are supplied, name2id need to be passed as well.
    suffix : str, optional
        Determines the output file format, by default 'gexf'
        Available formats: 'gexf', 'gml'
    dur : float, optional
        Duration of simulation, required to calculate rate
        If 0, get duration from pickle file, which needs to be present
    '''


    p_prq = Path(p_prq)
    p_pkl = p_prq.with_suffix('.pickle')
    p_graph = p_prq.with_suffix(f'.{suffix}')
    p_graph_cust = p_prq.parent / (p_prq.with_suffix('').name + f'_cust.{suffix}')

    if not dur:
        # determine duration
        with open(p_pkl, 'rb') as f:
            df = pickle.load(f)['df_inst']
        dur = df.loc[df.loc[:, 'mode'] == 'end'].loc[:, 't'].item()

    # load spike times/rate
    df_spkt = load_exps([p_prq], load_pickle=False)
    df_rate, _ = get_rate(df_spkt, duration=dur)
    ds = df_rate.iloc[:, 0]

    if neurons:
        # select subgraph based on custom list
        ids = [ name2id.get(i, i) for i in neurons ]
        G_sub = G.subgraph(ids).copy()
        nx.set_node_attributes(G_sub, 0.0, 'rate')
        p_graph = p_graph_cust

    else:
        # select subgraph with active neurons
        G_sub = G.subgraph(ds.index).copy()
        ids = ds.index

    # assign rate to
    for i in ids:
        if i in ds.index:
            r = ds.loc[i]
            G_sub.nodes[i]['rate'] = r

    # convert database IDs to names
    id2name = { int(j): i for i, j in name2id.items() }
    G_sub = nx.relabel_nodes(G_sub, lambda x: id2name.get(x, int(str(x)[9:])))

    # add attributes to nodes
    nx.set_node_attributes(G_sub, False, name='named')
    nx.set_node_attributes(G_sub, '?', name='side')
    for n in name2id.keys():
        if n in G_sub.nodes:
            G_sub.nodes[n]['named'] = True
            if n.endswith('_r'):
                G_sub.nodes[n]['side'] = 'R'
            elif n.endswith('_l'):
                G_sub.nodes[n]['side'] = 'L'

    # make edge weights absolute and add `sign` attribute
    for _, _, d in G_sub.edges(data=True):
        weight = abs(d['weight'])
        sign = -1 if d['weight'] < 0 else 1
        d['weight'] = int(weight)
        d['sign'] = sign

    # save to disk
    print('INFO: writing graph file {}'.format(p_graph))
    if suffix == 'gexf':
        nx.write_gexf(G_sub, p_graph)
    elif suffix == 'gml':
        nx.write_gml(G_sub, p_graph)
    else:
        raise NotImplementedError(f'Saving graph file format {suffix} is not implemented')

