import pandas as pd
import numpy as np
from utils import useful_mappings
from textwrap import dedent

# brian 2
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz, Mohm, uF

# file handling
import pickle
from pathlib import Path

# parallelization
from joblib import Parallel, delayed, parallel_backend
from time import time

default_params = {

    # network constants
    # Kakaria and de Bivort 2017 https://doi.org/10.3389/fnbeh.2017.00008
    # refereneces therein, e.g. Hodgkin and Huxley 1952
    'v_0'       : -52 * mV,               # resting potential
    'v_rst'     : -52 * mV,               # reset potential after spike
    'v_th'      : -45 * mV,               # threshold for spiking
    't_mbr'     : .002 * uF * 10. * Mohm, # membrane time scale (capacitance * resistance)

    # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
    'tau'       : 5 * ms,                 # time constant (this is the excitatory one, the inhibitory is 10 ms)

    # Lazar et at https://doi.org/10.7554/eLife.62362
    # they cite Kakaria and de Bivort 2017, but those have used 2 ms
    't_rfc'     : 2.2 * ms,               # refractory period

    # Paul et al 2015 doi: 10.3389/fncel.2015.00029
    't_dly'     : 1.8*ms,                 # delay for changes in post-synaptic neuron

    # empirical 
    'w_syn'     : .275 * mV,              # weight per synapse (note: modulated by exponential decay)
    'r_poi'     : 150*Hz,                 # default rate of the Poisson input
    'r_poi2'    :  10*Hz,                 # default rate of another Poisson input (useful for different frequencies)
    'f_poi'     : 250,                    # scaling factor for Poisson synapse

    # equations for neurons
    'eqs'       : dedent('''
                    dv/dt = (x - (v - v_0)) / t_mbr : volt (unless refractory)
                    dx/dt = -x / tau                : volt (unless refractory) 
                    rfc                             : second
                    '''),
    # condition for spike
    'eq_th'     : 'v > v_th', 
    # rules for spike        
    'eq_rst'    : 'v = v_rst; w = 0; x = 0 * mV', 
}


#######################
# brian2 model setup
def stimulate(neu, stim, params, r_poi_key='r_poi'):
    '''Create PoissonInput for neurons.

    For each neuron in 'names' a PoissonInput is generated and 
    the refractory period of that neuron is set to 0 in NeuronGroup.

    Parameters
    ----------
    neu : NeuronGroup
        Defined brian2.NeuronGroup object
    stim : list
        Indices of neurons for which to create Poisson input
    rate : Unit, optional
        Frequency for the Poisson spikes, by default 'r_poi'
    params : dict
        Constants and equations that are used to construct the brian2 network model
    r_poi_key : str
        Name for the Poisson frequency in `params`, useful when using multiple frequencies

    Returns
    -------
    poi_inp : list
        PoissonInput objects for each neuron in 'stim'
    '''

    poi_inp = []
    for i in stim:
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=params[r_poi_key], 
            weight=params['w_syn']*params['f_poi']
            )
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        poi_inp.append(p)
        
    return poi_inp

def silence(slnc, syn):
    '''Silence neuron by setting weights of all synapses from it to 0

    Parameters
    ----------
    slnc : list
        List of neuron indices to silence
    syn : brian2.Synapses
        Defined synapses object
    '''

    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0*mV
        syn.w[' {} == j'.format(i)] = 0*mV
    

def create_model(path_comp, path_con, params):
    '''Create default network model.

    Convert the "completeness materialization" and "connectivity" dataframes
    into a brian2 neural network model. Network constants and equations
    are defined at the beginning of this file.

    Parameters
    ----------
    path_comp : str
        path to "completeness materialization" dataframe
    path_con : str
        path to "connectivity" dataframe
    params : dict
        Constants and equations that are used to construct the brian2 network model


    Returns
    -------
    neu : NeuronGroup
        brian2.NeuronGroup object with neurons as in 'path_comp'
    syn : Synapses
        brian2.Synapses object with connections as in 'path_con'
    spk_mon : SpikeMonitor
        brian2.SpikeMonitor object, which records time of spike events
    '''

    # load neuron connectivity dataframes
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    neu = NeuronGroup( # create neurons
        N=len(df_comp),
        model=params['eqs'],
        method='linear',
        threshold=params['eq_th'],
        reset=params['eq_rst'],
        refractory='rfc',
        name='default_neurons',
        namespace=params,
    )
    neu.v = params['v_0'] # initialize values
    neu.x = 0
    neu.rfc = params['t_rfc']

    # create synapses
    syn = Synapses(neu, neu, 'w : volt', on_pre='x += w', delay=params['t_dly'], name='default_synapses')

    # connect synapses
    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    # define connection weight
    syn.w = df_con.loc[:,'Excitatory x Connectivity'].values * params['w_syn']

    # object to record spikes
    spk_mon = SpikeMonitor(neu) 

    return neu, syn, spk_mon

#####################
# running simulations
def get_spk_trn(spk_mon):
    '''Extracts spike times from 'spk_mon'

    The spike times recorded in the SpikeMonitor object during 
    simulation are converted to a list of times for each neurons.
    Returns dict wich "brian ID": "list of spike times".

    Parameters
    ----------
    spk_mon : SpikeMonitor
        Contains recorded spike times

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''

    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    
    return spk_trn

def get_res_df(res, exp_name, i2flyid):
    '''Take spike time dict and colltect spikes in pandas dataframe

    Parameters
    ----------
    res : list
        List with spike time dicts for each trial
    exp_name : str
        Name of the experiment
    i2flyid : dict
        Mapping between Brian IDs and flywire IDs

    Returns
    -------
    df : pandas.DataFrame
        Dataframe where each row is one spike
    '''
    
    ids, ts, nrun = [], [], []

    for n, i in enumerate(res):
        for j, k  in i.items():
            ids.extend([j for _ in k])
            nrun.extend([n for _ in k])
            ts.extend([float(l) for l in k])

    d = {
        't': ts,
        'trial': nrun,
        'brian_id': ids,
    }
    df = pd.DataFrame(d)
    df.loc[:, 'flywire_id'] = df.loc[:, 'brian_id'].map(i2flyid)
    df.loc[:, 'exp_name'] = exp_name

    return df



def get_df_inst(l_inst, name2i):
    '''Generate dataframe with instructions for the simulation
    See tutorial for usage

    Parameters
    ----------
    l_inst : list
        Each element is a tuple with 
        i) the time [s] as float, 
        ii) the  `mode` as str, and
        iii) the neurons as list
    name2i : dict
        Mapping from neuron names to canonical IDs

    Returns
    -------
    df : pd.DataFrame
        Each row represents an instruction for neuron stimulation, silencing etc
    '''
    

    df = pd.DataFrame(columns=['t', 'mode', 'name'], data=l_inst)
 
    df.loc[:, 'id'] = df.loc[:, 'name'].apply(lambda l: [name2i[i] for i in l])

    df = df.sort_values(by='t')

    dt = np.roll(df.loc[:, 't'].diff().values, -1)
    df.loc[:, 'dt'] = dt

    return df


def run_trial(df_inst, path_comp, path_con, params):
    '''Run single trial of simulation

    Parameters
    ----------
    df_inst : pd.DataFrame
        Instructions for the simulation
    path_comp: Path 
        path to "completeness materialization" dataframe
    path_con: Path
        path to "connectivity" dataframe
    params : dict
        Constants and equations that are used to construct the brian2 network model

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times

    '''


    # get default network
    neu, syn, spk_mon = create_model(path_comp, path_con, params)
    net = Network(neu, syn, spk_mon)

    for i in df_inst.index:
        mode, ids, dt = df_inst.loc[i, ['mode', 'id', 'dt']]
        if mode == 'stim':
            # add Poisson inputs to network (default frequency)
            poi_inp = stimulate(neu, ids, params)
            net.add(*poi_inp)
        elif mode == 'stim2':
            # add Poisson inputs to network (alternative frequency)
            poi_inp = stimulate(neu, ids, params, r_poi_key='r_poi2')
            net.add(*poi_inp)
        elif mode == 'slnc':
            # silence neurons
            syn = silence(ids, syn)
        elif mode == 'end':
            break
        else:
            raise NotImplementedError(f'Cannot interpret instruction {mode}')

        # run simulation
        net.run(duration=dt * 1000 * ms)

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_exp(exp_name, exp_inst, path_res, path_comp, path_con, params=default_params, name2flyid=dict(), n_trl=30, force_overwrite=False, n_proc=-1,):
    '''
    Run default network experiment with PoissonInputs as external input.
    Neurons chosen as Poisson sources spike with a default rate of 150 Hz
    and have a refractory period of 0 ms.

    Parameters
    ----------
        exp_name: str
            name of the experiment
        exp_inst: list
            Instructions what to do with which neurons when (see tutorial)
        path_res: str
            path to the output folder where spike data is stored
        path_comp: str 
            path to "completeness materialization" dataframe
        path_con: str
            path to "connectivity" dataframe
        params : dict
            Constants and equations that are used to construct the brian2 network model
        name2flyid : dict
            Mapping between custom neuron names and flywire IDs
        n_trl : int, optional
            Number of trials, by default 30
        force_overwrite : bool, optional
            If True, overwrite output files, by defaul False
        n_proc : int
            number of cores to be used for parallel runs
            default: -1 (use all available cores)
            n_proc=1 is equivalent serial code
    '''
    # convert to Path objects
    path_res, path_comp, path_con = [ Path(i) for i in [path_res, path_comp, path_con] ]
    path_res.mkdir(parents=True, exist_ok=True)

    # define output files
    out_pkl = path_res / '{}.pickle'.format(exp_name)
    out_prq = out_pkl.with_suffix('.parquet')

    if out_prq.is_file():
        if force_overwrite:
            print('INFO: {} will be overwritten'.format(out_prq))
        else:
            print('INFO: {} already exists, skipping calculation. '.format(out_prq) + \
                  'Choose different `exp_name` or set `force_overwrite=True`')
            return

    # load name/id mappings
    _, _, i2flyid, _, _, name_flyid2i = useful_mappings(name2flyid, path_comp)

    # generate instructions
    df_inst = get_df_inst(exp_inst, name2i=name_flyid2i)

    # print info
    print('>>> Experiment:     {}'.format(exp_name))
    print('    Output files:   {}'.format(out_prq))
    print('                    {}'.format(out_pkl))
    print('    Instructions:')
    for i in df_inst.index:
        t, n, m = df_inst.loc[i, ['t', 'name', 'mode']]
        print('{:>12} | {:>5} | {}'.format(t, m, ' '.join([str(j) for j in n])))

    # start time for simulation
    start = time() 

    # start parallel calculation
    with parallel_backend('loky', n_jobs=n_proc):
        res = Parallel()(
            delayed(
                run_trial)(df_inst, path_comp, path_con, params) for _ in range(n_trl))

    # print simulation time
    walltime = time() - start 
    print('    Elapsed time:   {} s'.format(int(walltime)))

    # dataframe with spike times
    df_res = get_res_df(res, exp_name, i2flyid)

    # store spike data 
    df_res.to_parquet(out_prq, compression='brotli')

    # store experiment metadata
    data = {
        'exp_name':     exp_name,
        'name2flyid':   name2flyid,
        'name_flyid2i': name_flyid2i,
        'df_inst':      df_inst,
        'n_trl':        n_trl,
        'path_res':     path_res,
        'path_comp':    path_comp,
        'path_con':     path_con,
        'n_proc':       n_proc,
        'walltime':     walltime,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)