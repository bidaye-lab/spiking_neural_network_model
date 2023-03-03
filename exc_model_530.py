import pandas as pd
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
    # trials
    't_run'     : 1000 * ms,              # duration of trial
    'n_run'     : 30,                     # number of runs

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
def poi(neu, exc, params):
    '''Create PoissonInput for neurons.

    For each neuron in 'names' a PoissonInput is generated and 
    the refractory period of that neuron is set to 0 in NeuronGroup.

    Parameters
    ----------
    neu : NeuronGroup
        Defined brian2.NeuronGroup object
    exc : list
        Indices of neurons for which to create Poisson input
    rate : Unit, optional
        Frequency for the Poisson spikes, by default 'r_poi'
    params : dict
        Constants and equations that are used to construct the brian2 network model

    Returns
    -------
    pois : list
        PoissonInput objects for each neuron in 'exc'
    neu : NeuronGroup
        NeuronGroup with adjusted refractory periods
    '''

    pois = []
    for i in exc:
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=params['r_poi'], 
            weight=params['w_syn']*params['f_poi']
            )
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        pois.append(p)
        
    return pois, neu

def silence(slnc, syn):
    '''Silence neuron by setting weights of all synapses from it to 0

    Parameters
    ----------
    slnc : list
        List of neuron indices to silence
    syn : brian2.Synapses
        Defined synapses object

    Returns
    -------
    syn : brian2.Synapses
        Synapses with modified weights
    '''

    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0*mV
    
    return syn

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

def construct_dataframe(res, exp_name, exc, i2flyid):
    '''Take spike time dict and colltect spikes in pandas dataframe

    Parameters
    ----------
    res : list
        List with spike time dicts for each trial
    exp_name : str
        Name of the experiment
    exc : list
        List with indices for excited neurons
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
    df.loc[:, 'exc'] = df.loc[:, 'brian_id'].map(lambda x: True if x in exc else False)
    df.loc[:, 'exp_name'] = exp_name

    return df

def run_trial_coac(exc, path_comp, path_con, params):
    '''Run single trial of coactivation experiment

    During the coactivation experiment, the neurons in 'exc' are
    Poisson inputs. The simulation runs for 't_run'.
    

    Parameters
    ----------
    exc: list
        contains indices of neurons for PoissonInput
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
    # define Poisson input for excitation
    poi_inp, neu = poi(neu, exc, params)
    # collect in Network object
    net = Network(neu, syn, spk_mon, *poi_inp)

    # run simulation
    net.run(duration=params['t_run'])

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_trial_dly(exc_tup, path_comp, path_con, params):
    '''Run single trial of delayed activation experiment

    During the delayed activation experiment, groups of neurons
    are made Poisson inputs consecutively during the simulation.
    Here, 'exc' is a tuple of lists, where each list contains neurons
    to be added after 't_run'.
    E.g. if exc=([1, 2], [3]) and t_run=1*s, the simulations runs
    for 1 s with 1 and 2 as Poisson inputs and for another 1 s
    with 1, 2, and 3 as Poisson inputs.

    
    Parameters
    ----------
    exc_tup: tuple
        contains tuple of lists of indices of neurons for PoissonInput
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

    for exc in exc_tup:
        # add Poisson inputs to network
        poi_inp, neu = poi(neu, exc, params)
        net.add(*poi_inp)

        # run simulation
        net.run(duration=params['t_run'])

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_trial_slnc(exc, slnc, path_comp, path_con, params):
    '''Run single trial of coactivation experiment

    During the coactivation experiment, the neurons in 'exc' are
    Poisson inputs. The simulation runs for 't_run'.
    

    Parameters
    ----------
    exc: list
        contains indices of neurons for PoissonInput
    slnc: list
        contains indices of neurons to silence
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
    # define Poisson input for excitation
    poi_inp, neu = poi(neu, exc, params)
    # silence neurons
    syn = silence(slnc, syn)
    # collect in Network object
    net = Network(neu, syn, spk_mon, *poi_inp)

    # run simulation
    net.run(duration=params['t_run'])

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_exp(exp_name, exp_type, neu_exc, path_res, path_comp, path_con, params=default_params, name2flyid=dict(), neu_slnc=[], n_proc=-1):
    '''
    Run default network experiment with PoissonInputs as external input.
    Neurons chosen as Poisson sources spike with a default rate of 150 Hz
    and have a refractory period of 0 ms.

    Different types of experiments are implemented, which are chosen via 'exp_type'
        coac: Coactivation of all neurons. Here, 'exc' is a list
            of neuron names, which are all set as Poisson inputs. The experiment
            consists of 30 trials of 1 s each.
        dly: Delayed activation of neurons. Here, 'exc' is a tuple containing 
            two lists: the first is active from the start of the simulation,
            the second is activated after 1 s. The experiments consists of 
            30 trials of 1 s + 1 s each. 
        slnc: Same as 'coac', but additionally silence neurons defines in 'neu_slnc'

    Parameters
    ----------
        exp_name: str
            name of the experiment
        exp_type: str
            type of the experiment (coac | dly)
        neu_exc: list or tuple
            contains custom names or flywire IDs of neurons to be excited (depending on exc_type, see above)
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
        n_proc: int
            number of cores to be used for parallel runs
            default: -1 (use all available cores)
            n_proc=1 is equivalent serial code
    '''

    # convert to Path objects
    path_res, path_comp, path_con = [ Path(i) for i in [path_res, path_comp, path_con] ]

    # load name/id mappings
    _, _, i2flyid, _, _, name_flyid2i = useful_mappings(name2flyid, path_comp)
    
    # define output files
    out_pkl = path_res / '{}_{}.pickle'.format(exp_type, exp_name)

    # print info
    print('>>> Experiment:     {}'.format(exp_name))
    print('    Output files:   {}'.format(out_pkl))

    # start time for simulation
    start = time() 

    # start parallel calculation
    n_run = params['n_run']
    with parallel_backend('loky', n_jobs=n_proc):

        if exp_type == 'coac':
            print('    Exited neurons: {}'.format(' '.join([str(i) for i in neu_exc])))
            exc = [ name_flyid2i[n] for n in neu_exc ]
            res = Parallel()(
                delayed(
                    run_trial_coac)(exc, path_comp, path_con, params) for _ in range(n_run))
            
        elif exp_type == 'dly':
            for i, e in enumerate(neu_exc):
                print('    Exited neurons: {}: {}'.format(i, ' '.join([str(i) for i in e])))
            exc = tuple( [ name_flyid2i[n] for n in o ]  for o in neu_exc )
            res = Parallel()(
                delayed(
                    run_trial_dly)(exc, path_comp, path_con, params) for _ in range(n_run))
        
        elif exp_type == 'slnc':
            print('    Exited neurons: {}'.format(' '.join([str(i) for i in neu_exc])))
            print('    Silenced neurons: {}'.format(' '.join([str(i) for i in neu_slnc])))
            exc = [ name_flyid2i[n] for n in neu_exc ]
            slnc = [ name_flyid2i[n] for n in neu_slnc ]
            res = Parallel()(
                delayed(
                    run_trial_slnc)(exc, slnc, path_comp, path_con, params) for _ in range(n_run))
             
        else:
            raise NameError('Unknown exp_type: {}'.format(exp_type))
        
    # print simulation time
    walltime = time() - start 
    print('    Elapsed time:   {} s'.format(int(walltime)))
    t_run = params['t_run'] if type(exc) == tuple else len(exc) * params['t_run']

    # dataframe with spike times
    df = construct_dataframe(res, exp_name, exc, i2flyid)

    # store spike data and experiment metadata
    data = {
        'spk_ts':       df,
        'exp_name':     exp_name,
        'exp_type':     exp_type,
        'neu_exc':     neu_exc,
        'name2flyid':   name2flyid,
        'name_flyid2i': name_flyid2i,
        't_run':        t_run,
        'n_run':        n_run,
        'path_res':     path_res,
        'path_comp':    path_comp,
        'path_con':     path_con,
        'n_proc':       n_proc,
        'walltime':     walltime,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)