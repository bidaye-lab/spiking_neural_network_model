import pandas as pd
from utils import useful_mappings

# brian 2
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz, Mohm, uF

# file handling
import pickle
from pathlib import Path

# parallelization
from joblib import Parallel, delayed, parallel_backend
from time import time

#####################
# trials and duration
t_sim   = 1000 * ms         # duration of trial
n_run   = 30                # number of runs

###################
# network constants
#                           # Kakaria and de Bivort 2017 https://doi.org/10.3389/fnbeh.2017.00008
#                           # refereneces therein, e.g. Hodgkin and Huxley 1952
v_0     = -52 * mV          # resting potential
v_rst   = -52 * mV          # reset potential after spike
v_th    = -45 * mV          # threshold for spiking
r_mbr   = 10. * Mohm        # membrane resistance
c_mbr   = .002 * uF         # membrane capacitance 
t_mbr   = c_mbr * r_mbr     # membrane time scale

#                           # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
tau     = 5 * ms            # time constant (this is the excitatory one, the inhibitory is 10 ms)

#                           # Lazar et at https://doi.org/10.7554/eLife.62362
#                           # they cite Kakaria and de Bivort 2017, but those have used 2 ms
t_rfc   = 2.2 * ms          # refractory period

#                           # Paul et al 2015 doi: 10.3389/fncel.2015.00029
t_dly   = 1.8*ms            # delay for changes in post-synaptic neuron

#                           # adjusted arbitrarily
w_syn   = .275 * mV         # weight per synapse (note: modulated by exponential decay)
r_poi   = 150*Hz            # default rate of the Poisson input
w_poi   = w_syn*250         # strength of Poisson

###################
# network equations
#                           # equations for neurons
eqs = '''
dv/dt = (x - (v - v_0)) / t_mbr : volt (unless refractory)
dx/dt = -x / tau                : volt (unless refractory) 
rfc                             : second
'''
eq_th   = 'v > v_th'        # condition for spike
eq_rst  = 'v = v_rst; w = 0; x = 0 * mV' # rules when spike 

#######################
# brian2 model setup
def poi(neu, exc, rate=r_poi):
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

    Returns
    -------
    pois : list
        PoissonInput objects for each neuron in 'exc'
    neu : NeuronGroup
        NeuronGroup with adjusted refractory periods
    '''

    pois = []
    for i in exc:
        p = PoissonInput(target=neu[i], target_var='v', N=1, rate=rate, weight=w_poi)
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        pois.append(p)
        
    return pois, neu

def create_model(path_comp, path_con):
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
        model=eqs,
        method='linear',
        threshold=eq_th,
        reset=eq_rst,
        refractory='rfc',
        name='default_neurons', 
    )
    neu.v = v_0 # initialize values
    neu.x = 0
    neu.rfc = t_rfc

    # create synapses
    syn = Synapses(neu, neu, 'w : volt', on_pre='x += w', delay=t_dly, name='default_synapses')

    # connect synapses
    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    # define connection weight
    syn.w = df_con.loc[:,"Excitatory x Connectivity"].values * w_syn

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

def run_trial_coac(exc, path_comp, path_con, r_poi):
    '''Run single trial of coactivation experiment

    During the coactivation experiment, the neurons in 'exc' are
    Poisson inputs. The simulation runs for 't_sim'.
    

    Parameters
    ----------
    exc: list
        contains indices of neurons for PoissonInput
    path_comp: Path 
        path to "completeness materialization" dataframe
    path_con: Path
        path to "connectivity" dataframe

    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''


    # get default network
    neu, syn, spk_mon = create_model(path_comp, path_con)
    # define Poisson input for excitation
    poi_inp, neu = poi(neu, exc, rate=r_poi)
    # collect in Network object
    net = Network(neu, syn, spk_mon, *poi_inp)

    # run simulation
    net.run(duration=t_sim)

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_trial_dly(exc_tup, path_comp, path_con, r_poi):
    '''Run single trial of delayed activation experiment

    During the delayed activation experiment, groups of neurons
    are made Poisson inputs consecutively during the simulation.
    Here, 'exc' is a tuple of lists, where each list contains neurons
    to be added after 't_sim'.
    E.g. if exc=([1, 2], [3]) and t_sim=1*s, the simulations runs
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


    Returns
    -------
    spk_trn : dict
        Mapping between brian neuron IDs and spike times
    '''

    # get default network
    neu, syn, spk_mon = create_model(path_comp, path_con)
    net = Network(neu, syn, spk_mon)

    for exc in exc_tup:
        # add Poisson inputs to network
        poi_inp, neu = poi(neu, exc, rate=r_poi)
        net.add(*poi_inp)

        # run simulation
        net.run(duration=t_sim)

    # spike times 
    spk_trn = get_spk_trn(spk_mon)

    return spk_trn


def run_exp(exp_name, exp_type, exc_name, name2flyid, path_res, path_comp, path_con, r_poi=r_poi, n_proc=-1):
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

    Parameters
    ----------
        exp_name: str
            name of the experiment
        exp_type: str
            type of the experiment (coac | dly)
        exc_name: list or tuple
            contains custom names or flywire IDs of neurons to be excited (depending on exc_type, see above)
        name2flyid : dict
            Mapping between custom neuron names and flywire IDs
        path_res: str
            path to the output folder where spike data is stored
        path_comp: str 
            path to "completeness materialization" dataframe
        path_con: str
            path to "connectivity" dataframe
        r_poi : brian2.Hz
            Rate of the Poisson input (default hardcoded at the top of this file)
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
    with parallel_backend('loky', n_jobs=n_proc):
        if exp_type == 'coac':
            print('    Exited neurons: {}'.format(' '.join(exc_name)))
            exc = [ name_flyid2i[n] for n in exc_name ]
            res = Parallel()(
                delayed(
                    run_trial_coac)(exc, path_comp, path_con, r_poi) for _ in range(n_run))
        elif exp_type == 'dly':
            for i, e in enumerate(exc_name):
                print('    Exited neurons: {}: {}'.format(i, ' '.join(e)))
            exc = tuple( [ name_flyid2i[n] for n in o ]  for o in exc_name )
            res = Parallel()(
                delayed(
                    run_trial_dly)(exc, path_comp, path_con, r_poi) for _ in range(n_run))
        else:
            raise NameError('Unknown exp_type: {}'.format(exp_type))
        
    # print simulation time
    walltime = time() - start 
    print('    Elapsed time:   {} s'.format(int(walltime)))
                
    # dataframe with spike times
    df = construct_dataframe(res, exp_name, exc, i2flyid)

    # store spike data and experiment metadata
    data = {
        'spk_ts':       df,
        'exp_name':     exp_name,
        'exp_type':     exp_type,
        'exc_name':     exc_name,
        'name2flyid':   name2flyid,
        'name_flyid2i': name_flyid2i,
        't_sim':        t_sim if type(exc) == tuple else len(exc) * t_sim,
        'n_run':        n_run,
        'path_res':     path_res,
        'path_comp':    path_comp,
        'path_con':     path_con,
        'n_proc':       n_proc,
        'walltime':     walltime,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)