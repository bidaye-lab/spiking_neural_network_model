import pandas as pd
import pickle

from utils import load_dfs, load_dicts

from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz, Mohm, uF

# define trials
t_sim   = 1000 * ms         # duration of trial
n_run   = 30                # number of runs

# define network
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
r_poi   = 150*Hz            # rate of the Poisson input
w_poi   = w_syn*250         # strength of Poisson

#                           # equations for neurons
eqs = '''
dv/dt = (x - (v - v_0)) / t_mbr : volt (unless refractory)
dx/dt = -x / tau                : volt (unless refractory) 
rfc                             : second
'''
eq_th   = 'v > v_th'        # condition for spike
eq_rst  = 'v = v_rst; w = 0; x = 0 * mV' # rules when spike 

# network createion
def poi(neu, names, name2i, rate=r_poi):
    'creates a list of PoissonInput objects for a list of neuron names and NeuronGroup neu'
    l = []
    for n in names:
        i = name2i[n]
        p = PoissonInput(target=neu[i], target_var='v', N=1, rate=rate, weight=w_poi)
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        l.append(p)
        
    return l, neu

def create_model(path_comp, path_con):
    '''create default model for neurons and synapses from flywire data
    relies on equations and parameters defined above
    returns NeuronGroup, Synapses
    '''

    df_comp, df_con = load_dfs(path_comp, path_con)

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

# running simulations
def run_trial(args):
    
    exc, name2i, path_comp, path_con = args

    neu, syn, spk_mon = create_model(path_comp, path_con) # get default network
    poi_inp, neu = poi(neu, exc, name2i, rate=r_poi) # define Poisson input for excitation
    net = Network(neu, syn, spk_mon, *poi_inp)  # define network

    net.run(duration=t_sim) # run simulation

    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}

    return spk_trn


def run_exp(exp, exc, path_out, path_name, path_comp, path_con, n_proc=-1, basename='model_530'):
    '''run default experiment
    supply name (exp) and list of neuron names to excite (exc)'''

    # convert to Path objects
    path_out, path_name, path_comp, path_con = [ Path(i) for i in [path_out, path_name, path_comp, path_con] ]

    # load name dicts
    flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i = load_dicts(path_name)

    # define output files
    out_fth = path_out / '{}_{}.feather'.format(basename, exp)
    out_pkl = path_out / '{}_{}.pickle'.format(basename, exp)

    print('>>> Experiment:     {}'.format(exp))
    print('    Output files:   {}'.format(out_fth))
    print('                    {}'.format(out_pkl))
    print('    Exited neurons: {}'.format(' '.join(exc)))

    args = [ (exc, name2i, path_comp, path_con) for _ in range(n_run) ]

    with parallel_backend('loky', n_jobs=n_proc):
        res = Parallel()(map(delayed(run_trial), args))

    # df_spk = pd.DataFrame(res).T # dataframe with spike times
    df_spk = pd.DataFrame(res, index=['run_{}'.format(i) for i, _ in enumerate(res)], columns=i2flyid.keys())
    # df_spk.columns = [ 'run_{}'.format(i) for i, _ in enumerate(res) ]
    df_spk.T.to_feather(out_fth) # write to disk

    # store metadata
    meta_data = {
        'exp':      exp,
        'exc':      exc,
        'exc_i':    [name2flyid[i] for i in exc], 
        't_sim':    t_sim,
        'n_run':    n_run,
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(meta_data, f)