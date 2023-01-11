import os

import pandas as pd
import pickle

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network
from brian2 import mV, ms, Hz, Mohm, uF

# load neuron data
df_comp = pd.read_csv('./data/2022_11_22_completeness_materialization_530_final.csv', index_col = 0) # neuron ids and excitation type
df_con = pd.read_csv('./data/2022_11_22_connectivity_530_final.csv', index_col = 0) # connectivity

# load name mappings
with open('./data/name_mappings_530.pickle', 'rb') as f:
    flyid2i, flyid2name, i2flyid, i2name, name2flyid, name2i = pickle.load(f)

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

def poi(neu, names, rate=r_poi):
    'creates a list of PoissonInput objects for a list of neuron names and NeuronGroup neu'
    l = []
    for n in names:
        i = name2i[n]
        p = PoissonInput(target=neu[i], target_var='v', N=1, rate=rate, weight=w_poi)
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        l.append(p)
        
    return l, neu

def default_model():
    '''create default model for neurons and synapses from flywire data
    relies on equations and parameters defined above
    returns NeuronGroup, Synapses
    '''
    
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

def run(exc): 
    pid = os.getpid()
    neu, syn, spk_mon = default_model() 
    poi_inp, neu = poi(neu, exc)
    net = Network(neu, syn, spk_mon, *poi_inp)  
    net.run(duration=1000*ms)
    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    return spk_trn