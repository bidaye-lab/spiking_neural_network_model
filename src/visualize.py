
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['savefig.facecolor'] = 'w'

def plot_raster(df_spkt, neu, name2id=dict(), xlims=(None, None), figsize=(), path=None):
    '''Plot raster plots for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    name2id : dict, optional
        Mapping betwen custon neuron names and database IDs, by default dict()
    xlims : tuple, optional
        xlims for plot, by default (None, None)
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 3*n_neu, 2*n_exp
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axmat = plt.subplots(ncols=n_neu, nrows=n_exp, squeeze=False, figsize=(dx, dy))

    for i, (e, df_exp) in enumerate(df_spkt.groupby('exp_name')):

        trl_max = df_exp.max()['trial'] # for axis limits

        gr_neu = df_exp.groupby('database_id')
        for j, n in enumerate(neu):
            ax = axmat[i,j]

            idx = name2id.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
            
                for trl, df_trl in df_neu.groupby('trial'):
                    t = df_trl.loc[:, 't']
                    ax.eventplot(t, lineoffset=trl, linewidths=.5)

            except KeyError:
                pass
            
            # formatting
            if j == 0:
                ax.set_ylabel(e)
            else:
                ax.set_yticklabels('')
                
            if i == 0:
                ax.set_title(n)

            ax.grid(None)
            ax.set_xlim(xlims)
            ax.set_ylim(-0.5, trl_max + 0.5)
           

    for ax in axmat[-1]:
        ax.set_xlabel('time [s]')
    fig.tight_layout()

    if path:
        fig.savefig(path)


def plot_rate(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, name2id=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    name2id : dict, optional
        Mapping betwen custon neuron names and database IDs, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp = len(exp)

    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, 4
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))
    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('database_id')

        df_bin = pd.DataFrame()

        for n in neu:
            idx = name2id.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
                gr_trl = df_neu.groupby('trial')

                for trl in range(n_trl):

                    try: 
                        df_trl = gr_trl.get_group(trl)
                        t = df_trl.loc[:, 't']
                    except KeyError:
                        t = []

                    y, _ = np.histogram(t, bins=bins)
                    y = gaussian_filter1d(y.astype(float), sigma=sigma, axis=0)
                    y *= 1e3
                    df = pd.DataFrame(data={
                        't' : bins[:-1],
                        'r': y,
                        'trl': trl,
                        'neu': n,
                    })
                    df_bin = pd.concat([df_bin, df], ignore_index=True)

            except KeyError:
                df = pd.DataFrame(data={
                    't' : bins[:-1],
                    'r': 0,
                    'neu': n,
                })
                df_bin = pd.concat([df_bin, df], ignore_index=True)

        if do_zscore:
            for n, df in df_bin.groupby('neu'):
                idx = df.index
                df_bin.loc[idx, 'r'] = zscore(df_bin.loc[idx, 'r'], ddof=1)

        sns.lineplot(data=df_bin, ax=ax, x='t', y='r', errorbar='sd', hue='neu')

        # formatting
        ax.legend()
        ax.set_title(e)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rate [Hz]')

    fig.tight_layout()
    if path:
        fig.savefig(path)

def plot_rate_heatmap(df_spkt, neu, xlims, sigma=25, n_trl=30, do_zscore=False, exclude_stim=False, color_range=(None, None), name2id=dict(), figsize=(), path=None):
    '''Plot rates for given experiments and neurons in a heatmap

    Parameters
    ----------
    df_spkt : pd.DataFrame
        Each row contains a spike event
    neu : list
        List of database IDs as appearing in df_spkt.
        `neu` can also contain custom neuron names, but in this case `name2id`
        must be supplied
    xlims : tuple
        xlims for plot [s]
    sigma : float, optional
        standard deviation for Gaussian kernel for smoothing [ms], by default 25
    n_trl : int, optional
        number of trials to calculate the avg rate, by default 30
    do_score : bool, optional
        If True, zscore the firing rate for each neuron, by default False
    exclude_stim : bool, optional
        If True, replace stimulated neurons with nan, by default False
    color_range : tuple, optional
        Values for min and max for the color map, by default (None, None)
    name2id : dict, optional
        Mapping betwen custon neuron names and database IDs, by default dict()
    figsize : tuple, optional
        dimension of the plot, passed to plt.subpolots
    path : str, optional
        Filename for saving the plot, by default None
    '''

    exp = df_spkt.loc[:, 'exp_name'].unique()
    n_exp, n_neu = len(exp), len(neu)
    if figsize:
        dx, dy = figsize
    else:
        dx, dy = 5 * n_exp, .25 * n_neu + 1
    print('INFO: setting figsize to ({}, {})'.format(dx, dy))

    fig, axarr = plt.subplots(ncols=n_exp, squeeze=False, figsize=(dx, dy))
    gr_exp = df_spkt.groupby('exp_name')
    
    bins = np.arange(*xlims, 1e-3)

    if do_zscore:
        cmap = 'coolwarm'
        norm = CenteredNorm()
    else: 
        cmap = 'viridis'
        norm = None

    for i, e in enumerate(exp):
        ax = axarr[0][i]

        df_exp = gr_exp.get_group(e)
        gr_neu = df_exp.groupby('database_id')

        # stuff for excluding stim
        # TODO: make more pretty
        id_b = df_spkt.attrs['stim_ids'][e]
        b2f = pd.Series(df_exp.loc[:, 'database_id'].values, index=df_exp.loc[:, 'brian_id']).to_dict()
        id_f = [ b2f[i] for i in id_b ]

        Z = []
        for n in neu:
            idx = name2id.get(n, n)
            idx = int(idx)

            try:
                df_neu = gr_neu.get_group(idx)
                t = df_neu.loc[:, 't']
            except KeyError:
                t = []

            z, _ = np.histogram(t, bins=bins)
            z = gaussian_filter1d(z.astype(float), sigma=sigma, axis=0)
            z = z / n_trl * 1e3
            if do_zscore:
                z = zscore(z, ddof=1)

            if exclude_stim and idx in id_f:
                z[:] = np.nan

            Z.append(z)

        Z = np.vstack(Z)
        x = bins[:-1]
        y = np.arange(n_neu)
        im = ax.pcolormesh(x, y, Z, cmap=cmap,  norm=norm, vmin=color_range[0], vmax=color_range[1])
        fig.colorbar(im, ax=ax, location='right', orientation='vertical')

        # TODO colorbar label and xlabel

        ax.set_yticks(y)
        ax.set_yticklabels(neu)

        # formatting
        ax.set_title(e)
        ax.set_xlabel('time [s]')

    fig.tight_layout()

    if path:
        fig.savefig(path)


def firing_rate_matrix(df_rate, rate_change=False, scaling=.5, path=''):
    '''Plot heatmap showing the firing rates of neurons in different experiments
 
    Parameters
    ----------
    df_rate : pd.DataFrame
        Rate data with experiments as columns and neurons as index
    rate_change : bool, optional
        If True, use diverging colormap and center on 0, by default False
    scaling : float, optional
        Scales figure size, by default .5
    path : path-like, optional
        Filename for saving the plot, by default ''
    '''

    # figure dimensions
    n_neu, n_exp = df_rate.shape
    figsize = (scaling*n_exp, scaling*n_neu)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('firing rate [Hz]')

    
    if rate_change: # plot settings for rate changes
        heatmap_kw_args = {
            'cmap': 'coolwarm',
            'center': 0,
        }
    else: # plot settings for absolute rates
        heatmap_kw_args = {
            'cmap': 'viridis',
        }

    sns.heatmap(
        ax=ax, data=df_rate, square=True,
        xticklabels=True, yticklabels=True,
        annot=True, fmt='.1f', annot_kws={'size': 'small'},
        cbar=False, **heatmap_kw_args,
    )
    ax.tick_params(axis='x', labeltop=True, labelbottom=True, labelrotation=90)
    
    if path:
        fig.savefig(path)
        plt.close(fig)