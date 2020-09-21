#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import pandas as pd
import mplhep as hep
from matplotlib import pyplot as plt

pjoin = os.path.join

cols_for_pandas = [
    'event',
    'mjj', 'recoil_pt',
    'recoil_phi',
    'met_pt_nom',
    'met_phi_nom',
    'leadak4_pt',
    'leadak4_pt_jer',
    'leadak4_eta',
    'leadak4_phi',
    'trailak4_pt',
    'trailak4_pt_jer',
    'trailak4_eta',
    'trailak4_phi'
]

binnings = {
    'mjj' : list(range(200,5000,100)),
    'leadak4_pt' : list(range(80,600,20)),
    'leadak4_pt_jer' : list(range(80,600,20)),
    'trailak4_pt' : list(range(20,40,10)),
    'trailak4_pt_jer' : list(range(20,100,5)),
    'leadak4_eta' : np.linspace(-5,5,51),
    'trailak4_eta' : np.linspace(-5,5,51)
}

pretty_labels = {
    'leadak4_pt' : r'$p_{T,0}$',
    'leadak4_pt_jer' : r'Smeared $p_{T,0}$',
    'trailak4_pt' : r'$p_{T,1}$',
    'trailak4_pt_jer' : r'Smeared $p_{T,1}$',
    'leadak4_eta' : r'$\eta_{0}$',
    'trailak4_eta' : r'$\eta_{1}$',
    'mjj' : r'$M_{jj}$'
}

def get_events_failing_smeared_jet_pt(f, proc='znunu'):
    '''Get events which have unsmeared trailing jet pt > 40 GeV, but smeared trailing jet pt < 40 GeV,
    using the region with relaxed cuts.'''
    region = 'sr_vbf' if proc == 'znunu' else 'cr_2m_vbf'
    events = f[f'{region}_relaxed_trailak4'].pandas.df()[cols_for_pandas]

    # Out of all events, get the ones we are interested in
    total_num_events = len(events)
    print(f'Total number of events: {total_num_events}')
    trailak4_pt = events['trailak4_pt']
    trailak4_pt_jer = events['trailak4_pt_jer']
    mask = (trailak4_pt_jer < 40) & (trailak4_pt > 40)

    events_masked = events[mask]
    print(f'Number of events with trailing jet failing the cut: {len(events_masked)}')

    return events_masked

def get_events_failing_unsmeared_jet_pt(f, proc='znunu'):
    '''Get events which have smeared trailing jet pt > 40 GeV, but non-smeared trailing jet pt < 40 GeV,
    using the region with relaxed cuts.'''
    region = 'sr_vbf' if proc == 'znunu' else 'cr_2m_vbf'
    events = f[f'{region}_relaxed_trailak4'].pandas.df()[cols_for_pandas]

    # Out of all events, get the ones we are interested in
    total_num_events = len(events)
    print(f'Total number of events: {total_num_events}')
    trailak4_pt = events['trailak4_pt']
    trailak4_pt_jer = events['trailak4_pt_jer']
    mask = (trailak4_pt_jer > 40) & (trailak4_pt < 40)

    events_masked = events[mask]
    print(f'Number of events with trailing jet failing the cut: {len(events_masked)}')

    return events_masked

def plot_distribution(events, ht_bin, variable='mjj', proc='znunu'):
    '''For the given list of events, plot distribution of given variable.'''
    variable_arr = events[variable]
    # Make a histogram for the variable
    h, edges = np.histogram(variable_arr, bins=binnings[variable])

    # Plot the histogram
    fig, ax = plt.subplots()
    hep.histplot(h, edges, ax=ax)
    
    ax.set_xlabel(pretty_labels[variable])
    if variable == 'mjj':
        ax.set_yscale('log')
        ax.set_ylim(1e-1, 1e4)

    # Save figure
    outdir = f'./output/19Sep20_trailak4_check/{proc}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{variable}_dist_{ht_bin}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')
    plt.close(fig)

def main():
    proc, ht_bin = sys.argv[1:]
    # Input Z(vv) tree files for different HT bins
    if proc == 'znunu':
        input_trees = {
            '100To200' : './input_trees/19Sep20/tree_ZJetsToNuNu_HT-100To200-mg_2017.root',
            '400To600' : './input_trees/19Sep20/tree_ZJetsToNuNu_HT-400To600-mg_new_pmx_2017.root',
            '600To800' : './input_trees/19Sep20/tree_ZJetsToNuNu_HT-600To800-mg_new_pmx_2017.root',
            '1200To2500' : './input_trees/19Sep20/tree_ZJetsToNuNu_HT-1200To2500-mg_new_pmx_2017.root',
            '2500ToInf' : './input_trees/19Sep20/tree_ZJetsToNuNu_HT-2500ToInf-mg_2017.root',
        }
    elif proc == 'zmumu':
        # TODO: Put the paths to trees here!
        input_trees = {}
    
    input_tree = input_trees[ht_bin]
    f = uproot.open(input_tree)

    # Get the list of events which have smeared trailing jet pt > 40 GeV, non-smeared trailing jet pt < 40 GeV
    events_masked = get_events_failing_unsmeared_jet_pt(f, proc)
    # Get the list of events which have smeared trailing jet pt < 40 GeV, non-smeared trailing jet pt > 40 GeV
    events_masked_v2 = get_events_failing_smeared_jet_pt(f, proc)

    # With the masked events, plot several distributions
    variables = ['mjj', 'leadak4_pt', 'leadak4_pt_jer', 'trailak4_pt', 'trailak4_pt_jer', 'leadak4_eta', 'trailak4_eta']
    for variable in variables:
        plot_distribution(events_masked, ht_bin, variable, proc)

if __name__ == '__main__':
    main()