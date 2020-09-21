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
    'leadak4_eta',
    'leadak4_phi',
    'trailak4_pt',
    'trailak4_eta',
    'trailak4_phi'
]

binnings = {
    'leadak4_pt' : list(range(80,600,20)),
    'trailak4_pt' : list(range(80,600,20)),
    'leadak4_eta' : np.linspace(-5,5,51),
    'trailak4_eta' : np.linspace(-5,5,51)
}

pretty_labels = {
    'leadak4_pt' : r'$p_{T,0}$',
    'trailak4_pt' : r'$p_{T,1}$',
    'leadak4_eta' : r'$\eta_{0}$',
    'trailak4_eta' : r'$\eta_{1}$',
    'mjj' : r'$M_{jj}$'
}

def prepare_merged_df(tree_noSmear, tree_withSmear, proc='znunu'):
    proc_to_region = {
        'znunu' : 'sr_vbf',
        'zmumu' : 'cr_2m_vbf'
    }
    region = proc_to_region[proc]
    f_noSmear = uproot.open(tree_noSmear)[region]
    f_withSmear = uproot.open(tree_withSmear)[region]

    df_noSmear = f_noSmear.pandas.df()[cols_for_pandas]
    df_withSmear = f_withSmear.pandas.df()[cols_for_pandas]

    df = pd.merge(df_noSmear, df_withSmear, on='event', suffixes=['_ns', '_ws'])

    return df

def plot_eta_distribution(tree_noSmear, tree_withSmear, diff_variable='mjj', diff_thresh=0.1, proc='znunu'):
    '''Plot jet eta distributions for events with large differences when the smearing is applied.'''
    df = prepare_merged_df(tree_noSmear, tree_withSmear, proc)

    percent_diff = np.abs(df[f'{diff_variable}_ns'] - df[f'{diff_variable}_ws']) / df[f'{diff_variable}_ns']
    mask = percent_diff > diff_thresh

    # Plot variables with high variable diff.
    outdir = './output/large_diff/dists'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Mask the df to get only the events which have large changes with smearing applied
    df = df[mask]
    eta_binning = np.linspace(-5,5,51)

    variables_to_plot = ['leadak4_eta', 'trailak4_eta']
    for variable in variables_to_plot:
        vals_noSmear = df[f'{variable}_ns']

        hist, edges = np.histogram(vals_noSmear, bins=eta_binning) 
        fig, ax = plt.subplots()
        hep.histplot(hist, edges, ax=ax)

        ax.set_ylabel('Counts')
        ax.set_xlabel(pretty_labels[variable])

        ax.set_title(f'% change in {pretty_labels[diff_variable]} > {diff_thresh*100}')

        outpath = pjoin(outdir, f'{variable}_{diff_variable}_{str(diff_thresh).replace(".", "_")}.pdf')
        fig.savefig(outpath)
        print(f'MSG% File saved: {outpath}')
        plt.close(fig)

def compare_events_with_large_diff(tree_noSmear, tree_withSmear, diff_variable='mjj', diff_thresh=0.1, proc='znunu'):
    '''Compare events with large differences when the smearing is applied.'''
    df = prepare_merged_df(tree_noSmear, tree_withSmear, proc)

    percent_diff = np.abs(df[f'{diff_variable}_ns'] - df[f'{diff_variable}_ws']) / df[f'{diff_variable}_ns']
    mask = percent_diff > diff_thresh

    # Plot variables with high variable diff.
    outdir = './output/large_diff'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Mask the df to get only the events which have large changes with smearing applied
    df = df[mask]
    bins = np.linspace(-1,1,51)

    # print(df[mask][['event', 'leadak4_pt_ns', 'leadak4_pt_ws', 'trailak4_pt_ns', 'trailak4_pt_ws', 'mjj_ns', 'mjj_ws']])

    variables_to_plot = ['leadak4_pt', 'trailak4_pt']
    for variable in variables_to_plot:
        vals_noSmear = df[f'{variable}_ns']
        vals_withSmear = df[f'{variable}_ws']

        # Calculate the relevant difference in these variables
        diff = (vals_noSmear - vals_withSmear) / vals_withSmear
        hist, edges = np.histogram(diff, bins=bins)

        # Plot the relative difference
        fig, ax = plt.subplots()
        hep.histplot(hist, edges, ax=ax)

        ax.set_ylabel('Counts')
        ax.set_xlabel(f'% change in {pretty_labels[variable]}')

        ax.set_title(f'% change in {pretty_labels[diff_variable]} > {diff_thresh*100}')

        outpath = pjoin(outdir, f'{variable}_{diff_variable}_{str(diff_thresh).replace(".", "_")}.pdf')
        fig.savefig(outpath)
        print(f'MSG% File saved: {outpath}')
        plt.close(fig)

def main():
    # Read the process from command line (i.e. zmumu or znunu)
    proc = sys.argv[1]
    # Tree files with and without smearing applied
    if proc == 'znunu':
        tree_noSmear = './input_trees/18Sep20/noJER/tree_ZJetsToNuNu_HT-400To600-mg_new_pmx_2017.root'
        tree_withSmear = './input_trees/18Sep20/withJER/tree_ZJetsToNuNu_HT-400To600-mg_new_pmx_2017.root'
    elif proc == 'zmumu': # TODO: Update!
        tree_noSmear = None
        tree_withSmear = None

    compare_events_with_large_diff(tree_noSmear, tree_withSmear, proc)
    plot_eta_distribution(tree_noSmear, tree_withSmear, proc)

if __name__ == '__main__':
    main()