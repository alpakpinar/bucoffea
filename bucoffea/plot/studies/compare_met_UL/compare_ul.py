#!/usr/bin/env python

import os
import re
import sys
import uproot
import pandas as pd
import numpy as np
import mplhep as hep
import argparse
import matplotlib.ticker
from matplotlib import pyplot as plt
from bucoffea.helpers.paths import bucoffea_path

pjoin = os.path.join

# Script to compare MET datasets: One being currently used and the UL dataset

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variables', help='The list of variables to plot the comparison.', default=['met_pt'], nargs='*')
    args = parser.parse_args()
    return args

def prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take):
    '''Given the input trees and the branches to take, prepare a merged dataframe, merged on run/lumi/event information.'''
    # Transform the individual trees into dataframes
    df_05Jun20v5 = tree_05Jun20v5.pandas.df()[branches_to_take]
    df_UL = tree_UL.pandas.df()[branches_to_take]

    # Add in the missing dphijj < 1.5 cut!
    dphijj_05Jun20v5 = df_05Jun20v5['dphijj']
    dphijj_UL = df_UL['dphijj']

    df_05Jun20v5 = df_05Jun20v5[dphijj_05Jun20v5 < 1.5]
    df_UL = df_UL[dphijj_UL < 1.5]

    # Merge the two dataframes on event/run/lumi information
    merged_df = pd.merge(df_05Jun20v5, df_UL, how='inner', on=['run', 'lumi', 'event'], suffixes=('_05Jun20v5','_UL'))

    return merged_df

def plot_met_comparison_for_large_eta(merged_df, variable='met_pt', eta_range=(3.0,5.0)):
    '''Given the merged dataframe and the eta range, plot distribution of MET for UL and non-UL.'''
    leading_jet_abseta = np.abs(merged_df['leadak4_eta_05Jun20v5'])
    trailing_jet_abseta = np.abs(merged_df['trailak4_eta_05Jun20v5'])

    # Get the relevant variable (met pt by default)
    arr_05Jun20v5 = merged_df[f'{variable}_05Jun20v5']
    arr_UL = merged_df[f'{variable}_UL']

    # Get the events where one of the leading jets is in the given eta range
    if eta_range is not None:
        low_eta, high_eta = eta_range
        mask = ((leading_jet_abseta > low_eta) & (leading_jet_abseta < high_eta)) | ((trailing_jet_abseta > low_eta) & (trailing_jet_abseta < high_eta))
    else:
        mask = np.ones_like(arr_05Jun20v5, dtype=bool)

    arr_05Jun20v5_masked = arr_05Jun20v5[mask]
    arr_UL_masked = arr_UL[mask]

    # Make a histogram for both cases and plot them both
    binning = {
        'met_pt' : [ 150, 175, 200, 225, 250,  280,  310,  340,  370,  400, 430, 470, 510, 550, 590, 640],
        'leadak4_pt' : list(range(80,500,20)),
        'trailak4_pt' : list(range(40,400,20)),
    }

    bins = binning[variable]
    histo_05Jun20v5, bins = np.histogram(arr_05Jun20v5_masked, bins=bins)
    histo_UL, bins = np.histogram(arr_UL_masked, bins=bins)

    # fig, ax = plt.subplots()
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(histo_05Jun20v5, bins, ax=ax, histtype='step', label='05Jun20v5')
    hep.histplot(histo_UL, bins, ax=ax, histtype='step', label='UL')

    xlabels = {
        'met_pt' : 'MET (GeV)',
        'leadak4_pt' : r'Leading Jet $p_T$ (GeV)',
        'trailak4_pt' : r'Trailing Jet $p_T$ (GeV)',
    }

    ax.set_ylabel('Events in Data')
    ax.legend()

    # Set fig title
    if eta_range is None:
        fig_title = r'$\eta$ Inclusive'
    else:
        fig_title = r'One Leading Jet In ${} < |\eta| < {}$'.format(low_eta, high_eta)

    ax.set_title(fig_title)

    ylim = ax.get_ylim()
    ax.plot([250., 250.], ylim, 'k')
    ax.set_ylim(ylim)

    # Calculate and plot the ratio
    ratio = histo_UL / histo_05Jun20v5
    # Guard against NaN values
    ratio[np.isnan(ratio) | np.isinf(ratio)] = 1.0

    data_err_opts = {
        'linestyle' : 'none',
        'marker' : '.',
        'markersize' : '10.',
        'color' : 'k'
    }

    hep.histplot(ratio, bins, ax=rax, histtype='errorbar', **data_err_opts)

    rax.set_xlabel(xlabels[variable])
    rax.set_ylabel('UL / 05Jun20v5')
    rax.grid(True)

    rax.plot([250., 250.], [0.6, 1.4], color='blue')
    rax.set_ylim(0.6, 1.4)

    loc = matplotlib.ticker.MultipleLocator(base=0.1)
    rax.yaxis.set_major_locator(loc)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.], 'r')
    rax.set_xlim(xlim)

    # Save figure
    outdir = f'./output/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if eta_range is not None:
        filename = f'{variable}_etaRange_{str(low_eta).replace(".", "_")}_{str(high_eta).replace(".","_")}.pdf' 
    else:
        filename = f'{variable}_etaRange_inclusive.pdf' 

    outpath = pjoin(outdir, filename)    
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    path_to_tree_05Jun20v5 = bucoffea_path('./scripts/trees/23Jul20/05Jun20v5/tree_MET_2017E.root')
    path_to_tree_UL = bucoffea_path('./scripts/trees/23Jul20/UL/tree_MET_2017E.root')

    # NOTE: These trees have MET>150 GeV cut applied
    tree_05Jun20v5 = uproot.open(path_to_tree_05Jun20v5)['sr_vbf']
    tree_UL = uproot.open(path_to_tree_UL)['sr_vbf']

    # Transform the trees into pandas dataframes
    branches_to_take = [
        'event', 'run', 'lumi',
        'met_pt', 'met_phi',
        'leadak4_pt', 'leadak4_eta', 
        'trailak4_pt', 'trailak4_eta', 
        'dphijj', 'detajj'
    ]

    merged_df = prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take)

    # Read in the list of variables
    args = parse_cli()
    variables = args.variables

    for variable in variables:
        # Plot the MET comparison
        plot_met_comparison_for_large_eta(merged_df, variable=variable)
        plot_met_comparison_for_large_eta(merged_df, variable=variable, eta_range=(2.5,3.0))
        plot_met_comparison_for_large_eta(merged_df, variable=variable, eta_range=None)

if __name__ == '__main__':
    main()



