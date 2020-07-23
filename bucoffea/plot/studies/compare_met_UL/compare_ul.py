#!/usr/bin/env python

import os
import re
import sys
import uproot
import pandas as pd
import numpy as np
import mplhep as hep
from matplotlib import pyplot as plt
from bucoffea.helpers.paths import bucoffea_path

pjoin = os.path.join

# Script to compare MET datasets: One being currently used and the UL dataset

def prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take):
    '''Given the input trees and the branches to take, prepare a merged dataframe, merged on run/lumi/event information.'''
    # Transform the individual trees into dataframes
    df_05Jun20v5 = tree_05Jun20v5.pandas.df()[branches_to_take]
    df_UL = tree_UL.pandas.df()[branches_to_take]

    # Merge the two dataframes on event/run/lumi information
    merged_df = pd.merge(df_05Jun20v5, df_UL, how='inner', on=['run', 'lumi', 'event'], suffixes=('_05Jun20v5','_UL'))

    return merged_df

def plot_met_comparison_for_large_eta(merged_df, eta_range=(3.0,5.0)):
    '''Given the merged dataframe and the eta range, plot distribution of MET for UL and non-UL.'''
    leading_jet_abseta = np.abs(merged_df['leadak4_eta_05Jun20v5'])
    trailing_jet_abseta = np.abs(merged_df['trailak4_eta_05Jun20v5'])
    met_pt_05Jun20v5 = merged_df['met_pt_05Jun20v5']
    met_pt_UL = merged_df['met_pt_UL']

    # Get the events where one of the leading jets is in the given eta range
    if eta_range is not None:
        low_eta, high_eta = eta_range
        mask = ((leading_jet_abseta > low_eta) & (leading_jet_abseta < high_eta)) | ((trailing_jet_abseta > low_eta) & (trailing_jet_abseta < high_eta))
    else:
        mask = np.ones_like(met_pt_05Jun20v5, dtype=bool)

    met_pt_05Jun20v5_masked = met_pt_05Jun20v5[mask]
    met_pt_UL_masked = met_pt_UL[mask]

    # Make a histogram for both cases and plot them both
    met_bins = [ 150, 175, 200, 225, 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640]
    met_histo_05Jun20v5, bins = np.histogram(met_pt_05Jun20v5_masked, bins=met_bins)
    met_histo_UL, bins = np.histogram(met_pt_UL_masked, bins=met_bins)

    fig, ax = plt.subplots()
    hep.histplot(met_histo_05Jun20v5, bins, ax=ax, histtype='step', label='05Jun20v5')
    hep.histplot(met_histo_UL, bins, ax=ax, histtype='step', label='UL')

    ax.set_xlabel('MET (GeV)')
    ax.set_ylabel('Events in Data')
    ax.legend()

    # Set fig title
    if eta_range is None:
        fig_title = r'$\eta$ Inclusive'
    else:
        fig_title = r'One Leading Jet In ${} < |\eta| < {}$'.format(low_eta, high_eta)

    ax.set_title(fig_title)

    # Save figure
    outdir = f'./output/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if eta_range is not None:
        filename = f'etaRange_{str(low_eta).replace(".", "_")}_{str(high_eta).replace(".","_")}.pdf' 
    else:
        filename = f'etaRange_inclusive.pdf' 

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
    ]

    merged_df = prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take)

    # Plot the MET comparison
    plot_met_comparison_for_large_eta(merged_df)
    plot_met_comparison_for_large_eta(merged_df, eta_range=(2.5,3.0))
    plot_met_comparison_for_large_eta(merged_df, eta_range=None)

if __name__ == '__main__':
    main()



