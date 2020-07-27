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
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from bucoffea.helpers.paths import bucoffea_path

pjoin = os.path.join

# Script to compare MET datasets: One being currently used and the UL dataset

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variables', help='The list of variables to plot the comparison.', default=['met_pt'], nargs='*')
    parser.add_argument('--applyMetCut', help='Apply MET>250 cut.', action='store_true')
    parser.add_argument('--plot2d', help='Plot 2D histogram of UL and non-UL MET.', action='store_true')
    args = parser.parse_args()
    return args

def prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take, apply_met_cut=False):
    '''Given the input trees and the branches to take, prepare a merged dataframe, merged on run/lumi/event information.'''
    # Transform the individual trees into dataframes
    df_05Jun20v5 = tree_05Jun20v5.pandas.df()[branches_to_take]
    df_UL = tree_UL.pandas.df()[branches_to_take]

    # Add in the missing dphijj < 1.5 cut!
    dphijj_05Jun20v5 = df_05Jun20v5['dphijj']
    dphijj_UL = df_UL['dphijj']

    mask_05Jun20v5 = dphijj_05Jun20v5 < 1.5
    mask_UL = dphijj_UL < 1.5

    # If requested, apply the MET > 250 GeV cut (trees should have >150 cut applied)
    if apply_met_cut:
        met_05Jun20v5 = df_05Jun20v5['met_pt']
        met_UL = df_UL['met_pt']

        mask_05Jun20v5 = mask_05Jun20v5 & (met_05Jun20v5 > 250)
        mask_UL = mask_UL & (met_UL > 250)

    df_05Jun20v5 = df_05Jun20v5[mask_05Jun20v5]
    df_UL = df_UL[mask_UL]

    # Merge the two dataframes on event/run/lumi information
    merged_df = pd.merge(df_05Jun20v5, df_UL, how='inner', on=['run', 'lumi', 'event'], suffixes=('_05Jun20v5','_UL'))

    return merged_df

def get_mask_from_eta_range(eta_range, leading_jet_abseta, trailing_jet_abseta):
    # Get the events where one of the leading jets is in the given eta range
    if eta_range is not None:
        low_eta, high_eta = eta_range
        mask = ((leading_jet_abseta > low_eta) & (leading_jet_abseta < high_eta)) | ((trailing_jet_abseta > low_eta) & (trailing_jet_abseta < high_eta))
    else:
        mask = np.ones_like(leading_jet_abseta, dtype=bool)

    return mask

def get_masked_array(merged_df, variable='met_pt', eta_range=(3.0,5.0)):
    leading_jet_abseta = np.abs(merged_df['leadak4_eta_05Jun20v5'])
    trailing_jet_abseta = np.abs(merged_df['trailak4_eta_05Jun20v5'])

    # Get the relevant variable (met pt by default)
    arr_05Jun20v5 = merged_df[f'{variable}_05Jun20v5']
    arr_UL = merged_df[f'{variable}_UL']

    mask = get_mask_from_eta_range(eta_range, leading_jet_abseta, trailing_jet_abseta)

    arr_05Jun20v5_masked = arr_05Jun20v5[mask]
    arr_UL_masked = arr_UL[mask]

    arrays = {
        '05Jun20v5' : arr_05Jun20v5_masked,
        'UL' : arr_UL_masked
    }
    return arrays

def eta_distribution_for_events_with_high_met_diff(merged_df, met_diff_factor=0.2):
    '''Plot the leading/trailing jet eta distributions for events with high MET difference between UL and non-UL.'''
    met_arrays = get_masked_array(merged_df, variable='met_pt', eta_range=None)
    met_05Jun20v5 = met_arrays['05Jun20v5']
    met_UL = met_arrays['UL']

    leading_jet_eta = merged_df['leadak4_eta_05Jun20v5']
    trailing_jet_eta = merged_df['trailak4_eta_05Jun20v5']

    mask = ( np.abs(met_05Jun20v5 - met_UL) / met_05Jun20v5 ) > met_diff_factor

    events_with_high_met_diff = {
        'leadak4_eta' : leading_jet_eta[mask],
        'trailak4_eta' : trailing_jet_eta[mask]
    }

    binning = {
        'leadak4_eta' : np.linspace(-5,5,51),
        'trailak4_eta' : np.linspace(-5,5,51),
    }

    for tag in ['leadak4_eta', 'trailak4_eta']:
        fig, ax = plt.subplots()
        h, bins = np.histogram(events_with_high_met_diff[tag], bins=binning[tag])
        hep.histplot(h, bins, histtype='step', ax=ax)
        
        if tag == 'leadak4_eta':
            ax.set_xlabel(r'Leading jet $\eta$')
        elif tag == 'trailak4_eta':
            ax.set_xlabel(r'Trailing jet $\eta$')
        ax.set_ylabel('Number of Events')
    
        fig_title = f'Events with % MET difference > {met_diff_factor*100}%'
        ax.set_title(fig_title)
    
        ax.set_yscale('log')
        ax.set_ylim(1e-1, 1e4)

        outdir = f'./output/events_with_high_met_diff'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'{tag}_met_diff_factor_{str(met_diff_factor).replace(".", "_")}.pdf')
        fig.savefig(outpath)
        print(f'MSG% File saved: {outpath}')
        plt.close(fig)

def plot_2d_histogram(merged_df, variable='met_pt', eta_range=(3.0,5.0)):
    '''Plot 2D histogram: UL MET vs non-UL MET.'''
    arrays = get_masked_array(merged_df, variable, eta_range)
    arr_05Jun20v5_masked = arrays['05Jun20v5']
    arr_UL_masked = arrays['UL']

    # Make a histogram for both cases and plot them both
    binning = {
        'met_pt' : np.arange(150,360,10),
        # 'met_pt' : [ 150, 175, 200, 225, 250,  280,  310,  340,  370,  400, 430, 470, 510, 550, 590, 640],
        'leadak4_pt' : list(range(80,500,20)),
        'trailak4_pt' : list(range(40,400,20)),
    }

    if eta_range is not None:
        low_eta, high_eta = eta_range
    bins = binning[variable]

    fig, ax = plt.subplots()
    if eta_range is None:
        vmin, vmax = 1e-1, 5e2
    else:
        vmin, vmax = 1e-1, 1e2
    h, xedges, yedges, im = ax.hist2d(arr_05Jun20v5_masked, arr_UL_masked, bins=(bins, bins), norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xlabel('MET: 05Jun20v5 (GeV)')
    ax.set_ylabel('MET: UL (GeV)')

    cb = fig.colorbar(im)
    cb.set_label('Number of Events')

    # Set fig title
    if eta_range is None:
        fig_title = r'$\eta$ Inclusive'
    else:
        fig_title = r'One Leading Jet In ${} < |\eta| < {}$'.format(low_eta, high_eta)

    ax.set_title(fig_title)
    
    # Diagonal line plot
    ax.plot(bins, bins, color='red')

    # Save figure
    outdir = f'./output/2d'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if eta_range is not None:
        filename = f'{variable}_etaRange_{str(low_eta).replace(".", "_")}_{str(high_eta).replace(".","_")}.pdf' 
    else:
        filename = f'{variable}_etaRange_inclusive.pdf' 

    outpath = pjoin(outdir, filename)    
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def plot_comparison(merged_df, variable='met_pt', eta_range=(3.0,5.0), apply_met_cut=False):
    '''Given the merged dataframe and the eta range, plot distribution of a variable (MET by default) for UL and non-UL.'''
    arrays = get_masked_array(merged_df, variable, eta_range)
    arr_05Jun20v5_masked = arrays['05Jun20v5']
    arr_UL_masked = arrays['UL']

    # Make a histogram for both cases and plot them both
    binning = {
        'met_pt' : [ 150, 175, 200, 225, 250,  280,  310,  340,  370,  400, 430, 470, 510, 550, 590, 640],
        'leadak4_pt' : list(range(80,500,20)),
        'trailak4_pt' : list(range(40,400,20)),
        'leadak4_eta' : np.arange(-5,5.25,0.25),
        'trailak4_eta' : np.arange(-5,5.25,0.25),
    }

    if eta_range is not None:
        low_eta, high_eta = eta_range
        
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
        'leadak4_eta' : r'Leading Jet $\eta$',
        'trailak4_eta' : r'Trailing Jet $\eta$',
    }

    ax.set_ylabel('Events in Data')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e5)
    ax.legend()

    # Set fig title
    if eta_range is None:
        fig_title = r'$\eta$ Inclusive'
    else:
        fig_title = r'One Leading Jet In ${} < |\eta| < {}$'.format(low_eta, high_eta)

    ax.set_title(fig_title)

    if variable == 'met_pt':
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

    if variable == 'met_pt':
        rax.plot([250., 250.], [0.6, 1.4], color='blue')
    rax.set_ylim(0.6, 1.4)

    loc = matplotlib.ticker.MultipleLocator(base=0.1)
    rax.yaxis.set_major_locator(loc)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.], 'r')
    rax.set_xlim(xlim)

    # Save figure
    if not apply_met_cut:
        outdir = f'./output/'
    else:
        outdir = f'./output/with_met_250_cut'
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
    # Read in the list of variables
    args = parse_cli()
    variables = args.variables
    apply_met_cut = args.applyMetCut

    merged_df = prepare_merged_df(tree_05Jun20v5, tree_UL, branches_to_take, apply_met_cut=apply_met_cut)

    for variable in variables:
        # Plot the MET comparison
        plot_comparison(merged_df, variable=variable, apply_met_cut=apply_met_cut)
        plot_comparison(merged_df, variable=variable, eta_range=(2.5,3.0), apply_met_cut=apply_met_cut)
        plot_comparison(merged_df, variable=variable, eta_range=None, apply_met_cut=apply_met_cut)

        if args.plot2d:
            plot_2d_histogram(merged_df, variable=variable)
            plot_2d_histogram(merged_df, variable=variable, eta_range=(2.5,3.0))
            plot_2d_histogram(merged_df, variable=variable, eta_range=None)

    eta_distribution_for_events_with_high_met_diff(merged_df)
    eta_distribution_for_events_with_high_met_diff(merged_df, met_diff_factor=0.3)
    eta_distribution_for_events_with_high_met_diff(merged_df, met_diff_factor=0.1)

if __name__ == '__main__':
    main()



