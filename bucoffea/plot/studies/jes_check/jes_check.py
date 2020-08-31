#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

def get_ak4_pt0_variations(trial_file, trial_mode, num_events=10000, thresh=1e-3):
    '''Check jet pt JES variations, returns True if all num_events have total JES unc close to individual ones combined.'''
    f = uproot.open(trial_file)['Events']
    # Get dataframe with T1 MET and all its variations 
    df = f.pandas.df(['Jet_pt*', 'Jet_phi*'])[:num_events]

    jet_pt_nom = df['Jet_pt_nom']
    jet_phi = df['Jet_phi']

    jet_px_nom = jet_pt_nom * np.cos(jet_phi)
    jet_py_nom = jet_pt_nom * np.sin(jet_phi)

    # Get the list of JES variations
    up_variations = [var for var in df.columns if 'Up' in var and 'jer' not in var]
    down_variations = [var for var in df.columns if 'Down' in var and 'jer' not in var]

    # Get x and y components of the momentum
    get_x_comp = lambda pt: pt * np.cos(jet_phi)
    get_y_comp = lambda pt: pt * np.sin(jet_phi)

    jet_px_up = df[up_variations].apply(get_x_comp, axis=0)
    jet_py_up = df[up_variations].apply(get_y_comp, axis=0)
    jet_px_down = df[down_variations].apply(get_x_comp, axis=0)
    jet_py_down = df[down_variations].apply(get_y_comp, axis=0)

    # Get the % up and down variations from nominal MET pt value
    px_up_ratios = np.abs(jet_px_up.divide(jet_px_nom, axis=0) - 1)
    py_up_ratios = np.abs(jet_py_up.divide(jet_py_nom, axis=0) - 1)
    px_down_ratios = np.abs(jet_px_down.divide(jet_px_nom, axis=0) - 1)
    py_down_ratios = np.abs(jet_py_down.divide(jet_py_nom, axis=0) - 1)

    # Function for summing the individual unc sources in quadrature 
    get_quad_sum = lambda row: np.sqrt( sum(map(lambda x: x*x, row[:-1])) )
    combined_unc_px_up = px_up_ratios.apply(get_quad_sum, axis=1) 
    combined_unc_py_up = py_up_ratios.apply(get_quad_sum, axis=1) 
    combined_unc_px_down = px_down_ratios.apply(get_quad_sum, axis=1) 
    combined_unc_py_down = py_down_ratios.apply(get_quad_sum, axis=1) 

    # Store a histogram: (Combined / Total) - 1, for each variation
    combined_over_total_px_up = (combined_unc_px_up / px_up_ratios['Jet_pt_jesTotalUp']) - 1 
    combined_over_total_py_up = (combined_unc_py_up / py_up_ratios['Jet_pt_jesTotalUp']) - 1 
    combined_over_total_px_down = (combined_unc_px_down / px_down_ratios['Jet_pt_jesTotalDown']) - 1 
    combined_over_total_py_down = (combined_unc_py_down / py_down_ratios['Jet_pt_jesTotalDown']) - 1 

    fig, axes = plt.subplots(2,2,figsize=(12,8))
    bins = np.linspace(-0.1,0.1)

    axes[0,0].hist(combined_over_total_px_up, bins=bins)
    axes[0,0].set_title(r'$p_x$ JES up')
    axes[0,0].set_ylabel('Counts')

    axes[0,1].hist(combined_over_total_py_up, bins=bins)
    axes[0,1].set_title(r'$p_y$ JES up')
    axes[0,1].set_ylabel('Counts')

    axes[1,0].hist(combined_over_total_px_down, bins=bins)
    axes[1,0].set_title(r'$p_x$ JES down')
    axes[1,0].set_ylabel('Counts')

    axes[1,1].hist(combined_over_total_py_down, bins=bins)
    axes[1,1].set_title(r'$p_y$ JES down')
    axes[1,1].set_ylabel('Counts')

    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{trial_mode}_jet_pt_combined_over_total.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

    # Get diffs with Total up/down uncertainties
    diff_px_up = np.abs(combined_unc_px_up - px_up_ratios['Jet_pt_jesTotalUp']) 
    diff_py_up = np.abs(combined_unc_py_up - py_up_ratios['Jet_pt_jesTotalUp'])
    diff_px_down = np.abs(combined_unc_px_down - px_down_ratios['Jet_pt_jesTotalDown'])
    diff_py_down = np.abs(combined_unc_py_down - py_down_ratios['Jet_pt_jesTotalDown'])

    mask = (diff_px_up < thresh).all() & (diff_py_up < thresh).all() & \
        (diff_py_up < thresh).all() & (diff_py_down < thresh).all()

    # Maximum difference
    max_diff = max(diff_px_up.max(), diff_px_down.max(), diff_py_up.max(), diff_py_down.max())

    return mask, max_diff

def get_met_variations(trial_file, num_events=10):
    '''Get all MET variations.'''
    f = uproot.open(trial_file)['Events']
    # Get dataframe with T1 MET and all its variations (first 10 events)
    df = f.pandas.df('MET*T1_*pt*')[:num_events]

    # Get the list of JES variations
    up_variations = [var for var in df.columns if 'Up' in var and 'jer' not in var]
    down_variations = [var for var in df.columns if 'Down' in var and 'jer' not in var]
    met_pt_nom = df['METFixEE2017_T1_pt']

    # Get the % up and down variations from nominal MET pt value
    up_ratios = df[up_variations].divide(met_pt_nom, axis=0) - 1
    down_ratios = df[down_variations].divide(met_pt_nom, axis=0) - 1

    print(up_ratios.iloc[0])
    # Add the individual sources in quadrature  
    up_ratios_ind = up_ratios[up_ratios.columns[:-1]]
    down_ratios_ind = down_ratios[down_ratios.columns[:-1]]
    print(up_ratios_ind.iloc[0])

def main():
    trial_mode = sys.argv[1]
    trial_files = {
        'DY'    : 'root://cmsxrootd.fnal.gov//store/user/aakpinar/nanopost/18Jun20_splitJEC/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_HT-100to200-MLM_ext1_2017/200620_082557/0000/tree_1.root',
        'WJets' : 'root://cmsxrootd.fnal.gov//store/user/aakpinar/nanopost/18Jun20_splitJEC/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400-MLM_2017/200618_122609/0000/tree_1.root',
        'ZJets' : 'root://cmsxrootd.fnal.gov//store/user/aakpinar/nanopost/18Jun20_splitJEC/ZJetsToNuNu_HT-200To400_13TeV-madgraph/ZJetsToNuNu_HT-200To400-mg_2017/200618_121835/0000/tree_1.root',
    }
    trial_file = trial_files[trial_mode]

    print('Checking leading jet pt...')
    mask, max_diff = get_ak4_pt0_variations(trial_file, trial_mode)
    if mask:
        print('Combined/total JES uncs agree!')
        print(f'Maximum difference: {max_diff:.5f}')
    else:
        print('Combined/total JES uncs do not agree within tolerance!')
        print(f'Maximum difference: {max_diff:.5f}')

if __name__ == '__main__':
    main()
