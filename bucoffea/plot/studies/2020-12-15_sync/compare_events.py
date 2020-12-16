#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pjoin = os.path.join

columns_bu = [
    'event',
    'run',
    'lumi',
    'leadak4_pt',
    'leadak4_pt_nosmear',
    'leadak4_pt_raw',
    'leadak4_eta',
    'trailak4_pt',
    'trailak4_pt_nosmear',
    'trailak4_pt_raw',
    'trailak4_eta',
    'mjj',
]

columns_ic = {
    'event' : 'event',
    'run' : 'run',
    'luminosityBlock' : 'lumi',
    'Leading_jet_pt' : 'leadak4_pt',
    'Leading_jet_eta' : 'leadak4_eta',
    'Subleading_jet_pt' : 'trailak4_pt',
    'Subleading_jet_eta' : 'trailak4_eta',
    'diCleanJet_M' : 'mjj',
}

def get_merged_df(bu_file, ic_file, region):
    bu_regions = {
        'singlemu'  : 'cr_1m_vbf',
        'inclusive' : 'cr_sync_vbf'
    }
    try:
        bu_region = bu_regions[region]
    except KeyError:
        raise ValueError(f'Invalid region: {region}')

    df_bu = bu_file[bu_region].pandas.df()[columns_bu]
    df_ic = ic_file['Events'].pandas.df()[columns_ic.keys()]

    # Rename IC columns so that they are compatible with BU
    df_ic.rename(
        columns=columns_ic,
        inplace=True
    )

    merged_df = pd.merge(df_bu, df_ic, on=['event', 'run', 'lumi'], suffixes=['_bu', '_ic'])

    return merged_df

def plot_jet_pt_dist(merged_df, jobtag, region, jet='trailak4'):
    '''Plot distribution of jet pts.'''
    jet_pt_bu = merged_df[f'{jet}_pt_bu']
    jet_pt_ic = merged_df[f'{jet}_pt_ic']

    fig, ax = plt.subplots()
    bins = np.arange(40,800,20)

    ax.hist(jet_pt_bu, bins=bins, label='BU', histtype='step')
    ax.hist(jet_pt_ic, bins=bins, label='IC', histtype='step')

    if jet == 'leadak4':
        xlabel = r'Leading Jet $p_T$ (GeV)'
    else:
        xlabel = r'Trailing Jet $p_T$ (GeV)'
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)

    outdir = f'./output/{jobtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{jet}_pt_comparison_{region}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def compare_jet_pt(merged_df, jobtag, region):
    '''Compare BU and IC jet pts.'''
    leading_jet_pts = {
        'BU' : merged_df['leadak4_pt_bu'],
        'BU unsmeared' : merged_df['leadak4_pt_nosmear'],
        'IC' : merged_df['leadak4_pt_ic'],
    } 
    trailing_jet_pts = {
        'BU' : merged_df['trailak4_pt_bu'],
        'BU unsmeared' : merged_df['trailak4_pt_nosmear'],
        'IC' : merged_df['trailak4_pt_ic'],
    } 

    leading_jet_pt_diff = (leading_jet_pts['BU'] - leading_jet_pts['IC']) / leading_jet_pts['BU']
    trailing_jet_pt_diff = (trailing_jet_pts['BU'] - trailing_jet_pts['IC']) / trailing_jet_pts['BU']

    leading_jet_pt_unsmeared_diff = (leading_jet_pts['BU unsmeared'] - leading_jet_pts['IC']) / leading_jet_pts['BU unsmeared']
    trailing_jet_pt_unsmeared_diff = (trailing_jet_pts['BU unsmeared'] - trailing_jet_pts['IC']) / trailing_jet_pts['BU unsmeared']

    merged_df['leading_jet_pt_diff'] = leading_jet_pt_diff
    merged_df['trailing_jet_pt_diff'] = trailing_jet_pt_diff

    # Plot the % differences
    fig, ax = plt.subplots()
    bins = np.linspace(-0.3,0.3)
    ax.hist(leading_jet_pt_diff, bins=bins, label=r'Leading jet $p_T$', histtype='step')
    ax.hist(trailing_jet_pt_diff, bins=bins, label=r'Trailing jet $p_T$', histtype='step')

    ax.hist(leading_jet_pt_unsmeared_diff, bins=bins, label=r'Leading jet $p_T$: BU non-smeared', histtype='step')
    ax.hist(trailing_jet_pt_unsmeared_diff, bins=bins, label=r'Trailing jet $p_T$: BU non-smeared', histtype='step')

    ax.set_xlabel('(BU-IC) / BU')
    ax.set_ylabel('Counts')
    ax.legend()

    title = r'$\approx$Inclusive QCD W' if region == 'inclusive' else r'QCD W Events Passing $1\mu$ CR'
    ax.set_title(title, fontsize=14)

    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e8)

    # Save figure
    outdir = f'./output/{jobtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'jet_pt_comp_{region}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

    # Return the updated merged df
    return merged_df

def main():
    # Region: 1m CR or inclusive?
    region = sys.argv[1]
    inpath = './trees/16Dec20_qcdW'
    jobtag = os.path.basename(inpath)

    bu_file = uproot.open( 
        pjoin(inpath, 'tree_WJetsToLNu_HT-600To800_2017.root')
    )
    ic_filetag = '_inclusive' if region == 'inclusive' else ''

    ic_file = uproot.open( 
        pjoin(inpath, f'Skim_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8{ic_filetag}.root')
    )

    merged_df = get_merged_df(bu_file, ic_file, region)
    merged_df = compare_jet_pt(merged_df, jobtag, region)

    for jet in ['leadak4', 'trailak4']:
        plot_jet_pt_dist(merged_df, jobtag, region, jet=jet)

if __name__ == '__main__':
    main()