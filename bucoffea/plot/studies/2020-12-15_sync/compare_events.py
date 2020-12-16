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
    'leadak4_pt',
    'leadak4_eta',
    'trailak4_pt',
    'trailak4_eta',
    'mjj',
]

columns_ic = {
    'event' : 'event',
    'Leading_jet_pt' : 'leadak4_pt',
    'Leading_jet_eta' : 'leadak4_eta',
    'Subleading_jet_pt' : 'trailak4_pt',
    'Subleading_jet_eta' : 'trailak4_eta',
    'diCleanJet_M' : 'mjj',
}

def get_merged_df(bu_file, ic_file):
    df_bu = bu_file['cr_1m_vbf'].pandas.df()[columns_bu]
    df_ic = ic_file['Events'].pandas.df()[columns_ic.keys()]

    # Rename IC columns so that they are compatible with BU
    df_ic.rename(
        columns=columns_ic,
        inplace=True
    )

    merged_df = pd.merge(df_bu, df_ic, on='event', suffixes=['_bu', '_ic'])

    return merged_df

def compare_jet_pt(merged_df, jobtag):
    '''Compare BU and IC jet pts.'''
    leading_jet_pts = {
        'BU' : merged_df['leadak4_pt_bu'],
        'IC' : merged_df['leadak4_pt_ic'],
    } 
    trailing_jet_pts = {
        'BU' : merged_df['trailak4_pt_bu'],
        'IC' : merged_df['trailak4_pt_ic'],
    } 

    leading_jet_pt_diff = (leading_jet_pts['BU'] - leading_jet_pts['IC']) / leading_jet_pts['BU']
    trailing_jet_pt_diff = (trailing_jet_pts['BU'] - trailing_jet_pts['IC']) / trailing_jet_pts['BU']

    # Plot the % differences
    fig, ax = plt.subplots()
    bins = np.linspace(-0.3,0.3)
    ax.hist(leading_jet_pt_diff, bins=bins, label=r'Leading jet $p_T$', histtype='step')
    ax.hist(trailing_jet_pt_diff, bins=bins, label=r'Trailing jet $p_T$', histtype='step')

    ax.set_xlabel('(BU-IC) / BU')
    ax.set_ylabel('Counts')
    ax.legend()

    # Save figure
    outdir = f'./output/{jobtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'jet_pt_comp.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = './trees/16Dec20_qcdW'
    jobtag = os.path.basename(inpath)

    bu_file = uproot.open( 
        pjoin(inpath, 'tree_WJetsToLNu_HT-600To800_2017.root')
    )
    ic_file = uproot.open( 
        pjoin(inpath, 'Skim_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.root')
    )

    merged_df = get_merged_df(bu_file, ic_file)
    compare_jet_pt(merged_df, jobtag)

if __name__ == '__main__':
    main()