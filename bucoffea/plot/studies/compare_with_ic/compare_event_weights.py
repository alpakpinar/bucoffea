#!/usr/bin/env python
import uproot
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

def get_input_files(year):
    inputdir  = './inputs/for_weights'
    bu_file = pjoin(inputdir, f'tree_wjets_bu_{year}.root')
    ic_file = pjoin(inputdir, f'tree_wjets_ic_{year}.root')

    return bu_file, ic_file

def get_merged_df(bu_file, ic_file, year):
    columns_bu = ['run', 'lumi', 'event', 'weight_theory_qcd', 'weight_theory_ewk', 'weight_trigger_met', 'weight_pileup', 'mjj']
    columns_ic = ['run', 'luminosityBlock', 'event', 'fnlo_SF_QCD_corr_QCD_proc_MTR', 'fnlo_SF_EWK_corr', f'trigger_weight_METMHT{year}', 'puWeight', 'diCleanJet_M']
    df_bu = uproot.open(bu_file)['sr_vbf_no_veto_all'].pandas.df()[columns_bu]
    df_ic = uproot.open(ic_file)['Events'].pandas.df()[columns_ic]

    df_ic.rename(
        columns={
            'fnlo_SF_QCD_corr_QCD_proc_MTR' : 'weight_theory_qcd',
            'fnlo_SF_EWK_corr' : 'weight_theory_ewk',
            f'trigger_weight_METMHT{year}' : 'weight_trigger_met',
            'puWeight' : 'weight_pileup',
            'diCleanJet_M' : 'mjj',
            'luminosityBlock' : 'lumi'
            },
        inplace=True
    )

    merged_df = pd.merge(df_bu, df_ic, on=['run', 'lumi', 'event'], suffixes=['_bu', '_ic'])
    
    return merged_df

def check_mjj(merged_df, year):
    bu_mjj = merged_df['mjj_bu']
    ic_mjj = merged_df['mjj_ic']

    diff = (bu_mjj - ic_mjj) / bu_mjj
    fig, ax = plt.subplots()
    ax.hist(diff, bins=np.linspace(-0.1,0.1,20))
    # ax.hist(diff, bins=np.linspace(-0.1,0.1,20))

    ax.set_xlabel(r'% difference in $M_{jj}$')

    # Save figure
    outdir = f'./output/compare_weights'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'mjj_check_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def compare_weights(merged_df, year):
    # QCD weights
    bu_qcd_w = merged_df['weight_theory_qcd_bu']
    ic_qcd_w = merged_df['weight_theory_qcd_ic']
    qcd_diff = (bu_qcd_w - ic_qcd_w) / bu_qcd_w

    # EWK weights
    bu_ewk_w = merged_df['weight_theory_ewk_bu']
    ic_ewk_w = merged_df['weight_theory_ewk_ic']
    ewk_diff = (bu_ewk_w - ic_ewk_w) / bu_ewk_w

    # MET trigger weights
    bu_mettrig_w = merged_df['weight_trigger_met_bu']
    ic_mettrig_w = merged_df['weight_trigger_met_ic']
    trig_diff = (bu_mettrig_w - ic_mettrig_w) / bu_mettrig_w
    
    # Pileup weights
    bu_pileup_w = merged_df['weight_pileup_bu']
    ic_pileup_w = merged_df['weight_pileup_ic']
    pu_diff = (bu_pileup_w - ic_pileup_w) / bu_pileup_w

    bins = np.linspace(-0.2,0.2)

    fig, ax = plt.subplots(2,2,figsize=(12,8))
    ax[0,0].hist(qcd_diff, bins=bins)
    ax[0,0].set_title('QCD NLO')
    ax[0,0].set_xlabel('(BU-IC)/BU')

    ax[0,1].hist(ewk_diff, bins=bins)
    ax[0,1].set_title('EWK NLO')
    ax[0,1].set_xlabel('(BU-IC)/BU')

    ax[1,0].hist(trig_diff, bins=bins)
    ax[1,0].set_title('MET trig weight')
    ax[1,0].set_xlabel('(BU-IC)/BU')

    ax[1,1].hist(pu_diff, bins=bins)
    ax[1,1].set_title('PU weight')
    ax[1,1].set_xlabel('(BU-IC)/BU')

    plt.subplots_adjust(
        hspace=0.3
    )

    # Save figure
    outdir = f'./output/compare_weights'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'weight_comp_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    for year in [2017, 2018]:
        bu_file, ic_file = get_input_files(year)
        merged_df = get_merged_df(bu_file, ic_file, year)
        # Test the merged df
        check_mjj(merged_df, year)
        compare_weights(merged_df, year)

if __name__ == '__main__':
    main()
