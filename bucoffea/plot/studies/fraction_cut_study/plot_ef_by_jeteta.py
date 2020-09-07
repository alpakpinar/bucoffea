#!/usr/bin/env python

import os
import sys
import re
import warnings
from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

# ================================
# Plot neutral EM energy fractions in data and MC
# categorized by the jet eta 
# ================================

def plot_ef_by_jeteta(acc, outtag, year):
    '''Plot neutral EM energy fraction in data and MC, categorized by jet eta'''
    acc.load('ak4_nef0')
    h = acc['ak4_nef0']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    regions = [
        'cr_2m_noEmEF_jeteta_lt_2_3',
        'cr_2m_noEmEF_jeteta_jet_eta_gt_2_3_lt_2_7',
        'cr_2m_noEmEF_jeteta_jet_eta_gt_2_7_lt_3_0',
        'cr_2m_noEmEF_jeteta_gt_3_0'
    ]

    for region in regions:
        region_tag = re.findall('jeteta_.*', region)[0]
        print(f'Region tag: {region_tag}')
        histo = h[re.compile(f'.*{region_tag}.*')]

        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        hist.plot1d(histo, ax=ax, overlay='dataset')
        ax.set_xlabel('')
        ax.set_yscale('log')
        ax.set_ylim(1e-1,1e6)

        # Plot data/MC ratio
        h_data = histo.integrate('dataset', f'MET_{year}')
        h_mc = histo.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))
        hist.plotratio(h_data, h_mc, unc='num', ax=rax)
        
        rax.set_xlabel('Jet neutral EM fraction')
        rax.grid(True)
        rax.set_ylim(0.5,1.5)

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outpath = pjoin(outdir, f'{region_tag}_data_mc.pdf')
        fig.savefig(outpath)

        print(f'File saved: {outpath}')
        plt.close(fig)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('^merged_.*', inpath)[0].replace('/','')

    plot_ef_by_jeteta(acc, outtag, year=2017)

if __name__ == '__main__':
    main()