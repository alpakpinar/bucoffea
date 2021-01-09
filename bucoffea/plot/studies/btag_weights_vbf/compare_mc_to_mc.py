#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

def compare_mc_to_mc(acc, outtag, year, distribution='mjj'):
    '''
    Compare the mjj distribution, between the two cases:
    1. b-jet weights are applied
    2. Hard b-jet veto is applied instead
    '''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)

    # Output directory to save plots
    outdir = f'./output/{outtag}/mc_vs_mc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Regions + datasets to look at
    regions_and_datasets = [
        {'region': re.compile('sr_vbf.*'), 'dataset': f'ZJetsToNuNu.*{year}', 'tag': 'qcd_znunu', 'plot_tag' : r'QCD $Z(\nu\nu)$'},
        {'region': re.compile('sr_vbf.*'), 'dataset': f'WJetsToLNu.*{year}', 'tag': 'qcd_wlnu', 'plot_tag' : r'QCD $W(\ell\nu)$'},
        {'region': re.compile('cr_1m_vbf.*'), 'dataset': f'WJetsToLNu.*{year}', 'tag': 'qcd_wmunu', 'plot_tag' : r'QCD $W(\mu\nu)$'},
        {'region': re.compile('cr_1e_vbf.*'), 'dataset': f'WJetsToLNu.*{year}', 'tag': 'qcd_wenu', 'plot_tag' : r'QCD $W(e\nu)$'},
        {'region': re.compile('cr_2m_vbf.*'), 'dataset': f'DYJetsToLL.*{year}', 'tag': 'qcd_zmumu', 'plot_tag' : r'QCD $Z(\mu\mu)$'},
        {'region': re.compile('cr_2e_vbf.*'), 'dataset': f'DYJetsToLL.*{year}', 'tag': 'qcd_zee', 'plot_tag' : r'QCD $Z(ee)$'},
        {'region': re.compile('cr_g_vbf.*'), 'dataset': f'GJets_DR-0p4.*{year}', 'tag': 'qcd_gjets', 'plot_tag' : r'QCD $\gamma$+jets'},
    ]

    for data in regions_and_datasets:
        _h = h.integrate('dataset', data['dataset'])[ data['region'] ]

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='region')

        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e6)

        ax.text(0., 1., data['plot_tag'],
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year,
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'{data["tag"]}_bweight_conp_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        compare_mc_to_mc(acc, outtag, year)

if __name__ == '__main__':
    main()