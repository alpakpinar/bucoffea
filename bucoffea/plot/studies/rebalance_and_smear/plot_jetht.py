#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_jetht(acc, outtag, distribution='htmiss_ht'):
    '''Plot given distribution from JetHT dataset, for events passing the jet trigger.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    # Rebin HT for the 2D histogram
    if distribution == 'htmiss_ht':
        new_ht_ax = hist.Bin("ht", r"$H_{T}$ (GeV)", list(range(100,900,200)) + [900,1300,2000,5000])
        h = h.rebin('ht', new_ht_ax)

    # Rebin HTmiss
    if 'htmiss' in distribution:
        new_htmiss_ax = hist.Bin("htmiss", r"$H_{T}^{miss}$ (GeV)", list(range(0,400,20)) + list(range(400,900,100)))
        h = h.rebin('htmiss', new_htmiss_ax)

    # Get the events passing the jet trigger
    h = h.integrate('region', 'trig_pass')

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'JetHT_{year}')
        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='ht')

        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)

        ax.text(0., 1., f'JetHT {year}',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., f'Passing HLT_PFJet40',
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'jetht_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def plot_high_htmiss_events(acc, outtag, distribution='ak4_eta', region='high_htmiss_loose'):
    '''Plot events from a high HTmiss region.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    h = h.integrate('region', region)

    outdir = f'./output/{outtag}/high_htmiss_plots'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    htmisstag = {
        'high_htmiss_loose' : r'$H_T^{miss} > 100 \ GeV$', 
        'high_htmiss_tight' : r'$H_T^{miss} > 200 \ GeV$', 
    }

    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'JetHT_{year}')
        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax)

        if distribution == 'ak4_pt':
            ax.set_yscale('log')
            ax.set_ylim(1e0,1e6)
        
        ax.get_legend().remove()

        ax.text(0., 1., htmisstag[region],
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year,
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(0.98, 0.9, 'Events passing HLT_PFJet40',
            fontsize=12,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'{region}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')    

    plot_jetht(acc, outtag)

    for region in ['high_htmiss_loose', 'high_htmiss_tight']:
        for distribution in ['ak4_pt', 'ak4_eta', 'ak4_phi']:
            plot_high_htmiss_events(acc, outtag, distribution, region)

if __name__ == '__main__':
    main()