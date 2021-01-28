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

mapping = {
    'ak4_pt0' : r'Leading jet $p_T$ (GeV)',
    'ak4_eta0': r'Leading jet $\eta$',
    'ak4_phi0': r'Leading jet $\phi$',
    'ak4_pt' : r'Jet $p_T$ (GeV)',
    'ak4_eta': r'Jet $\eta$',
    'ak4_phi': r'Jet $\phi$',
}

def plot_htmiss_in_ht_bins(acc, outtag):
    '''Plot given distribution from JetHT dataset, for events passing the jet trigger.'''
    distribution='htmiss_ht'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    # Rebin HT and HTmiss
    new_ht_ax = hist.Bin("ht", r"$H_{T}$ (GeV)", list(range(100,900,200)) + [900,1300,2000,5000])
    h = h.rebin('ht', new_ht_ax)

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

def plot_distribution_from_region(acc, outtag, distribution='ak4_eta', region='high_htmiss_loose'):
    '''Plot events from a specified region.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    h = h.integrate('region', region)

    outdir = f'./output/{outtag}/distributions'
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

        if distribution in ['ak4_pt', 'ak4_pt0', 'ht', 'htmiss']:
            ax.set_yscale('log')
            ax.set_ylim(1e0,1e6)
        
        if distribution == 'htmiss':
            ax.set_xlim(0,400)

        ax.get_legend().remove()

        if distribution in mapping.keys():
            ax.set_xlabel(mapping[distribution])

        if region in htmisstag.keys():
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

    plot_htmiss_in_ht_bins(acc, outtag)

    distributions = [
        'ak4_pt',
        'ak4_eta',
        'ak4_phi',
        'ak4_pt0',
        'ak4_eta0',
        'ak4_phi0',
        'ak4_mult',
        'ht',
        'htmiss'
    ]

    for region in ['trig_pass', 'high_htmiss_loose', 'high_htmiss_tight']:
        for distribution in distributions:
            try:
                plot_distribution_from_region(acc, outtag, distribution, region)
            except KeyError:
                print(f'Could not find distribution: {distribution}, skipping.')
                continue

if __name__ == '__main__':
    main()