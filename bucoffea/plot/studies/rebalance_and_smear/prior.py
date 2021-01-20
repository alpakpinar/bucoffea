#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def preprocess(h, acc, region):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)
    return h

def get_plotlabel(region):
    labels = {
        'inclusive' : 'Inclusive',
        'trk_trk' : 'Trk-Trk Events',
        'not_trk_trk' : 'Non Trk-Trk Events',
    }
    return labels[region]

def plot_prior(acc, outtag, distribution, region='inclusive'):
    '''Plot the given distribution (i.e. HTmiss or genMET) in the given region.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, region)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Plot the distribution for both years
    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'QCD_HT_{year}')
        fig, ax = plt.subplots()

        hist.plot1d(_h, ax=ax)

        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e10)
        ax.set_xlim(0,2000)

        ax.get_legend().remove()

        ax.text(0., 1., get_plotlabel(region),
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

        # Save figure
        outpath = pjoin(outdir, f'prior_{distribution}_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def plot_2d_prior(acc, outtag, distribution, outputrootfile=None):
    acc.load(distribution)
    # For 2D distributions, we look at inclusive region by default
    h = preprocess(acc[distribution], acc, region='inclusive')

    outdir = f'./output/{outtag}/2d'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Rebin HT axis
    ht_new_binning = list(range(0,1000,200)) + list(range(1000,4000,1000)) + [5000]
    new_ht_ax = hist.Bin("ht", r"$H_{T}$ (GeV)", ht_new_binning)
    h = h.rebin('ht', new_ht_ax)

    # Plot the distribution for both years
    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'QCD_HT_{year}')
        fig, ax = plt.subplots()

        # Plot HTmiss in bins of HT
        hist.plot1d(_h, overlay='ht', ax=ax)
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e14)

        ax.text(0., 1., 'QCD MC',
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

        # Save figure
        outpath = pjoin(outdir, f'prior_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

        # If an output ROOT file is given, save the histograms into this file
        if outputrootfile is not None:
            ht_bins = _h.identifiers('ht')
            for ht_bin in ht_bins:
                lo_int, high_int = int(ht_bin.lo), int(ht_bin.hi)
                bin_label = f'ht_{lo_int}_to_{high_int}'

                # Get the values for this ht bin
                h_int = _h.integrate('ht', ht_bin)
                dist_label = f'{distribution.replace("ht_", "gen_")}_{bin_label}_{year}'
                outputrootfile[dist_label] = (h_int.values()[()], h_int.axes()[0].edges())

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    distributions = [
        'gen_htmiss',
        'genmet_pt',
    ]

    regions = [
        'inclusive',
        'trk_trk',
        'not_trk_trk'
    ]

    # Create output ROOT file to save the distributions
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outputrootpath = pjoin(outdir, f'rbs_prior_dists.root')
    outputrootfile = uproot.recreate(outputrootpath)

    # 1D priors
    for region in regions:
        for distribution in distributions:
            plot_prior(acc, outtag, distribution=distribution, region=region)

    # 2D priors: Plot HTmiss in bins of HT
    plot_2d_prior(acc, outtag, distribution='htmiss_ht', outputrootfile=outputrootfile)


if  __name__ == '__main__':
    main()