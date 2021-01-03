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

def rebin(h, distribution):
    mapping = {
        'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    }
    # Rebin if neccessary
    if distribution in mapping.keys():
        h = h.rebin(mapping[distribution].name, mapping[distribution])
    
    return h

def plot_vjets_vbf_comparison(acc, distribution='mjj'):
    '''Plot a comparison of V+jets (QCD and EWK) and signal (VBF and ggH) spectra, in terms of the given distribution.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin(h, distribution)
    # Take the region with no mjj/detajj/dphijj cut applied 
    # h = h.integrate('region', 'sr_vbf_nodijetcut')
    h = h.integrate('region', 'sr_vbf')

    # Output directory to save plots
    outdir = f'./output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        # Check the regex!
        h_qcdz = h.integrate('dataset', re.compile(f'ZJetsToNuNu.*{year}'))
        h_qcdw = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))
        h_ewkz = h.integrate('dataset', re.compile(f'EWKZ2Jets.*{year}'))
        h_ewkw = h.integrate('dataset', re.compile(f'EWKW2Jets.*{year}'))

        h_vbf = h.integrate('dataset', re.compile(f'VBF_HToInv.*M125.*{year}'))
        h_ggh = h.integrate('dataset', re.compile(f'GluGlu_HToInv.*M125.*HiggspTgt190.*{year}'))

        # Combined QCD and EWK V+jets yields
        h_qcdz.add(h_qcdw)
        h_ewkz.add(h_ewkw)

        # Now, plot 'em all!
        fig, ax = plt.subplots()

        hist.plot1d(h_qcdz, ax=ax, density=True)
        hist.plot1d(h_ewkz, ax=ax, clear=False, density=True)
        hist.plot1d(h_vbf, ax=ax, clear=False, density=True)
        hist.plot1d(h_ggh, ax=ax, clear=False, density=True)

        ylims = {
            'mjj' : (1e-9, 1e1), 
            'detajj' : (1e-6, 1e1), 
            'dphijj' : (1e-1, 1e1), 
        }

        ax.set_yscale('log')
        ax.set_ylim(ylims[distribution])
        ax.set_ylabel('Normalized Counts')

        ax.text(1., 1., year,
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Fix legend labels
        labels = [
            'QCD V+jets',
            'EWK V+jets',
            'VBF H(inv)',
            'ggH(inv)',
        ]

        ax.legend(title='Process', labels=labels)

        # Save figure
        outpath = pjoin(outdir, f'signal_bkg_comparison_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    for distribution in ['mjj', 'detajj', 'dphijj']:
        plot_vjets_vbf_comparison(acc, distribution=distribution)

if __name__ == '__main__':
    main()
