#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep
import matplotlib
# Use a different backend for matplotlib, otherwise HEP styling doesn't work for some reason
matplotlib.use('tkagg')

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

# Use default CMS styling
plt.style.use(hep.style.CMS)

def rebin(h, distribution):
    mapping = {
        'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]),
        'detajj' : hist.Bin("deta", r"$\Delta\eta_{jj}$", 25, 0, 10),
        'dphijj' : hist.Bin("dphi", r"$\Delta\phi_{jj}$", 25, 0, 3.5),
    }
    # Rebin if neccessary
    if distribution in mapping.keys():
        h = h.rebin(mapping[distribution].name, mapping[distribution])
    
    return h

def plot_vjets_vbf_comparison(acc, outtag, distribution='mjj', file_format='pdf'):
    '''Plot a comparison of V+jets (QCD and EWK) and signal (VBF and ggH) spectra, in terms of the given distribution.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin(h, distribution)
    # Take the region with no mjj/detajj/dphijj cut applied 
    if distribution != 'ak4_eta1':
        h = h.integrate('region', 'sr_vbf_nodijetcut')
    else:
        h = h.integrate('region', 'sr_vbf')

    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        h_qcdz = h.integrate('dataset', re.compile(f'(ZJetsToNuNu|DYJetsToLL).*{year}'))
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

        line_opts = {
            'linewidth' : 2
        }

        hist.plot1d(h_qcdz, ax=ax, density=True, line_opts=line_opts)
        hist.plot1d(h_ewkz, ax=ax, clear=False, density=True, line_opts=line_opts)
        hist.plot1d(h_vbf, ax=ax, clear=False, density=True, line_opts=line_opts)
        hist.plot1d(h_ggh, ax=ax, clear=False, density=True, line_opts=line_opts)

        ylims = {
            'mjj' : (1e-7, 1e-1), 
            'detajj' : (1e-5, 1e1), 
            'dphijj' : (1e-3, 1e1), 
            'ak4_eta1' : (0, 0.4), 
        }

        if distribution == 'mjj':
            ax.set_yscale('log')
            ax.set_ylim(ylims[distribution])
        
        elif distribution == 'ak4_eta1':
            ax.set_ylim(ylims[distribution])

        ax.set_ylabel('Normalized Counts')

        if distribution == 'ak4_eta1':
            ax.set_xlabel(r'Trailing jet $\eta$')

        # CMS label & text
        hep.cms.label(year="", paper=True)
        hep.cms.text()

        # Fix legend labels
        labels = [
            'QCD V+jets',
            'EWK V+jets',
            'VBF H(inv)',
            'ggH(inv)',
        ]

        ax.legend(labels=labels)

        # Save figure
        outpath = pjoin(outdir, f'signal_bkg_comparison_{distribution}_{year}.{file_format}')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for distribution in ['mjj', 'detajj', 'dphijj', 'ak4_eta1']:
        plot_vjets_vbf_comparison(acc, outtag, distribution=distribution)

if __name__ == '__main__':
    main()
