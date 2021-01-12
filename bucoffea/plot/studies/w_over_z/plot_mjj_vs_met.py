#!/usr/bin/env python

import os
import sys
import re
import argparse
import numpy as np
import matplotlib.colors as colors

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.]),
    # 'met' : hist.Bin('met', r'$p_T^{miss} \ (GeV)$', list(range(0,400,40)) + list(range(400,1000,100))),
    'met' : hist.Bin('met', r'$p_T^{miss} \ (GeV)$', [0,80,200] + list(range(400,900,250))),
}

def plot_mjj_vs_met(acc, outtag):
    'Plot 2D mjj-MET histogram for QCD W(en) process.'
    distribution = 'met_mjj'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get QCD W in 1e CR (without the MET cut)
    h = h.integrate('region', 'cr_1e_vbf_nometcut')

    # Do rebinning
    h = h.rebin('mjj', REBIN['mjj']).rebin('met', REBIN['met'])

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))
        fig, ax = plt.subplots()
        hist.plot2d(_h, ax=ax, 
            xaxis='met',
            patch_opts={
                'norm' : colors.LogNorm()
            }
            )

        ax.text(0., 1., r'QCD $W(e\nu)$',
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

        ax.axvline(80, ymin=0, ymax=1, color='red', lw=2)

        xcenters = _h.axis('met').centers()
        ycenters = _h.axis('mjj').centers()

        sumw = _h.values()[()]

        for iy in range(len(ycenters)):
            met_counts = sumw[:,iy]
            # Calculate the percentage of events with MET < 80 (first bin vs. the rest)
            percentage_with_low_met = met_counts[0] / np.sum(met_counts) * 100

            # Print these values onto the left-most column
            xcenter = xcenters[0]
            color = 'black' if percentage_with_low_met > 15 else 'white'
            ax.text(xcenter, ycenters[iy], f'{percentage_with_low_met:.2f}%',
                fontsize=8,
                ha='center',
                va='center',
                color=color
            )

        # Save figure
        outpath = pjoin(outdir, f'qcd_w_mjj_vs_met_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_mjj_vs_met(acc, outtag)

if __name__ == '__main__':
    main()
