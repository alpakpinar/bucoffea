#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

def preprocess(h, acc, region):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)
    return h

def plot_prior(acc, outtag, distribution, region='inclusive'):
    '''Plot the given distribution (i.e. HTmiss or genMET) in the given region.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, region)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Plot the distribution for both years
    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'QCD.*{year}')
        fig, ax = plt.subplots()

        hist.plot1d(_h, ax=ax)

        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)

        ax.text(0., 1., region,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'prior_{distribution}_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

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

    for distribution in distributions:
        plot_prior(acc, outtag, distribution=distribution)

if  __name__ == '__main__':
    main()