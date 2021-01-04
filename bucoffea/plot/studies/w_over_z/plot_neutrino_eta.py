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

def plot_neutrino_eta(acc, outtag, dataset='WJetsToLNu', region='cr_1m_vbf'):
    '''Plot neutrino eta distribution.'''
    acc.load('gen_nu_eta')
    h = acc['gen_nu_eta']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    outdir = f'./output/{outtag}/neutrino_eta'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'{dataset}.*{year}'))

        # Histograms with and without the eta cut on the GEN-neutrino
        h_nocut = _h.integrate('region', region)
        h_withcut = _h.integrate('region', f'{region}_central_nu')

        fig, ax = plt.subplots()
        hist.plot1d(h_nocut, ax=ax)
        hist.plot1d(h_withcut, ax=ax, clear=False)

        labels = [r'$\nu$: $\eta$ Inclusive', r'$\nu$: $|\eta|<2.5$']
        ax.legend(labels=labels)

        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e5)
        ax.set_title(r'GEN-$\nu$ $\eta$ Distribution (From W)', fontsize=14)
        if dataset == 'WJetsToLNu':
            proclabel = 'QCD'
        else:
            proclabel = 'EWK'

        ax.text(0., 1., proclabel,
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

        # Save figure
        outpath = pjoin(outdir, f'{dataset}_neutrino_eta_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # QCD and EWK datasets
    datasets = ['WJetsToLNu', 'EWKW2Jets']

    for dataset in datasets:
        plot_neutrino_eta(acc, outtag, dataset=dataset)

if __name__ == '__main__':
    main()