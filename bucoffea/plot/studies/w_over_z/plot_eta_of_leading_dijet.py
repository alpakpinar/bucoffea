#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.colors as colors

from coffea import hist
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_eta_of_leading_dijet(acc, outtag, tag, dataset):
    '''Plot 2D eta distribution of the leading dijet (specifically for Z(mm) or W(mn) processes).'''
    distribution = 'ak4_eta0_eta1'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    regions = {
        'EWKW2Jets.*' : 'cr_1m_vbf_nohfveto',
        'EWKZ2Jets.*ZToLL.*' : 'cr_2m_vbf_nohfveto',
    }

    region_to_look = regions[dataset]

    outdir = f'./output/{outtag}/dijet_eta_distributions'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset_tag = r'EWK $W(\mu\nu)$' if dataset == 'EWKW2Jets.*' else r'EWK $Z(\mu\mu)$'

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'{dataset}{year}')).integrate('region', region_to_look)

        fig, ax = plt.subplots()
        hist.plot2d(_h, ax=ax, 
            xaxis='jeteta0',
            patch_opts={'norm' : colors.LogNorm()}
            )

        ax.text(0., 1., dataset_tag,
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

        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)

        ax.axhline(-3, xmin=0, xmax=1, color='k', ls='--')
        ax.axhline(3, xmin=0, xmax=1, color='k', ls='--')

        ax.axvline(-3, ymin=0, ymax=1, color='k', ls='--')
        ax.axvline(3, ymin=0, ymax=1, color='k', ls='--')

        # Save figure
        outpath = pjoin(outdir, f'{tag}_dijet_eta_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    datasets = {
        'ewk_w' : 'EWKW2Jets.*',
        'ewk_z' : 'EWKZ2Jets.*ZToLL.*',
    }
    for tag, dataset in datasets.items():
        plot_eta_of_leading_dijet(acc, outtag, tag, dataset=dataset)

if __name__ == '__main__':
    main()