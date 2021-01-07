#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_dpfcalo_dist(acc, outtag):
    '''Plot dPfCalo distribution for EWK Z(mumu) and EWK W(munu).'''
    distribution = 'dpfcalo_cr'
    acc.load(distribution)
    h = acc[distribution]

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Pick the regions without the dpfcalo cut
    h_2m = h.integrate('region', 'cr_2m_vbf_nodpfcalo')
    h_1m = h.integrate('region', 'cr_1m_vbf_nodpfcalo')

    for year in [2017, 2018]:
        _h_2m = h_2m.integrate('dataset', re.compile(f'EWKZ.*ZToLL.*{year}'))
        _h_1m = h_1m.integrate('dataset', re.compile(f'EWKW.*{year}'))
        fig, ax = plt.subplots()
        hist.plot1d(_h_2m, ax=ax)
        hist.plot1d(_h_1m, ax=ax, clear=False)

        ax.legend(labels=[
            r'EWK $Z(\mu\mu)$',
            r'EWK $W(\mu\nu)$',
        ])

        ax.set_yscale('log')
        ax.set_ylim(1e-1, 1e5)

        ax.text(1., 1., year,
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.axvline(-0.5, ymin=0, ymax=1, color='black')
        ax.axvline(0.5, ymin=0, ymax=1, color='black')

        # Save figure
        outpath = pjoin(outdir, f'ewk_z_w_dpfcalo_dist_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_dpfcalo_dist(acc, outtag)

if __name__ == '__main__':
    main()