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
    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))
        fig, ax = plt.subplots()
        hist.plot2d(_h, ax=ax, 
            xaxis='met',
            patch_opts={
                'norm' : colors.LogNorm()
            }
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
