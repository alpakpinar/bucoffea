#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt

pjoin = os.path.join

def plot_mjj_by_leadingjetpt(acc, outtag, pt_slices, year):
    '''In bins of leading jet pt, plot mjj distribution for the total background in SR.'''
    distribution = 'ak4_pt0_mjj'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get signal region
    h = h.integrate('region', 'sr_vbf_no_veto_all')

    # Get background MC (total)
    mc_regex = re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}')
    h = h.integrate('dataset', mc_regex)

    fig, ax = plt.subplots()
    for pt_slice in pt_slices:
        # Get the mjj distribution with the relevant jet pt slice
        _h = h.integrate('jetpt', pt_slice)

        hist.plot1d(_h, ax=ax, clear=False)

    # TODO: Aesthetic details here

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'mjj_by_jet_pt_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # Jet pt slices
    pt_slices = [
        slice(40,80),
        slice(80,120),
        slice(120,200),
        slice(200,300),
        slice(300,),
    ]

    for year in [2017, 2018]:
        plot_mjj_by_leadingjetpt(acc, outtag, pt_slices, year)

if __name__ == '__main__':
    main()