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

def plot_mjj_by_leadingjetpt(acc, outtag, pt_slices, year, etaslice='pos'):
    '''In bins of leading jet pt, plot mjj distribution for the total background in SR.'''
    distribution = 'ak4_pt0_mjj'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get signal region
    h = h.integrate('region', f'sr_vbf_ak40_{etaslice}_endcap')

    # Get background MC (total)
    mc_regex = re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}')
    h = h.integrate('dataset', mc_regex)

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)

    fig, ax = plt.subplots()
    for pt_slice in pt_slices:
        # Get the mjj distribution with the relevant jet pt slice
        _h = h.integrate('jetpt', pt_slice)

        hist.plot1d(_h, ax=ax, clear=False)

    labels = [
        r'$40 < p_T < 80$',
        r'$80 < p_T < 120$',
        r'$120 < p_T < 200$',
        r'$200 < p_T < 300$',
        r'$p_T > 300$',
    ]

    ax.legend(labels=labels, title=r'Leading jet $p_T$')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e4)
    ax.set_title('Total Background in SR')

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
        slice(300,None),
    ]

    for year in [2017, 2018]:
        # Positive and negative eta slices (for endcap)
        for etaslice in ['pos', 'neg']:
            plot_mjj_by_leadingjetpt(acc, outtag, pt_slices, year, etaslice)

if __name__ == '__main__':
    main()