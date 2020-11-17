#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

def plot_jet_pt(acc, outtag, distribution):
    '''Plot the jet pt distribution for leading or trailing jet (in endcap).'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # NOTE: Hopefully this works
    endcap_slice = slice(-3.0,-2.5) + slice(2.5, 3.0)
    h = h.integrate('jeteta', endcap_slice)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    signal_line_opts = {
        'color' : 'crimson'
    }

    for year in [2017, 2018]:
        h_data = h[f'MET_{year}']
        h_mc = h[f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}']
        h_signal = h[f'VBF_HToInv_{year}']

        fig, ax = plt.subplots()
        hist.plot1d(h_data, ax=ax, overlay='dataset', error_opts=data_err_opts)
        hist.plot1d(h_mc, ax=ax, overlay='dataset', stack=True, clear=False)
        hist.plot1d(h_signal, ax=ax, overlay='dataset', clear=False, line_opts=signal_line_opts)

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outpath = pjoin(outdir, f"{distribution.replace('_eta0', '')}_{year}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for distribution in ['ak4_pt0_eta0', 'ak4_pt1_eta1']:
        plot_jet_pt(acc, outtag, distribution=distribution)

if __name__ == '__main__':
    main()
