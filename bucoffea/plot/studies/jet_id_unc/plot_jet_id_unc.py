#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_jet_id_unc(acc, outtag, dataset='ZJetsToNuNu'):
    '''Plot the variation in mjj distribution in signal with the propagation of jet ID uncertainties in the analysis.'''
    acc.load('mjj')
    h = acc['mjj']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)
            
    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'{dataset}.*{year}'))
        h_nom = _h.integrate('region', 'sr_vbf_no_veto_all')
        h_sfup = _h.integrate('region', 'sr_vbf_no_veto_all_jetsfUp')
        h_sfdown = _h.integrate('region', 'sr_vbf_no_veto_all_jetsfDown')

        fig, ax, rax = fig_ratio()
        hist.plot1d(h_sfup, ax=ax)
        hist.plot1d(h_sfdown, ax=ax, clear=False)
        hist.plot1d(h_nom, ax=ax, clear=False)

        ax.set_xlabel('')
        ax.set_yscale('log')
        ax.set_ylim(1e0, 1e5)
        ax.set_title(f'Jet SF Uncertainties on QCD Z(vv): {year}')

        labels = [
            'Jet ID SF up',
            'Jet ID SF down',
            'Nominal',
        ]

        ax.legend(labels=labels)

        # Plot ratios w.r.t. nominal
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        hist.plotratio(h_sfup, h_nom, ax=rax, unc='num', error_opts=data_err_opts)
        hist.plotratio(h_sfdown, h_nom, ax=rax, unc='num', clear=False, error_opts=data_err_opts)

        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
        rax.set_ylabel('Ratio to Nominal')
        rax.set_ylim(0.95,1.05)
        rax.grid(True)

        rax.axhline(1, xmin=0, xmax=1, color='black')

        loc = MultipleLocator(0.05)
        rax.yaxis.set_major_locator(loc)

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outpath = pjoin(outdir, f'jet_sf_uncs_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_jet_id_unc(acc, outtag)

if __name__ == '__main__':
    main()