#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from coffea import hist
from pprint import pprint

pjoin = os.path.join

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

rebin = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}\ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'recoil' : hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016)
}

def compare_gjets(acc_lo, acc_nlo, variable='mjj'):
    '''Make comparison plot between LO GJets + SF and NLO GJets'''
    acc_lo.load(variable)
    acc_nlo.load(variable)

    h_lo = acc_lo[variable]
    h_nlo = acc_nlo[variable]

    # Rebin variable
    newbin = rebin[variable]
    h_lo = h_lo.rebin(variable, newbin)
    h_nlo = h_nlo.rebin(variable, newbin)

    # Merge, rescale
    h_lo = merge_extensions(h_lo, acc_lo, reweight_pu=False)
    scale_xs_lumi(h_lo)
    h_lo = merge_datasets(h_lo)

    h_nlo = merge_extensions(h_nlo, acc_nlo, reweight_pu=False)
    scale_xs_lumi(h_nlo)
    h_nlo = merge_datasets(h_nlo)

    # Pick the relevant region and dataset
    h_lo = h_lo.integrate('region', 'cr_g_vbf').integrate('dataset', re.compile('GJets_DR-0p4.*2017'))
    h_nlo = h_nlo.integrate('region', 'cr_g_vbf').integrate('dataset', re.compile('GJets_1j.*2017'))

    # Plot comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # Plot LO as histogram
    hist.plot1d(h_lo, ax=ax, binwnorm=True)

    # Plot NLO as data points
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    hist.plot1d(h_nlo, ax=ax, error_opts=data_err_opts, clear=False, binwnorm=True)

    # Fix legend labels
    handles, _ = ax.get_legend_handles_labels()
    new_labels = ['LO + SF', 'NLO']
    ax.legend(handles, new_labels)

    ax.set_ylabel('Counts / Bin width')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e5)
    ax.set_xlabel('')

    # Plot ratio pad
    hist.plotratio(h_nlo, h_lo, ax=rax, unc='num', error_opts=data_err_opts)
    rax.set_ylabel('NLO / LO + SF')
    rax.set_ylim(0,2)
    rax.grid(True)

    # Save the plot
    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outfile = pjoin(outdir, f'comparison_{variable}.pdf')

    fig.savefig(outfile)
    print(f'File saved: {outfile}')

def main():
    # Paths to coffea files with LO and NLO GJets samples
    path_lo, path_nlo = sys.argv[1:3]

    acc_lo = dir_archive(
        path_lo,
        memsize=1e3,
    )

    acc_nlo = dir_archive(
        path_nlo,
        memsize=1e3,
    )

    acc_lo.load('sumw')
    acc_lo.load('sumw2')
    acc_nlo.load('sumw')
    acc_nlo.load('sumw2')

    compare_gjets(acc_lo, acc_nlo, variable='mjj')
    compare_gjets(acc_lo, acc_nlo, variable='recoil')

if __name__ == '__main__':
    main()