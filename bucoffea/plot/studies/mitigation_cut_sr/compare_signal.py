#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
}

def compare_signal(acc, outtag, variable='mjj'):
    '''Compare the signal distribution with two different noise mitigation cuts applied.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        h = h.rebin(variable, REBIN[variable])

    # Get the signal dataset
    h = h.integrate('dataset', re.compile('VBF_HToInv.*2017'))[re.compile('^sr_vbf((?!veto).)*$')]

    # Plot comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region')
    
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e5)

    # Calculate and plot ratio
    h_sr_vbf = h.integrate('region', 'sr_vbf')
    h_eemitigation_v1 = h.integrate('region', 'sr_vbf_eemitigationv1')
    h_eemitigation_v2 = h.integrate('region', 'sr_vbf_eemitigationv2')

    centers = h_sr_vbf.axes()[0].centers()
    r_eemitigation_v1 = h_eemitigation_v1.values()[()] / h_sr_vbf.values()[()]
    r_eemitigation_v2 = h_eemitigation_v2.values()[()] / h_sr_vbf.values()[()]
    rax.plot(centers, r_eemitigation_v1, ls='', marker='o', label='EEv1')
    rax.plot(centers, r_eemitigation_v2, ls='', marker='o', label='EEv2')
    
    rax.grid(True)
    rax.set_ylim(0.94,1.06)
    rax.set_ylabel('Ratio to nominal SR')
    rax.legend()

    loc = matplotlib.ticker.MultipleLocator(base=0.02)
    rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, 'signal_yields.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    compare_signal(acc, outtag)

if __name__ == '__main__':
    main()