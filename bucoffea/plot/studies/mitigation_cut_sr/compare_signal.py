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

def compute_loss(h_nom, h_mitigated):
    '''Compute the additional loss of signal with the mitigation cut, by calculating the area under the histograms.'''
    vals_nom = h_nom.values(overflow='over')[()]
    vals_mitigated = h_mitigated.values(overflow='over')[()]
    bins = h_nom.axes()[0].edges(overflow='over')
    bin_widths = np.diff(bins)

    print(vals_nom)
    print(vals_mitigated)

    integral_nom = np.sum(vals_nom * bin_widths)
    integral_mitigated = np.sum(vals_mitigated * bin_widths)

    # Calculate percentage difference between the two
    percent_diff = (integral_nom - integral_mitigated) / integral_nom * 100
    return percent_diff

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
    h = h.integrate('dataset', re.compile('VBF_HToInv.*2017'))[re.compile('^sr_vbf((?!veto_all).)*$')]

    # Plot comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region', overflow='over')
    
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e5)

    # Calculate and plot ratio
    h_sr_vbf = h.integrate('region', 'sr_vbf')
    h_eemitigation_v1 = h.integrate('region', 'sr_vbf_eemitigationv1')
    h_eemitigation_v2 = h.integrate('region', 'sr_vbf_eemitigationv2')
    h_eemitigation_v1_vetohfhf = h.integrate('region', 'sr_vbf_eemitigationv1_vetohfhf')

    # Compute the percent losses in yield for both mitigation strategies
    loss_v1 = compute_loss(h_sr_vbf, h_eemitigation_v1)
    loss_v2 = compute_loss(h_sr_vbf, h_eemitigation_v2)
    loss_v1_vetohfhf = compute_loss(h_sr_vbf, h_eemitigation_v1_vetohfhf)

    centers = h_sr_vbf.axes()[0].centers(overflow='over')
    r_eemitigation_v1 = h_eemitigation_v1.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    r_eemitigation_v2 = h_eemitigation_v2.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    r_eemitigation_v1_vetohfhf = h_eemitigation_v1_vetohfhf.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    rax.plot(centers, r_eemitigation_v1, ls='', marker='o', label='EEv1')
    rax.plot(centers, r_eemitigation_v2, ls='', marker='o', label='EEv2')
    rax.plot(centers, r_eemitigation_v1_vetohfhf, ls='', marker='o', label='EEv1 + HF-HF veto')
    
    rax.grid(True)
    rax.set_ylim(0.9,1.1)
    rax.set_ylabel('Ratio to nominal SR')
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.legend(ncol=3)

    loc = matplotlib.ticker.MultipleLocator(base=0.05)
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