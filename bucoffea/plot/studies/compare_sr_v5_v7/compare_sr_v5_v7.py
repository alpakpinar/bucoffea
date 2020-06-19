#!/usr/bin/env python

import argparse
import os
import re
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker

from klepto.archives import dir_archive

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from pprint import pprint

pjoin = os.path.join

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

# Define rebinnings
REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'recoil' : hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) )
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath_v5', help='Path containing merged coffea files produced from nanoAOD v5.')
    parser.add_argument('inpath_v7', help='Path containing merged coffea files produced from nanoAOD v7.')
    args = parser.parse_args()
    return args

def compare(acc_v5, acc_v7, distribution='mjj', year=2017):
    '''Compare the data in signal region (MET) for v5 and v7 NanoAOD.'''
    # Get the relevant axis name in the histogram 
    if 'ak4_pt' in distribution:
        ax_name = 'jetpt'
    elif 'ak4_eta' in distribution:
        ax_name = 'jeteta'
    else:
        ax_name = distribution

    def preprocess(h, acc):
        h = merge_extensions(h, acc, reweight_pu=False)
        h = merge_datasets(h)

        # Rebin (if needed)
        if distribution in REBIN.keys():
            h = h.rebin(h.axis(ax_name), REBIN[distribution])

        # Select the signal region + MET dataset
        h = h.integrate('dataset', f'MET_{year}').integrate('region', 'sr_vbf')
        return h

    acc_v5.load(distribution)
    h_v5 = acc_v5[distribution]
    acc_v7.load(distribution)
    h_v7 = acc_v7[distribution]

    h_v5 = preprocess(h_v5, acc_v5)    
    h_v7 = preprocess(h_v7, acc_v7)    

    # Make a comparison plot
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    values_v5 = h_v5.values(overflow='over')[()]
    values_v7 = h_v7.values(overflow='over')[()]

    edges = h_v5.axis(ax_name).edges(overflow='over')
    centers = h_v5.axis(ax_name).centers(overflow='over')

    # Get bin widths + bin width normalized values
    bin_widths = np.diff(edges)
    values_v5_binw_norm = values_v5 / bin_widths
    values_v7_binw_norm = values_v7 / bin_widths

    plot_opts = {
        'linestyle' : '',
        'marker' : 'o'
    }

    hist.plot1d(h_v5, ax=ax, binwnorm=True, overflow='over')
    ax.plot(centers, values_v7_binw_norm, label='NanoAOD v7', **plot_opts)
    # Fix the first label (should be NanoAOD v5)
    ax.legend(title=f'Data in SR: {year}', labels=['NanoAODv5', 'NanoAODv7'])

    ax.set_xlabel('')
    ax.set_ylabel('Counts / Bin Width')
    ax.set_yscale('log')
    ax.set_xlim(0, edges[-1])
    # Handle y-limit automatically
    low_ylim = 1e-2
    max_val = np.max(values_v5_binw_norm)
    high_ylim = max_val*10
    ax.set_ylim(low_ylim, high_ylim)
    ax.grid(True)

    # Plot the ratio pad 
    xlim = rax.get_xlim()
    rax.plot(xlim, [1,1], 'r')
    rax.set_xlim(xlim)

    ratio = values_v7 / values_v5
    rax.plot(centers, ratio, color='k', **plot_opts)

    rax.set_xlabel(h_v5.axis(ax_name).label)
    rax.set_ylabel('v7 / v5')
    rax.set_ylim(0.8, 1.2)

    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    rax.yaxis.set_major_locator(loc)
    rax.grid(True)

    # Save output
    outdir = f'./output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outfile = pjoin(outdir, f'{distribution}_v5_v7_comp.pdf')
    fig.savefig(outfile)
    print(f'File saved: {outfile}')

def main():
    args = parse_cli()
    # Load the two accumulators, one for v5 and one for v7 
    acc_v5 = dir_archive(
                    args.inpath_v5,
                    serialized=True,
                    compression=0,
                    memsize=1e3
                    )
    acc_v7 = dir_archive(
                    args.inpath_v7,
                    serialized=True,
                    compression=0,
                    memsize=1e3
                    )

    # The distributions to look at
    distributions = ['mjj', 'recoil', 'ak4_pt0', 'ak4_eta0', 'ak4_eta1']

    for distribution in distributions:
        compare(acc_v5, acc_v7, distribution=distribution)

if __name__ == '__main__':
    main()
