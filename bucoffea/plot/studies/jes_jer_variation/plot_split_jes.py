#!/usr/bin/env python

# -------------------------
# Script for plotting split JES uncertainties
# -------------------------

import os
import sys
import re
import argparse
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt
import matplotlib.ticker
import mplhep as hep
import numpy as np
# import pandas as pd
from pprint import pprint
from itertools import chain

pjoin = os.path.join

titles = {
    'GluGlu' : r'$ggH(inv) \ 2017$ split JEC uncertainties',
    'VBF' : r'$VBF \ H(inv) \ 2017$ split JEC uncertainties',
    'ZJets' : r'$Z(\nu\nu) \ 2016$ split JEC uncertainties'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--tag', help='Tag for dataset to be used.')
    args = parser.parse_args()
    return args

def plot_split_jecunc(acc, out_tag, dataset_tag, plot_total=True):
    '''Plot all split JEC uncertainties on the same plot.'''
    acc.load('recoil')
    h = acc['recoil']

    # recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]
    recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900]
    recoil_bins_2016_coarse = [250,1000]
    recoil_bin = hist.Bin('recoil', 'Recoil (GeV)', recoil_bins_2016_coarse)
    h = h.rebin('recoil', recoil_bin)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('dataset', re.compile(f'{dataset_tag}.*'))[re.compile('sr_j.*')]

    h_nom = h.integrate('region', 'sr_j')
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'elinewidth': 1,
    }

    fig, ax = plt.subplots()
    # Setup the color map
    colormap = plt.cm.nipy_spectral 
    num_plots = 12 if plot_total else 11
    colors = []
    for i in np.linspace(0,1,num_plots):
        colors.append([colormap(i), colormap(i)])

    # Flatten the color list
    colors = list(chain.from_iterable(colors))
    ax.set_prop_cycle('color', colors)

    for region in h.identifiers('region'):
        if region.name == 'sr_j':
            continue
        if not plot_total:
            if "Total" in region.name:
                continue
        h_var = h.integrate('region', region)
        var_label = region.name.replace('sr_j_', '')
        hist.plotratio(h_var, h_nom, ax=ax, clear=False, label=var_label, unc='num',  guide_opts={}, error_opts=data_err_opts)
    
    ax.legend(ncol=2, prop={'size': 4.5})
    ax.set_ylabel('JEC Variation / Nominal')
    ax.set_xlim(500,750)
    ax.set_title(titles[dataset_tag])
    ax.grid(True)

    # Save figure
    outdir = f'./output/{out_tag}/splitJEC'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = pjoin(outdir, f'{dataset_tag}_splitJEC.pdf')
    fig.savefig(outfile)

def main():
    args = parse_cli()
    inpath = args.inpath
    dataset_tag = args.tag

    acc = dir_archive(
        inpath,
        serialized=True,
        compression=0,
        memsize=1e3
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        out_tag = inpath.split('/')[-2]
    else:
        out_tag = inpath.split('/')[-1]

    plot_split_jecunc(acc, out_tag, dataset_tag)

if __name__ == '__main__':
    main()