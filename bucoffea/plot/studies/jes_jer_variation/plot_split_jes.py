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

# Bin selection should be one of the following:
# Coarse, single bin, initial
def plot_split_jecunc(acc, out_tag, dataset_tag, year, plot_total=True, skimmed=True, bin_selection='coarse'):
    '''Plot all split JEC uncertainties on the same plot.'''
    acc.load('met')
    h = acc['met']

    # The bin selection that was used initially
    if bin_selection == 'initial':
        if year == 2016:
            met_bins = list(range(0,500,50)) + list(range(500,1100,100)) 
        else:
            met_bins = list(range(0,500,100)) + list(range(500,1250,250)) 

    elif bin_selection == 'coarse':
        met_bins = [250,300,400,500,800,1500]

    elif bin_selection == 'single bin':
        met_bins = [250,1500]

    # met_bins_v1 = [250,275,300,350,400,450,500,650,800,1150,1500]
    met_bin = hist.Bin('met', 'MET (GeV)', met_bins)
    # h = h.rebin('recoil', recoil_bin)
    h = h.rebin('met', met_bin)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('dataset', re.compile(f'{dataset_tag}.*'))[re.compile('sr_j.*')]

    h_nom = h.integrate('region', 'sr_j')
    
    data_err_opts = {
        'linestyle':'-',
        'marker': '.',
        'markersize': 10.,
        'elinewidth': 1,
    }

    fig, ax = plt.subplots()

    # Look at only certain variations if we are not to plot everything
    vars_to_look_at = ['jesRelativeBal', f'jesRelativeSample_{year}', 'jesAbsolute', f'jesAbsolute_{year}', 'jesFlavorQCD', 'jesTotal']
    
    # Setup the color map
    colormap = plt.cm.nipy_spectral
    num_plots = len(vars_to_look_at) if skimmed else 12
    colors = []
    for i in np.linspace(0,0.9,num_plots):
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
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)
        if skimmed:
            if var_label_skimmed not in vars_to_look_at:
                continue
        hist.plotratio(h_var, h_nom, ax=ax, clear=False, label=var_label, unc='num',  guide_opts={}, error_opts=data_err_opts)
    
    ax.legend(ncol=2, prop={'size': 4.5})
    ax.set_ylabel('JEC Variation / Nominal')
    ax.set_ylim(0.9,1.1)
    ax.set_title(titles[dataset_tag])
    ax.grid(True)

    loc = matplotlib.ticker.MultipleLocator(base=0.02)
    ax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{out_tag}/splitJEC'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if bin_selection == 'coarse':
        fname = f'{dataset_tag}_splitJEC_skimmed_METbinsv2.pdf' if skimmed else f'{dataset_tag}_splitJEC_METbinsv2.pdf'
    elif bin_selection == 'single bin':
        fname = f'{dataset_tag}_splitJEC_skimmed_singlebin.pdf' if skimmed else f'{dataset_tag}_splitJEC_singlebin.pdf'
    elif bin_selection == 'initial':
        fname = f'{dataset_tag}_splitJEC_skimmed.pdf' if skimmed else f'{dataset_tag}_splitJEC.pdf'
        
    outfile = pjoin(outdir, fname)
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

    year = 2017 if dataset_tag in ['VBF', 'GluGlu'] else 2016

    plot_split_jecunc(acc, out_tag, dataset_tag, year)

if __name__ == '__main__':
    main()