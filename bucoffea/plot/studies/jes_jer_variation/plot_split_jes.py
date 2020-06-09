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
    'GluGlu2017' : r'$ggH(inv) \ 2017$ split JEC uncertainties',
    'GluGlu2018' : r'$ggH(inv) \ 2018$ split JEC uncertainties',
    'VBF2017' : r'$VBF \ H(inv) \ 2017$ split JEC uncertainties',
    'VBF2018' : r'$VBF \ H(inv) \ 2018$ split JEC uncertainties',
    'ZJets' : r'$Z(\nu\nu) \ 2016$ split JEC uncertainties'
}

# Define all possible binnings for all variables in this dictionary
mjj_binning_v1 = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
mjj_binning_single_bin = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200,3500])

met_binning_v1_2016 = hist.Bin('met', r'$MET \ (GeV)$', list(range(0,500,50)) + list(range(500,1100,100))) 
met_binning_v1_2017 = hist.Bin('met', r'$MET \ (GeV)$', list(range(0,500,100)) + list(range(500,1250,250))) 
met_binning_single_bin = hist.Bin('met', r'$MET \ (GeV)$', [200,1500])
met_binning_coarse = hist.Bin('met', r'$MET \ (GeV)$', [250,300,400,500,800,1500])

binnings = {
    'met' : {
        'initial' : {'2016' : met_binning_v1_2016, '2017': met_binning_v1_2017, '2018' : met_binning_v1_2017},
        'single bin' : {'2016' : met_binning_single_bin, '2017' : met_binning_single_bin, '2018' : met_binning_single_bin},
        'coarse' : {'2016' : met_binning_coarse, '2017' : met_binning_coarse, '2018': met_binning_coarse}
    },
    'mjj' : {
        'initial' : {'2016' : mjj_binning_v1, '2017': mjj_binning_v1, '2018': mjj_binning_v1},
        'single bin' : {'2016': mjj_binning_single_bin, '2017': mjj_binning_single_bin, '2018': mjj_binning_single_bin}
    }
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--tag', help='Tag for dataset to be used.')
    parser.add_argument('--analysis', help='The analysis being considered, default is vbf.', default='vbf')
    args = parser.parse_args()
    return args

# Bin selection should be one of the following:
# Coarse, single bin, initial
def plot_split_jecunc(acc, out_tag, dataset_tag, year, plot_total=True, skimmed=True, bin_selection='initial', analysis='vbf'):
    '''Plot all split JEC uncertainties on the same plot.'''
    # Load the relevant variable to analysis, select binning
    variable_to_use = 'mjj' if analysis == 'vbf' else 'met'
    acc.load(variable_to_use)
    h = acc[variable_to_use]

    # Rebin the histogram
    new_bins = binnings[variable_to_use][bin_selection][year]
    h = h.rebin(variable_to_use , new_bins)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    region_suffix = '_j' if analysis == 'monojet' else '_vbf'

    dataset_name = dataset_tag.replace(f'{year}', '')
    h = h.integrate('dataset', re.compile(f'{dataset_name}.*{year}'))[re.compile(f'sr{region_suffix}.*')]

    h_nom = h.integrate('region', f'sr{region_suffix}')
    
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
        if region.name == f'sr{region_suffix}':
            continue
        if not plot_total:
            if "Total" in region.name:
                continue
        h_var = h.integrate('region', region)
        var_label = region.name.replace(f'sr{region_suffix}_', '')
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)
        if skimmed:
            if var_label_skimmed not in vars_to_look_at:
                continue
        hist.plotratio(h_var, h_nom, ax=ax, clear=False, label=var_label, unc='num',  guide_opts={}, error_opts=data_err_opts)
    
    ax.legend(ncol=2, prop={'size': 4.5})
    ax.set_ylabel('JEC Variation / Nominal')
    # Aesthetics, different for different analyses
    if analysis == 'monojet':
        ax.set_ylim(0.9,1.1)
        loc = matplotlib.ticker.MultipleLocator(base=0.02)
        ax.yaxis.set_major_locator(loc)
    elif analysis == 'vbf':
        if bin_selection == 'single bin':
            loc = matplotlib.ticker.MultipleLocator(base=0.02)
            ax.set_ylim(0.87,1.13)
        elif bin_selection == 'initial':
            loc = matplotlib.ticker.MultipleLocator(base=0.05)
            ax.set_ylim(0.75,1.35)
        ax.yaxis.set_major_locator(loc)
        
    ax.set_title(titles[dataset_tag])
    ax.grid(True)


    # Save figure
    outdir = f'./output/{out_tag}/splitJEC/{analysis}'
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
    analysis = args.analysis

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

    # Determine the year of the dataset
    if '2017' in dataset_tag:
        year = '2017' # 2017 VBF + ggH signals
    elif '2018' in dataset_tag:
        year = '2018' # 2018 VBF + ggH signals
    else:
        year = '2016' # 2016 Znunu

    # Plot split JEC uncertainties in two ways: 
    # 1. All uncertainty sources plotted on a single bin
    # 2. Only the largest sources are plotted, with multiple bins
    plot_split_jecunc(acc, out_tag, dataset_tag, year, analysis, skimmed=False, bin_selection='single bin')
    plot_split_jecunc(acc, out_tag, dataset_tag, year, analysis, skimmed=True, bin_selection='initial')

if __name__ == '__main__':
    main()