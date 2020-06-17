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
    'zoverw_2017' : r'$Z(\nu\nu) \ / \ W(\ell\nu) \ 2017$',
    'zoverw_2018' : r'$Z(\nu\nu) \ / \ W(\ell\nu) \ 2018$',
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
        'defaultBinning' : {'2016' : met_binning_v1_2016, '2017': met_binning_v1_2017, '2018' : met_binning_v1_2017},
        'singleBin' : {'2016' : met_binning_single_bin, '2017' : met_binning_single_bin, '2018' : met_binning_single_bin},
        'coarseBin' : {'2016' : met_binning_coarse, '2017' : met_binning_coarse, '2018': met_binning_coarse}
    },
    'mjj' : {
        'defaultBinning' : {'2016' : mjj_binning_v1, '2017': mjj_binning_v1, '2018': mjj_binning_v1},
        'singleBin' : {'2016': mjj_binning_single_bin, '2017': mjj_binning_single_bin, '2018': mjj_binning_single_bin}
    }
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--tag', help='Tag for the transfer factor to be used.')
    parser.add_argument('--analysis', help='The analysis being considered, default is vbf.', default='vbf')
    # parser.add_argument('--regroup', help='Construct the uncertainty plot with the sources grouped into correlated and uncorrelated.', action='store_true')
    args = parser.parse_args()
    return args

def plot_split_jecunc_ratios(acc, out_tag, transfer_factor_tag, tag_num, tag_denom, year, plot_total=True, skimmed=True, bin_selection='defaultBinning', analysis='vbf'):
    '''Plot all split JEC uncertainties on transfer factors in the same plot.'''
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

    # Get the histograms for numerator and denominator
    h_num = h.integrate('dataset', re.compile(f'{tag_num}.*{year}'))[re.compile(f'sr{region_suffix}.*')]
    h_den = h.integrate('dataset', re.compile(f'{tag_denom}.*{year}'))[re.compile(f'sr{region_suffix}.*')]

    h_nominal_num = h_num.integrate('region', f'sr{region_suffix}')
    h_nominal_den = h_den.integrate('region', f'sr{region_suffix}')
    # Get the nominal ratio and store it in a dict
    ratios = {}
    nominal_ratio = h_nominal_num.values()[()] / h_nominal_den.values()[()]
    ratios['nominal'] = nominal_ratio

    data_err_opts = {
        'linestyle':'-',
        'marker': '.',
        'markersize': 10.,
        'elinewidth': 1,
    }

    fig, ax = plt.subplots()
    centers = h_num.axis(variable_to_use).centers()

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

    for region in h_num.identifiers('region'):
        if region.name == f'sr{region_suffix}':
            continue
        if not plot_total:
            if "Total" in region.name:
                continue
        h_varied_num = h_num.integrate('region', region)
        h_varied_den = h_den.integrate('region', region)
        var_label = region.name.replace(f'sr{region_suffix}_', '')
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)

        # Get the varied ratio and store it
        varied_ratio = h_varied_num.values()[()] / h_varied_den.values()[()]
        # ratios[var_label] = varied_ratio

        if skimmed:
            if var_label_skimmed not in vars_to_look_at:
                continue
        dratio = varied_ratio / nominal_ratio
        ax.plot(centers, dratio, marker='o', label=var_label)
        # hist.plotratio(h_var, h_nom, ax=ax, clear=False, label=var_label, unc='num',  guide_opts={}, error_opts=data_err_opts)

    # Aesthetics
    ax.grid(True)
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel('JEC uncertainty')
    if bin_selection == 'singleBin':
        ax.set_ylim(0.97,1.03)
        ticker_base = 0.01
    else:
        ax.set_ylim(0.9,1.1)
        ticker_base = 0.02
    ax.set_title(titles[transfer_factor_tag])
    ax.legend(ncol=2, prop={'size': 4.5})

    loc = matplotlib.ticker.MultipleLocator(base=ticker_base)
    ax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{out_tag}/splitJEC/vbf/transfer_factors'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    skimming_suffix = '_skimmed' if skimmed else ''

    filename = f'{transfer_factor_tag}_splitJEC{skimming_suffix}_{bin_selection}.pdf'
    outpath = pjoin(outdir, filename)

    fig.savefig(outpath)

###########################

def main():
    args = parse_cli()
    inpath = args.inpath
    transfer_factor_tag = args.tag
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

    # Determine the datasets for the transfer factor
    if 'zoverw' in transfer_factor_tag:
        tag_num = 'ZJets'
        tag_denom = 'WJets'

    # Get the year for the datasets, from the TF tag
    # NOTE: TF tag should be in the format of "process1overprocess2_year"
    year = transfer_factor_tag.split('_')[1]

    # Ratio plotting to be called here
    plot_split_jecunc_ratios(acc, out_tag, transfer_factor_tag, tag_num, tag_denom, year, skimmed=True, bin_selection='defaultBinning')
    plot_split_jecunc_ratios(acc, out_tag, transfer_factor_tag, tag_num, tag_denom, year, skimmed=False, bin_selection='singleBin')

if __name__ == '__main__':
    main()