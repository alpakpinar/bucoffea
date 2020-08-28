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
from pprint import pprint
from itertools import chain
from data import tag_to_dataset_pairs
import uproot
from collections import OrderedDict
from tabulate import tabulate

pjoin = os.path.join

# Figure titles for all ratios
titles = {
    'znunu_over_wlnu17' : r'$Z(\nu\nu) \ / \ W(\ell\nu) \ 2017$',
    'znunu_over_wlnu18' : r'$Z(\nu\nu) \ / \ W(\ell\nu) \ 2018$',
    'znunu_over_zmumu17' : r'$Z(\nu\nu) \ / \ Z(\mu\mu) \ 2017$',
    'znunu_over_zmumu18' : r'$Z(\nu\nu) \ / \ Z(\mu\mu) \ 2018$',
    'znunu_over_zee17' : r'$Z(\nu\nu) \ / \ Z(ee) \ 2017$',
    'znunu_over_zee18' : r'$Z(\nu\nu) \ / \ Z(ee) \ 2018$',
    'wlnu_over_wenu17' : r'$W(\ell\nu) \ / \ W(e\nu) \ 2017$',
    'wlnu_over_wenu18' : r'$W(\ell\nu) \ / \ W(e\nu) \ 2018$',
    'wlnu_over_wmunu17' : r'$W(\ell\nu) \ / \ W(\mu\nu) \ 2017$',
    'wlnu_over_wmunu18' : r'$W(\ell\nu) \ / \ W(\mu\nu) \ 2018$',
    'gjets_over_znunu17' : r'$\gamma + jets \ / \ Z(\nu\nu) \ 2017$',
    'gjets_over_znunu18' : r'$\gamma + jets \ / \ Z(\nu\nu) \ 2018$',
    'wlnu_over_gjets17' : r'$W(\ell\nu) \ / \ \gamma + jets \ 2017$',
    'wlnu_over_gjets18' : r'$W(\ell\nu) \ / \ \gamma + jets \ 2018$'
}

# Define all possible binnings for all variables in this dictionary
mjj_binning_v1 = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
mjj_binning_single_bin = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200,3500])

met_binning_v1_2016 = hist.Bin('recoil', r'$Recoil \ (GeV)$', list(range(0,500,50)) + list(range(500,1100,100))) 
met_binning_v1_2017 = hist.Bin('recoil', r'$Recoil \ (GeV)$', list(range(0,500,100)) + list(range(500,1250,250))) 
met_binning_single_bin = hist.Bin('recoil', r'$Recoil \ (GeV)$', [200,1500])
met_binning_coarse = hist.Bin('recoil', r'$Recoil \ (GeV)$', [250,300,400,500,800,1500])

binnings = {
    'recoil' : {
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
    parser.add_argument('--analysis', help='The analysis being considered, default is vbf.', default='vbf')
    parser.add_argument('--run', help='Which samples to run on: qcd, ewk.', nargs='*')
    parser.add_argument('--onlyRun', help='Specify regex to only run over some transfer factors and skip others.')
    parser.add_argument('--tabulate', help='Tabulate the variation/unc values.', action='store_true')
    args = parser.parse_args()
    return args

def plot_split_jecunc_ratios(acc, out_tag, transfer_factor_tag, dataset_info, year, process, outputrootfile, plot_total=True, skimmed=True, bin_selection='defaultBinning', analysis='vbf', tabulate_top5=False):
    '''Plot all split JEC uncertainties on transfer factors in the same plot.'''
    # Load the relevant variable to analysis, select binning
    print(f'MSG% Working on: {transfer_factor_tag}')
    variable_to_use = 'mjj' if analysis == 'vbf' else 'recoil'
    acc.load(variable_to_use)
    h = acc[variable_to_use]

    # Rebin the histogram
    new_bins = binnings[variable_to_use][bin_selection][str(year)]
    h = h.rebin(variable_to_use , new_bins)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    region_suffix = '_j' if analysis == 'monojet' else '_vbf'

    # Get information about the datasets in num and denom from the "dataset_info" parameter: Regex to match and the region to consider
    dataset_info_num = dataset_info['dataset1']
    dataset_info_den = dataset_info['dataset2']

    regex_num, region_num = dataset_info_num['regex'], dataset_info_num['region']
    regex_den, region_den = dataset_info_den['regex'], dataset_info_den['region']

    # Get the histograms for numerator and denominator
    h_num = h.integrate('dataset', re.compile(regex_num))[re.compile(f'{region_num}{region_suffix}.*')]
    h_den = h.integrate('dataset', re.compile(regex_den))[re.compile(f'{region_den}{region_suffix}.*')]

    h_nominal_num = h_num.integrate('region', f'{region_num}{region_suffix}')
    h_nominal_den = h_den.integrate('region', f'{region_den}{region_suffix}')
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
    edges = h_num.axis(variable_to_use).edges()

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

    # Store the uncertainty from each source
    uncs = {}
    varied = {}

    for region in h_num.identifiers('region'):
        if region.name.endswith(region_suffix):
            continue
        if not plot_total:
            if "Total" in region.name:
                continue

        var_label = region.name.replace(f'{region_num}{region_suffix}_', '')
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)

        region_for_num = f'{region_num}{region_suffix}_{var_label}'
        region_for_den = f'{region_den}{region_suffix}_{var_label}'

        h_varied_num = h_num.integrate('region', region_for_num)
        h_varied_den = h_den.integrate('region', region_for_den)

        # Get the varied ratio and store it
        varied_ratio = h_varied_num.values()[()] / h_varied_den.values()[()]

        if skimmed:
            if var_label_skimmed not in vars_to_look_at:
                continue
        
        dratio = varied_ratio / nominal_ratio
        # Do not plot JER for now
        if not 'jer' in region.name:
            ax.plot(centers, dratio, marker='o', label=var_label)

        # Save the uncertainties to an output root file
        hist_name = f'{transfer_factor_tag}_{process}_{var_label}'
        outputrootfile[hist_name] = (dratio, edges)

        # Store the uncs and variations
        uncs[var_label] = np.abs(dratio - 1) * 100
        varied[var_label] = varied_ratio

    # Aesthetics
    ax.grid(True)
    if analysis == 'vbf':
        ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    elif analysis == 'monojet':
        ax.set_xlabel('Recoil (GeV)')
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
    outdir = f'./output/{out_tag}/splitJEC/{analysis}/transfer_factors'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    skimming_suffix = '_skimmed' if skimmed else ''

    filename = f'{transfer_factor_tag}_{process}_splitJEC{skimming_suffix}_{bin_selection}.pdf'
    outpath = pjoin(outdir, filename)

    fig.savefig(outpath)
    print(f'MSG% File created: {outpath}')
    
    # Sort the sources
    if tabulate_top5:
        print('MSG% Sorting the uncertainties and tabulating top 5.')
        sorted_uncs = OrderedDict()
        for k, v in sorted(uncs.items(), reverse=True, key=lambda item: item[1]):
            # Do not save JER uncertainties in these tables for now
            if 'jer' in k:
                continue 
            sorted_uncs[k] = v

        unc_sources = sorted_uncs.keys()
        new_unc_sources = []
        for idx, source in enumerate(unc_sources):
            if 'Up' in source:
                new_unc_sources.append(source)
                unc_label = source.replace('Up', '')
                new_unc_sources.append(f'{unc_label}Down')

        # Finally, get the top 5 unc sources + the total JES unc and tabulate
        num_sources_to_keep = 5
        new_sorted_uncs = OrderedDict()
        for idx, source in enumerate(new_unc_sources):        
            if idx == 2*(num_sources_to_keep+1):
                break
            new_sorted_uncs[source] = sorted_uncs[source]

        # Dump the table into an output txt file
        table = [["Uncertainty Source", "Nominal", "Up Variation", "Down Variation", "Up (%)", "Down (%)"]]
        for source in new_sorted_uncs.keys():
            if 'Up' in source:
                label = source.replace('Up', '')
                unc_up = new_sorted_uncs[f'{label}Up']
                unc_down = new_sorted_uncs[f'{label}Down']
                var_up = varied[f'{label}Up']
                var_down = varied[f'{label}Down']
                table.append([label, nominal_ratio, var_up, var_down, unc_up[0], unc_down[0]])

        # Save the table as a txt file
        outdir = f'./output/{out_tag}/splitJEC/{analysis}/tables'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outfile = pjoin(outdir, f'{transfer_factor_tag}_top_five_uncs.txt')

        with open(outfile, 'w+') as f:
            f.write('='*10 + '\n')
            f.write(transfer_factor_tag + '\n')
            f.write('='*10 + '\n')
            t = tabulate(table, floatfmt='.3f', headers='firstrow')
            f.write(t)

        print(f'MSG% Table saved at: {outfile}')

def main():
    args = parse_cli()
    inpath = args.inpath

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

    # List of all ratios
    all_ratios = tag_to_dataset_pairs.keys()

    # Save the uncertainties on TFs on an output root file
    rootdir = f'./output/{out_tag}/splitJEC/{args.analysis}/root'
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

    rootfile = pjoin(rootdir, f'{args.analysis}_tf_uncs.root')
    outputrootfile = uproot.recreate(rootfile)

    print(f'MSG% ROOT file created: {rootfile}')

    # Loop over each transfer factor
    for transfer_factor_tag in all_ratios:
        skip=False
        year = 2017 if '17' in transfer_factor_tag else 2018

        # Skip over some TFs if requested so
        if args.onlyRun:
            if not re.match(args.onlyRun, transfer_factor_tag):
                print(f'MSG% Skipping: {transfer_factor_tag}')
                continue

        # Run QCD ratio
        if 'qcd' in args.run:
            qcd_dataset_info = tag_to_dataset_pairs[transfer_factor_tag]['qcd']
            # Plot split uncertainties as flat uncertainties (single bin)
            plot_split_jecunc_ratios(acc, out_tag, 
                transfer_factor_tag=transfer_factor_tag, 
                dataset_info=qcd_dataset_info, 
                year=year,
                process='qcd',
                outputrootfile=outputrootfile, 
                skimmed=False, 
                bin_selection='singleBin',
                analysis=args.analysis,
                tabulate_top5=args.tabulate)
                
        # Run EWK ratio
        if 'ewk' in args.run:
            ewk_dataset_info = tag_to_dataset_pairs[transfer_factor_tag]['ewk']
            # Plot split uncertainties as flat uncertainties (single bin)
            plot_split_jecunc_ratios(acc, out_tag, 
                transfer_factor_tag=transfer_factor_tag, 
                dataset_info=ewk_dataset_info, 
                year=year, 
                process='ewk', 
                outputrootfile=outputrootfile, 
                skimmed=False, 
                bin_selection='singleBin',
                analysis=args.analysis,
                tabulate_top5=args.tabulate)

if __name__ == '__main__':
    main()