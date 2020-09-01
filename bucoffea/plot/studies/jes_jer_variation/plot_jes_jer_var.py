#!/usr/bin/env python

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
import pandas as pd
from data import tag_to_dataset_pairs, dataset_regex, indices_from_tags
from pprint import pprint

pjoin = os.path.join

# Plot aesthetics
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]

# Legend labels for each variation
var_to_legend_label = {
    ''         : 'Nominal',
    '_jer'     : 'JER',
    '_jesTotalUp'   : 'JES up',
    '_jesTotalDown' : 'JES down'
}

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing input coffea files.')
    parser.add_argument('--calc', help='List specifying calculation of individual processes, ratios or both. Specify "ind" for individual procs, "ratio" for ratios.', nargs='*', default=['ind','ratio'])
    parser.add_argument('--run', help='Which samples to run on: qcd, ewk.', nargs='*', default=['qcd', 'ewk'])
    parser.add_argument('--analysis', help='Type of analysis, VBF or monojet. Default is VBF', default='vbf')
    parser.add_argument('--onlyJES', help='Only plot JES uncertainties.', action='store_true')
    parser.add_argument('--onlyJER', help='Only plot JER uncertainties.', action='store_true')
    parser.add_argument('--save_to_df', help='Save results to output pandas dataframe, stored in an output pkl file.', action='store_true')
    parser.add_argument('--onlyRun', help='Specify regex to only run over some transfer factors and skip others.')
    args = parser.parse_args()
    return args

def dict_to_arr(d):
    '''Given a dictionary containing different weights as
       its values, concatenate the weights as a 2D numpy array.'''
    shape = len(d), len(list(d.values())[0])
    arr = np.zeros(shape)
    for idx, weight_arr in enumerate(d.values()):
        arr[idx] = weight_arr
    return arr

def get_unc(d, edges, out_tag, tag, sample_type):
    '''Given a dictionary containing different weights as
       its values, calculate the uncertainty in each bin.'''
    # Transform to 2D array
    arr = dict_to_arr(d)
    nom = arr[0] 
    # Calculate uncertainty in each mjj bin
    unc = np.zeros_like(nom)
    for idx, entry in enumerate(nom):
        bin_ = arr[:, idx]
        rng = bin_.max() - bin_.min()
        unc[idx] = rng/(2*nom[idx])
        
    # Dump the results to a .txt file
    outdir = f'./output/{out_tag}/txt'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{tag}_{sample_type}_jes_jer_unc.txt')
    with open(outpath, 'w+') as f:
        f.write('*'*20 + '\n')
        f.write('Combined Uncertainties' + '\n')
        f.write('*'*20 + '\n')
        for idx, sigma in enumerate(unc):
            f.write(f'{edges[idx]:.0f} < mjj < {edges[idx+1]:.0f}: {sigma*100:.2f}%\n')

    print(f'MSG% File saved: {outpath}')

def plot_jes_jer_var(acc, regex, region, tag, out_tag, title, sample_type, analysis='vbf'):
    '''Given the input accumulator and the regex
    describing the dataset, plot the mjj distribution
    with different JES/JER variations in the same canvas.
    =============
    PARAMETERS:
    =============
    acc         : The input accumulator containing the histograms.
    regex       : Regular expression matching the dataset name.
    region      : The region from which the event yields will be taken.
    tag         : Tag representing the process. (e.g. "wjet")
    out_tag     : Out-tag for output directory naming. The output files are going to be saved
                  under this directory.
    title       : Histogram title for plotting.
    sample_type : QCD ("qcd") or EWK ("ewk") sample.
    analysis    : Type of analysis under consideration, "vbf" or "monojet". Default is vbf.
    '''
    # If analysis is VBF, look at mjj distribution. If analysis is monojet, look at recoil.
    if analysis == 'vbf':
        acc.load('mjj')
        h = acc['mjj']
        # Rebin mjj
        mjj_bins = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
        mjj_bins_coarse = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(0,4000,1000))) 
        mjj_bins_very_coarse = hist.Bin('mjj', r'$M_{jj}$ (GeV)', [0,4000]) 
        h = h.rebin('mjj', mjj_bins_very_coarse)

    elif analysis == 'monojet':
        acc.load('recoil')
        h = acc['recoil']
        # Rebin recoil into one large bin
        recoil_bins_very_coarse = hist.Bin('recoil', r'Recoil (GeV)', [0,2000])
        h = h.rebin('recoil', recoil_bins_very_coarse)

    print(f'MSG% Working on: {tag}, {sample_type}')

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    
    # Analysis tag to specify correct regions
    if analysis == 'vbf':
        analysis_tag = '_vbf'
    elif analysis == 'monojet':
        analysis_tag = '_j'
    else:
        raise ValueError('Invalid value for analysis, should be "monojet" or "vbf"')

    # Pick the relevant dataset
    h = h[re.compile(regex)].integrate('dataset')

    nom_region_name = f'{region}{analysis_tag}'
    h = h[re.compile(f'{nom_region_name}.*')]
    # Pick the nominal yields
    h_nom = h.integrate('region', nom_region_name).values()[()]

    # Calculate the ratios of each variation
    # with respect to nominal counts
    ratios = {}
    variations = ['', '_jer', '_jesTotalUp', '_jesTotalDown']
    for variation in variations:
        ratios[variation] = h.integrate('region', f'{region}{analysis_tag}{variation}').values()[()] / h_nom

    edges = h.axes()[1].edges()
    centers = h.axes()[1].centers()
    
    # Store counts with all variations
    counts = []

    # Plot the variation + ratio pad
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 2)}, sharex=True)
    for idx, (var, ratio_arr) in enumerate(ratios.items()):
        # Retrieve the varied weights
        h_var = ratio_arr * h_nom
        hep.histplot(h_var, 
                     edges, 
                     label=var_to_legend_label[var],
                     ax=ax,
                     histtype='step'
                     )

        counts.append(h_var)

        if var != '':
            rax.plot(centers, ratio_arr, 'o', label=var_to_legend_label[var], c=colors[idx])

    # Aesthetics
    if analysis == 'vbf':
        xlim = (200,4000)
    elif analysis == 'monojet':
        xlim = (250,2000)
    ax.set_xlim(xlim)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.legend()
    
    # Handle y-limit dynamically
    min_count, max_count = min(counts), max(counts)
    lower_ylim = 0.8 * min_count
    upper_ylim = 1.2 * max_count
    ax.set_ylim(lower_ylim, upper_ylim)

    if '17' in tag:
        ax.text(1., 1., '2017',
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
                )
    
    elif '18' in tag:
        ax.text(1., 1., '2018',
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
                )

    rax.set_ylim(0.8, 1.2)
    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    rax.yaxis.set_major_locator(loc)
    rax.set_ylabel('Varied / Nominal')
    if analysis == 'vbf':
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    elif analysis == 'monojet':
        rax.set_xlabel(r'Recoil (GeV)')
    rax.legend()
    rax.grid(True)

    # Save figure
    outdir = f'./output/{out_tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{tag}_{sample_type}_jes_jer_variations.pdf')
    fig.savefig(outpath)
    plt.close()
    
    print(f'MSG% Histogram saved in: {outpath}')

def plot_jes_jer_var_ratio(acc, regex1, regex2, region1, region2, tag, out_tag, sample_type, analysis='vbf', plot_onlyJES=False, plot_onlyJER=False, bin_selection='singleBin'):
    '''Given the input accumulator, plot ratio of two datasets
    for each JES/JER variation, on the same canvas.
    ==============
    PARAMETERS:
    ==============
    acc           : Input accumulator containing the histograms.
    regex1        : Regular expression matching the dataset name in the numerator of the ratio.
    regex2        : Regular expression matching the dataset name in the denominator of the ratio.
    region1       : The region from which the data for the numerator is going to be taken from.
    region2       : The region from which the data for the denominator is going to be taken from.
    tag           : Tag for the process name. (e.g "wjet")
    out_tag       : Out-tag for naming output directory, output files are going to be saved under this directory.
    sample_type   : QCD ("qcd") or EWK ("ewk") sample. 
    analysis      : Type of analysis under consideration, "vbf" or "monojet". Default is vbf.
    plot_onlyJES  : Only plot JES uncertaintes on the plot.
    plot_onlyJER  : Only plot JER uncertaintes on the plot.
    bin_selection : Selection for binning, use "singleBin" for one bin or "coarse" or "fine".  
    '''
    # If analysis is VBF, look at mjj distribution. If analysis is monojet, look at recoil.
    if analysis == 'vbf':
        acc.load('mjj')
        h = acc['mjj']
        # Rebin mjj
        mjj_binning = {
            'singleBin' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', [0,4000]),
            'coarse' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(0,4000,1000))),
            'fine' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
        }
        # mjj_bins_coarse = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(0,4000,1000))) 
        binning_to_use = mjj_binning[bin_selection]
        h = h.rebin('mjj', binning_to_use)

    elif analysis == 'monojet':
        acc.load('recoil')
        h = acc['recoil']
        # Rebin recoil into one large bin
        recoil_bins_very_coarse = hist.Bin('recoil', r'Recoil (GeV)', [0,2000])
        binning_to_use = recoil_bins_very_coarse
        h = h.rebin('recoil', binning_to_use)

    print(f'MSG% Working on: {tag}, {sample_type}')

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    
    # Analysis tag to specify correct regions
    if analysis == 'vbf':
        analysis_tag = '_vbf'
    elif analysis == 'monojet':
        analysis_tag = '_j'
    else:
        raise ValueError('Invalid value for analysis, should be "monojet" or "vbf"')

    # Pick the relevant datasets 
    # Regex 1 matches the dataset for the numerator
    # Regex 2 matches the dataset for the denominator
    h = h[re.compile(f'{regex1}|{regex2}')]

    # h1: Histogram for the numerator
    # h2: Histogram for the denominator
    h1 = h[re.compile(regex1)].integrate('dataset')
    h2 = h[re.compile(regex2)].integrate('dataset')

    h1 = h1[re.compile(f'{region1}{analysis_tag}.*')]
    h2 = h2[re.compile(f'{region2}{analysis_tag}.*')]

    # Calculate the ratios and errors in ratios 
    # for each JES/JER variation
    ratios = {}
    err = {}

    for region in h1.values().keys():
        region_name = region[0]
        # Extract the variation type from the region name
        if region_name.endswith('_vbf') or region_name.endswith('_j'):
            var_name = ''
        else:
            var_name = f'_{region_name.split("_")[-1]}'

        # Only plot the total JES/JER variations in this script
        if var_name not in ['', '_jer', '_jesTotalUp', '_jesTotalDown']:
            continue

        # Get sumw and sumw2 from respective regions for two samples
        h1_sumw, h1_sumw2 = h1.integrate('region', re.compile(f'^{region1}{analysis_tag}{var_name}$')).values(sumw2=True)[()]
        h2_sumw, h2_sumw2 = h2.integrate('region', re.compile(f'^{region2}{analysis_tag}{var_name}$')).values(sumw2=True)[()]
        
        ratios[var_name] = h1_sumw / h2_sumw 
        
        # Gaussian error propagation
        gaus_error = np.sqrt((h2_sumw*np.sqrt(h1_sumw2))**2 + (h1_sumw*np.sqrt(h2_sumw2))**2)/h2_sumw**2
        err[var_name] = gaus_error

    # Set y-label for either QCD or EWK samples
    sample_label = sample_type.upper()

    # The y-axis labels for each tag
    tag_to_ylabel = {
        'znunu_over_wlnu17' : r'{} $Z\rightarrow \nu \nu$ SR / {} $W\rightarrow \ell \nu$ SR'.format(sample_label, sample_label),
        'znunu_over_wlnu18' : r'{} $Z\rightarrow \nu \nu$ SR / {} $W\rightarrow \ell \nu$ SR'.format(sample_label, sample_label),
        'znunu_over_zmumu17' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow \mu \mu$ CR'.format(sample_label, sample_label),
        'znunu_over_zmumu18' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow \mu \mu$ CR'.format(sample_label, sample_label),
        'znunu_over_zee17' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow ee$ CR'.format(sample_label, sample_label),
        'znunu_over_zee18' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow ee$ CR'.format(sample_label, sample_label),
        'znunu_over_zll17' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow \ell \ell$ CR'.format(sample_label, sample_label),
        'znunu_over_zll18' : r'{} $Z\rightarrow \nu \nu$ SR / {} $Z\rightarrow \ell \ell$ CR'.format(sample_label, sample_label),
        'gjets_over_znunu17' : r'{} $\gamma$ + jets CR / {} $Z\rightarrow \nu \nu$ SR'.format(sample_label, sample_label),
        'gjets_over_znunu18' : r'{} $\gamma$ + jets CR / {} $Z\rightarrow \nu \nu$ SR'.format(sample_label, sample_label),
        'wlnu_over_wenu17' : r'{} $W\rightarrow \ell \nu$ SR / {} $W\rightarrow e\nu$ CR'.format(sample_label, sample_label),
        'wlnu_over_wenu18' : r'{} $W\rightarrow \ell \nu$ SR / {} $W\rightarrow e\nu$ CR'.format(sample_label, sample_label),
        'wlnu_over_wmunu17' : r'{} $W\rightarrow \ell \nu$ SR / {} $W\rightarrow \mu \nu$ CR'.format(sample_label, sample_label),
        'wlnu_over_wmunu18' : r'{} $W\rightarrow \ell \nu$ SR / {} $W\rightarrow \mu \nu$ CR'.format(sample_label, sample_label),
        'wlnu_over_gjets17' : r'{} $W\rightarrow \ell \nu$ SR / {} $\gamma$ + jets CR'.format(sample_label, sample_label),
        'wlnu_over_gjets18' : r'{} $W\rightarrow \ell \nu$ SR / {} $\gamma$ + jets CR'.format(sample_label, sample_label),
    }
    
    edges = h1.axes()[1].edges()
    centers = h1.axes()[1].centers()

    # Get maximum and minimum ratios, fix y-axis limits
    if bin_selection == 'singleBin':
        counts = list(ratios.values())
        lower_ylim = min(counts) * 0.98
        upper_ylim = max(counts) * 1.02

    # Plot the ratios for each variation
    if plot_onlyJES:
        variations = ['', '_jesTotalUp', '_jesTotalDown']
    elif plot_onlyJER:
        variations = ['', '_jer']
    else:
        variations = ['', '_jer', '_jesTotalUp', '_jesTotalDown']
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    for idx, var_name in enumerate(variations):
        ratio_arr = ratios[var_name]
        hep.histplot(ratio_arr, 
                     edges, 
                     label=var_to_legend_label[var_name],
                     ax=ax,
                     histtype='step',
                     yerr=err[var_name]
                     )

        if var_name != '':
            r = ratios[var_name] / ratios['']
            rax.plot(centers, r, 'o', label=var_to_legend_label[var_name], c=colors[idx])
            rax.fill_between(centers, 1-(err['']/ratios['']), 1+(err['']/ratios['']), color='gray', alpha=0.5)

    edges = binning_to_use.edges()

    ax.set_xlim(edges[0], edges[-1])
    if bin_selection == 'singleBin':
        ax.set_ylim(lower_ylim, upper_ylim)
    ax.set_ylabel(tag_to_ylabel[tag])
    ax.legend()

    if '17' in tag:
        ax.text(1., 1., '2017',
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
                )
    
    elif '18' in tag:
        ax.text(1., 1., '2018',
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
                )

    if bin_selection == 'singleBin':
        rax.set_ylim(0.96, 1.04)
    else:
        rax.set_ylim(0.9, 1.1)

    loc = matplotlib.ticker.MultipleLocator(base=0.02)
    rax.yaxis.set_major_locator(loc)
    rax.set_ylabel('Varied / Nominal')
    if analysis == 'vbf':
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    elif analysis == 'monojet':
        rax.set_xlabel(r'Recoil (GeV)')
    rax.legend()
    rax.grid(True)

    # Save the figure
    if plot_onlyJES:
        outdir = f'./output/{out_tag}/onlyJES'
    elif plot_onlyJER:
        outdir = f'./output/{out_tag}/onlyJER'
    else:
        outdir = f'./output/{out_tag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    filename = f'{tag}_{sample_type}_jes_jer_variations_{bin_selection}.pdf'

    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)
    plt.close()

    print(f'MSG% Histogram saved in: {outpath}')

    # Calculate and print the uncertainties
    # for each mjj bin
    get_unc(ratios, edges, out_tag, tag, sample_type)

    # Flatten and return
    for key in ratios.keys():
        ratios[key] = ratios[key][0]

    return ratios

def main():
    args = parse_commandline()
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

    # Determine which processes (QCD/EWK) to run on from the command line option
    run_over_samples = {}
    for proc in ['qcd', 'ewk']:
        run_over_samples[proc] = proc in args.run

    if 'ind' in args.calc:
        for tag, data_dict in dataset_regex.items():
            for sample_type, run in run_over_samples.items():
                if not run:
                    continue
                # If "onlyRun" option is specified, skip the procs which do not match to the regular expression 
                if args.onlyRun:
                    if not re.match(args.onlyRun, tag):
                        print(f'Skipping: {tag}')
                        continue
                    
                title, regex, region = data_dict[sample_type].values()
                plot_jes_jer_var(acc, regex=regex, 
                    title=title, 
                    tag=tag, 
                    out_tag=out_tag, 
                    region=region, 
                    sample_type=sample_type,
                    analysis=args.analysis)

    # All bin types for the ratio plot
    # For now: Just plot single bin (not interested in shape information for TFs)
    bin_types = ['singleBin']

    if 'ratio' in args.calc:
        # Store the ratios and dataset names/types to tabulate values (using pandas) later
        ratio_list = []
        index_list = []
        for tag, data_dict in tag_to_dataset_pairs.items():
            for sample_type, run in run_over_samples.items():
                if not run:
                    continue
                # If "onlyRun" option is specified, skip the procs which do not match to the regular expression 
                if args.onlyRun:
                    if not re.match(args.onlyRun, tag):
                        print(f'Skipping: {tag}')
                        continue
                    
                index_list.append(indices_from_tags[tag][sample_type])
                datapair_dict = data_dict[sample_type] 
                data1_info = datapair_dict['dataset1']
                data2_info = datapair_dict['dataset2']
                for bin_selection in bin_types:
                    ratio_dict = plot_jes_jer_var_ratio( acc, 
                                            regex1=data1_info['regex'], 
                                            regex2=data2_info['regex'], 
                                            region1=data1_info['region'], 
                                            region2=data2_info['region'], 
                                            tag=tag, 
                                            out_tag=out_tag,
                                            sample_type=sample_type,
                                            analysis=args.analysis,
                                            plot_onlyJES=args.onlyJES,
                                            plot_onlyJER=args.onlyJER,
                                            bin_selection=bin_selection)
                    
                    if bin_selection == 'singleBin':
                        ratio_list.append(ratio_dict)
    
    # Create a DataFrame out of ratios
    if args.save_to_df:
        rename_columns = {
            '' : 'Nominal',
            '_jer' : 'JER',
            '_jesTotalUp' : 'JES up',
            '_jesTotalDown' : 'JES down'
        }
        df = pd.DataFrame(ratio_list, index=index_list) 
        df.rename(columns=rename_columns, inplace=True)
        # Save to pkl file
        pkl_file = 'ratios_df.pkl'
        df.to_pickle(pkl_file)

if __name__ == '__main__':
    main()
