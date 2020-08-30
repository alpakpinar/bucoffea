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
import uproot
from pprint import pprint
from itertools import chain
from tabulate import tabulate
from collections import OrderedDict

pjoin = os.path.join

titles = {
    'GluGlu2017' : r'$ggH(inv) \ 2017$ split JEC uncertainties',
    'GluGlu2018' : r'$ggH(inv) \ 2018$ split JEC uncertainties',
    'VBF2017' : r'$VBF \ H(inv) \ 2017$ split JEC uncertainties',
    'VBF2018' : r'$VBF \ H(inv) \ 2018$ split JEC uncertainties',
    'ZJetsToNuNu2016' : r'$Z(\nu\nu) \ 2016$ split JEC uncertainties',
    'ZJetsToNuNu2017' : r'$Z(\nu\nu) \ 2017$ split JEC uncertainties',
    'ZJetsToNuNu2018' : r'$Z(\nu\nu) \ 2018$ split JEC uncertainties',
    'WJetsToLNu2017' : r'$W(\ell\nu) \ 2017$ split JEC uncertainties',
    'WJetsToLNu2018' : r'$W(\ell\nu) \ 2018$ split JEC uncertainties'
}

titles_two_nuisances = {
    'GluGlu2017' : r'$ggH(inv) \ 2017$ corr vs uncorr JEC uncertainties',
    'GluGlu2018' : r'$ggH(inv) \ 2018$ corr vs uncorr JEC uncertainties',
    'VBF2018' : r'$VBF \ H(inv) \ 2018$ corr vs uncorr JEC uncertainties',
    'VBF2017' : r'$VBF \ H(inv) \ 2017$ corr vs uncorr JEC uncertainties',
    'ZJetsToNuNu2016' : r'$Z(\nu\nu) \ 2016$ corr vs uncorr JEC uncertainties',
    'ZJetsToNuNu2017' : r'$Z(\nu\nu) \ 2017$ corr vs uncorr JEC uncertainties',
    'ZJetsToNuNu2018' : r'$Z(\nu\nu) \ 2018$ corr vs uncorr JEC uncertainties'
}

# Define all possible binnings for all variables in this dictionary
mjj_binning_v1 = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
mjj_binning_single_bin = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200,3500])

recoil_bins_2016_monoj = hist.Bin('recoil', 'Recoil (GeV)', [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400])
recoil_bins_2016_monov = hist.Bin('recoil', 'Recoil (GeV)', [250,300,350,400,500,600,750,1000])

met_binning_v1_2016 = hist.Bin('recoil', 'Recoil (GeV)', list(range(0,500,50)) + list(range(500,1100,100))) 
met_binning_v2_2016 = hist.Bin('recoil', 'Recoil (GeV)', [250,275,300,350,400,450,500,650,800,1150,1500]) 
met_binning_v1_2017 = hist.Bin('recoil', 'Recoil (GeV)', list(range(250,550,100)) + list(range(550,1300,250))) 
met_binning_single_bin = hist.Bin('recoil', 'Recoil (GeV)', [250,1500])
met_binning_coarse = hist.Bin('recoil', 'Recoil (GeV)', [250,300,400,500,800,1500])


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--analysis', help='The analysis being considered, default is vbf.', default='vbf')
    parser.add_argument('--regroup', help='Construct the uncertainty plot with the sources grouped into correlated and uncorrelated.', action='store_true')
    parser.add_argument('--onlyRun', help='Specify regex to only run over some transfer factors and skip others.')
    parser.add_argument('--save_to_root', help='Save output uncertaintes to a root file.', action='store_true')
    parser.add_argument('--tabulate', help='Tabulate unc/variation values.', action='store_true')
    parser.add_argument('--znunu2016', help='Only run over Z(vv) 2016.', action='store_true')
    parser.add_argument('--use_monov_binning', help='If this option is specified, use mono-V analysis binning for MET.', action='store_true')
    parser.add_argument('--use_monoj_binning', help='If this option is specified, use monojet analysis binning for MET.', action='store_true')
    args = parser.parse_args()
    return args

# Bin selection should be one of the following:
# Coarse, single bin, initial
def plot_split_jecunc(acc, out_tag, dataset_tag, year, binnings, plot_total=True, skimmed=True, 
            bin_selection='initial', analysis='vbf', root_config={'save': False, 'file': None}, 
            tabulate_top5=False, use_monoj_binning=False, use_monov_binning=False
            ):
    '''Plot all split JEC uncertainties on the same plot.'''
    # Load the relevant variable to analysis, select binning
    variable_to_use = 'mjj' if analysis == 'vbf' else 'recoil'
    acc.load(variable_to_use)
    h = acc[variable_to_use]

    # Rebin the histogram
    new_bins = binnings[variable_to_use][bin_selection][year]
    h = h.rebin(variable_to_use , new_bins)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Determine the region to take the data from
    if re.match('GluGlu|VBF|ZJets|WJets.*', dataset_tag):
        region_to_use = 'sr'
    else:
        raise NotImplementedError('Not implemented for this region yet.')
    region_suffix = '_j' if analysis == 'monojet' else '_vbf'

    dataset_name = dataset_tag.replace(f'{year}', '')
    h = h.integrate('dataset', re.compile(f'{dataset_name}.*{year}'))[re.compile(f'{region_to_use}{region_suffix}.*')]

    h_nom = h.integrate('region', f'{region_to_use}{region_suffix}')
    
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

    # If requested, store the top 5 uncertainty sources (as well as the total) and dump them into a table
    if tabulate_top5:
        uncs = {}
        varied = {}

    for region in h.identifiers('region'):
        if region.name == f'{region_to_use}{region_suffix}':
            continue
        if not plot_total:
            if "Total" in region.name:
                continue

        h_var = h.integrate('region', region)
        var_label = region.name.replace(f'{region_to_use}{region_suffix}_', '')
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)
        if skimmed:
            if var_label_skimmed not in vars_to_look_at:
                continue
        # Do not plot JER for now
        if not "jer" in region.name:
            hist.plotratio(h_var, h_nom, ax=ax, clear=False, label=var_label, unc='num',  guide_opts={}, error_opts=data_err_opts)

        # As we loop through each uncertainty source, save into ROOT file if this is requested
        if root_config['save']:
            rootfile = root_config['file']
            if not "jer" in region.name:
                ratio = h_var.values()[()] / h_nom.values()[()]
                edges = h_nom.axis(variable_to_use).edges()
                # Guard against inf/nan values
                ratio[np.isnan(ratio) | np.isinf(ratio)] = 1.
                rootfile[f'{dataset_tag}_{var_label}'] = (ratio, edges)
            else:
                # Symmetric JER up/down variation calculation 
                ratio_jerUp = h_var.values()[()] / h_nom.values()[()]
                ratio_jerDown = 2 - ratio_jerUp

                ratio_jerUp[np.isnan(ratio_jerUp) | np.isinf(ratio_jerUp)] = 1.
                ratio_jerDown[np.isnan(ratio_jerDown) | np.isinf(ratio_jerDown)] = 1.
                
                edges = h_nom.axis(variable_to_use).edges()

                rootfile[f'{dataset_tag}_jerUp'] = (ratio_jerUp, edges)
                rootfile[f'{dataset_tag}_jerDown'] = (ratio_jerDown, edges)

        # Store all uncertainties, later to be tabulated (top 5 only + total)
        if tabulate_top5:
            ratio = h_var.values()[()] / h_nom.values()[()]
            uncs[var_label] = np.abs(ratio - 1)*100
            varied[var_label] = h_var.values()[()]

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
            if 'ZJets' in dataset_tag or 'WJets' in dataset_tag:
                ax.set_ylim(0.65,1.45)
            else:
                ax.set_ylim(0.75,1.35)
            loc = matplotlib.ticker.MultipleLocator(base=0.05)
        
        ax.yaxis.set_major_locator(loc)
        
    ax.set_title(titles[dataset_tag])
    ax.grid(True)

    # Save figure
    if use_monoj_binning:
        outdir = f'./output/{out_tag}/splitJEC/{analysis}/monojet_binning'
    elif use_monov_binning:
        outdir = f'./output/{out_tag}/splitJEC/{analysis}/monoV_binning'
    else:
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
    print(f'MSG% File saved: {outfile}')
    plt.close()

    # Save the top 5 uncertainties in a table under "./tables" directory
    # Save the nominal value
    nom_value = h_nom.values()[()]
    if tabulate_top5:
        print('MSG% Sorting the uncertainties and tabulating top 5.')
        sorted_uncs = OrderedDict()
        for k, v in sorted(uncs.items(), reverse=True, key=lambda item: item[1]):
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
                table.append([label, nom_value[0], var_up[0], var_down[0], unc_up[0], unc_down[0]])

        outdir = f'./output/{out_tag}/splitJEC/{analysis}/tables'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outfile = pjoin(outdir, f'{dataset_tag}_top_five_uncs.txt')

        with open(outfile, 'w+') as f:
            f.write('='*10 + '\n')
            f.write(dataset_tag + '\n')
            f.write('='*10 + '\n')
            t = tabulate(table, floatfmt='.3f', headers='firstrow')
            f.write(t)

        print(f'MSG% Table saved at: {outfile}')

def plot_split_jecunc_regrouped(acc, out_tag, dataset_tag, year, binnings, bin_selection='initial', analysis='vbf'):
    '''Plot split JEC uncertainties regrouped into two:
    1. Correlated across years
    2. Uncorrelated across years
    '''
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

    # All the uncertainty sources: Classified as "uncorrelated", "correlated" and "partially correlated"
    unc_sources = {
        'uncorrelated' : [f'jesAbsolute_{year}', f'jesBBEC1_{year}', f'jesEC2_{year}', f'jesHF_{year}', f'jesRelativeSample_{year}'],
        'correlated' : ['jesFlavorQCD', 'jesAbsolute'],
        'partially correlated' : ['jesHF', 'jesBBEC1', 'jesEC2', 'jesRelativeBal']
    }
    
    # Group all the uncertainty sources, at the end we'll have four variations:
    # CorrelatedUp, CorrelatedDown, UncorrelatedUp, UncorrelatedDown
    regrouped_variations = {}
    regrouped_variation_errors = {}
    # Initialize the two nuisances and the uncertainties on them as all zeros
    for nuisance in ['uncorrelatedUp', 'uncorrelatedDown', 'correlatedUp', 'correlatedDown']:
        regrouped_variations[nuisance] = np.zeros_like(h_nom.values()[()])
    for nuisance in ['uncorrelatedUp', 'uncorrelatedDown', 'correlatedUp', 'correlatedDown']:
        regrouped_variation_errors[nuisance] = np.zeros((2, len(h_nom.values()[()]) ))
    
    # ================================================
    # Regroup the variations into two and calculate combined variations + uncertainties 
    # ================================================
    for region in h.identifiers('region'):
        if region.name == f'sr{region_suffix}' or 'Total' in region.name:
            continue
        h_var = h.integrate('region', region)
        sumw_nom, sumw2_nom = h_nom.values(sumw2=True)[()] 
        sumw_var, sumw2_var = h_var.values(sumw2=True)[()] 

        # Variation for this uncertainty source: Varied / nominal - 1
        variation = sumw_var / sumw_nom - 1
        r_sumw = sumw_var / sumw_nom
        
        # Calculate the errors on this ratio using coffea hist tools
        # Imitating the coffea uncertainty calculation here: 
        # https://github.com/CoffeaTeam/coffea/blob/78534edcc16dbabe961e7b93b57a6c9476d7c6c3/coffea/hist/plot.py#L362
        
        variation_err = np.abs(hist.poisson_interval(r_sumw, sumw2_nom/sumw_var**2) - r_sumw)
        var_label = region.name.replace(f'sr{region_suffix}_', '')
        var_label_skimmed = re.sub('(Up|Down)', '', var_label)

        # Do the grouping of all sources into two 
        if var_label_skimmed in unc_sources['uncorrelated']:
            if 'Up' in var_label:
                regrouped_variations['uncorrelatedUp'] += variation**2
                regrouped_variation_errors['uncorrelatedUp'] += variation_err**2
            elif 'Down' in var_label:
                regrouped_variations['uncorrelatedDown'] += variation**2
                regrouped_variation_errors['uncorrelatedDown'] += variation_err**2
        
        elif var_label_skimmed in unc_sources['correlated']:
            if 'Up' in var_label:
                regrouped_variations['correlatedUp'] += variation**2
                regrouped_variation_errors['correlatedUp'] += variation_err**2
            elif 'Down' in var_label:
                regrouped_variations['correlatedDown'] += variation**2
                regrouped_variation_errors['correlatedDown'] += variation_err**2

        elif var_label_skimmed in unc_sources['partially correlated']:
            # If the source is partially (50%) correlated, put half of the variation in the correlated
            # collection, and the other half to uncorrelated one
            if 'Up' in var_label:
                regrouped_variations['correlatedUp'] += variation**2 / 2
                regrouped_variation_errors['correlatedUp'] += variation_err**2 / 2
                regrouped_variations['uncorrelatedUp'] += variation**2 / 2
                regrouped_variation_errors['uncorrelatedUp'] += variation_err**2 / 2
            elif 'Down' in var_label:
                regrouped_variations['correlatedDown'] += variation**2 / 2
                regrouped_variation_errors['correlatedDown'] += variation_err**2 / 2
                regrouped_variations['uncorrelatedDown'] += variation**2 / 2
                regrouped_variation_errors['uncorrelatedDown'] += variation_err**2 / 2

        else:
            raise ValueError(f'Unrecognized uncertainty source: {var_label_skimmed}')

    # Take the square root to calculate the combined variations, add 1 to each of them so that 
    # values are all centered around 1, instead of 0
    for nuisance, variation in regrouped_variations.items():
        if 'Up' in nuisance:
            regrouped_variations[nuisance] = 1 + np.sqrt(variation)
        elif 'Down' in nuisance:
            regrouped_variations[nuisance] = 1 - np.sqrt(variation) 

    for nuisance, err in regrouped_variation_errors.items():
        regrouped_variation_errors[nuisance] = np.sqrt(err)

    # ================================================
    # Now, plot the regrouped sources in the same plot
    # ================================================
    fig, ax = plt.subplots()
    centers = h_nom.axis(variable_to_use).centers()
    for nuisance, variation in regrouped_variations.items():
        ax.errorbar(centers, variation, yerr=regrouped_variation_errors[nuisance], marker='o', label=nuisance)
    ax.legend()
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel('JEC uncertainty')
    ax.grid(True)

    xlim = ax.get_xlim()
    ax.plot(xlim, [1.0, 1.0], 'k--')
    ax.set_xlim(xlim)
    ax.set_title(titles_two_nuisances[dataset_tag])

    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    ax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{out_tag}/splitJEC/{analysis}/two_nuisances'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    filename = f'{dataset_tag}_splitJEC.pdf'
    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)

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

    # Create an output root file to save the uncertainties
    if args.analysis == 'monojet' and args.use_monoj_binning:
        outputrootdir = f'./output/{out_tag}/splitJEC/{args.analysis}/root/monojet_binning'
    elif args.analysis == 'monojet' and args.use_monov_binning:
        outputrootdir = f'./output/{out_tag}/splitJEC/{args.analysis}/root/monoV_binning'
    else:
        outputrootdir = f'./output/{out_tag}/splitJEC/{args.analysis}/root'
        
    if not os.path.exists(outputrootdir):
        os.makedirs(outputrootdir)

    outputrootfile = pjoin(outputrootdir, f'{args.analysis}_shape_jes_uncs.root')
    rootfile = uproot.recreate(outputrootfile)
    print(f'MSG% ROOT file is created: {rootfile}')

    # Configure root usage for the function 
    root_config = {
        'save' : args.save_to_root,
        'file' : rootfile if args.save_to_root else None
    }

    # If requested so, only run over Z(nunu) 2016
    if not args.znunu2016:
        dataset_tags = ['ZJetsToNuNu2017', 'ZJetsToNuNu2018', 'WJetsToLNu2017', 'WJetsToLNu2018', 'VBF2017', 'VBF2018', 'GluGlu2017', 'GluGlu2018']
    else:
        dataset_tags = ['ZJetsToNuNu2016']

    # Figure out the binnings to be used
    binnings = {
        'recoil' : {
            'initial' : {'2016' : met_binning_v2_2016, '2017': met_binning_v1_2017, '2018' : met_binning_v1_2017},
            'single bin' : {'2016' : met_binning_single_bin, '2017' : met_binning_single_bin, '2018' : met_binning_single_bin},
            'coarse' : {'2016' : met_binning_coarse, '2017' : met_binning_coarse, '2018': met_binning_coarse}
        },
        'mjj' : {
            'initial' : {'2016' : mjj_binning_v1, '2017': mjj_binning_v1, '2018': mjj_binning_v1},
            'single bin' : {'2016': mjj_binning_single_bin, '2017': mjj_binning_single_bin, '2018': mjj_binning_single_bin}
        }
    }

    if args.analysis == 'monojet' and args.use_monoj_binning:
        for year in binnings['recoil']['initial'].keys():
            binnings['recoil']['initial'][year] = recoil_bins_2016_monoj

    elif args.analysis == 'monojet' and args.use_monov_binning:
        for year in binnings['recoil']['initial'].keys():
            binnings['recoil']['initial'][year] = recoil_bins_2016_monov

    for dataset_tag in dataset_tags:
        # If specified, run only on specified processes, skip otherwise
        if args.onlyRun:
            if not re.match(args.onlyRun, dataset_tag):
                print(f'MSG% Skipping: {dataset_tag}')
                continue

        print(f'MSG% Working on: {dataset_tag}')
        # Determine the year of the dataset
        if '2017' in dataset_tag:
            year = '2017' # 2017 VBF + ggH signals or Z(nunu)
        elif '2018' in dataset_tag:
            year = '2018' # 2018 VBF + ggH signals or Z(nunu)
        else:
            year = '2016' # 2016 Z(nunu)
        # Plot split JEC uncertainties in three ways: 
        # 1. All uncertainty sources plotted on a single bin
        # 2. Only the largest sources are plotted, with multiple bins
        # 3. Plot all the sources and save them into a ROOT file as a shape uncertainty
        plot_split_jecunc(acc, out_tag, dataset_tag, year, binnings, 
                    analysis=args.analysis, skimmed=False, bin_selection='single bin', tabulate_top5=args.tabulate,
                    use_monoj_binning=args.use_monoj_binning, use_monov_binning=args.use_monov_binning
                    )
        plot_split_jecunc(acc, out_tag, dataset_tag, year, binnings, 
                    analysis=args.analysis, skimmed=True, bin_selection='initial',
                    use_monoj_binning=args.use_monoj_binning, use_monov_binning=args.use_monov_binning
                    )
        # Only save to ROOT file the unskimmed shapes
        plot_split_jecunc(acc, out_tag, dataset_tag, year, binnings, 
                    analysis=args.analysis, skimmed=False, bin_selection='initial', root_config=root_config,
                    use_monoj_binning=args.use_monoj_binning, use_monov_binning=args.use_monov_binning
                    )
    
        # Produce the plots with regrouping, if requested:
        if args.regroup:
            plot_split_jecunc_regrouped(acc, out_tag, dataset_tag, year, binnings, bin_selection='initial', analysis=args.analysis)

if __name__ == '__main__':
    main()