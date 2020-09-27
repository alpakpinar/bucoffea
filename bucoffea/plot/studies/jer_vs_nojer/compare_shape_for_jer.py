#!/usr/bin/env python

import os
import sys
import re
import warnings
import argparse

from collections import OrderedDict
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

binnings = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'recoil' : hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016),
    'recoil_relaxed' : hist.Bin('recoil','Recoil (GeV)', list(range(160,250,30)) + recoil_bins_2016),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) )
}

# Aesthetics for plotting
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset to be looked at, default is znunu.', default='znunu')
    parser.add_argument('--relaxed_recoil', help='Look at region with MET > 150 GeV cut.', action='store_true')
    args = parser.parse_args()
    return args

def get_label_for_tag(tag):
    mapping = {
        'noSmear' : 'No JER',
        '09Jun20v7' : 'MiniAOD-like JER',
        '21Sep20v7' : 'NanoAOD-like JER'
    }
    return mapping[tag]

def do_rebinning(h, variable, relaxed_recoil=False):
    if variable == 'mjj':
        h = h.rebin('mjj', binnings['mjj'])
    elif variable == 'recoil':
        if relaxed_recoil:
            h = h.rebin('recoil', binnings['recoil_relaxed'])
        else:
            h = h.rebin('recoil', binnings['recoil'])
    elif 'ak4_pt' in variable:
        h = h.rebin('jetpt', binnings[variable])

    return h

def preprocess(h, acc, variable, year, dataset, relaxed_recoil=False):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    dataset_to_regex = {
        'zjets' : f'ZJetsToNuNu.*{year}',
        'vbf' : f'VBF_HToInv.*M125.*{year}',
    }
    
    if relaxed_recoil:
        h = h.integrate('region', 'sr_vbf_relaxed_recoil').integrate('dataset', re.compile(dataset_to_regex[dataset]))
    else:
        h = h.integrate('region', 'sr_vbf').integrate('dataset', re.compile(dataset_to_regex[dataset]))

    if variable in binnings.keys():
        h = do_rebinning(h, variable, relaxed_recoil)

    return h

def compare_shapes(acc_dict, variable='mjj', year=2017, dataset='zjets', relaxed_recoil=False):
    h_dict = OrderedDict()
    for tag, acc in acc_dict.items():
        acc.load(variable)
        # Get the pre-processed histograms
        h_dict[tag] = preprocess(acc[variable], acc, variable, year, dataset, relaxed_recoil)

    # Ready to plot! Plot the comparison of the three distributions
    legend_labels = []
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    for tag, h in h_dict.items():
        hist.plot1d(h, ax=ax, overflow='over', clear=False)
        legend_labels.append(get_label_for_tag(tag))

    ax.legend(labels=legend_labels)

    dataset_to_title = {
        'zjets' : r'QCD $Z(\nu\nu)$',
        'vbf' : r'VBF $H(inv)$'
    }

    ax.set_title(dataset_to_title[dataset])
    # if variable in ['mjj', 'recoil']:
    if variable == 'mjj':
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e6)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    # Plot ratio to "no JER" case
    for idx, tag in enumerate(['09Jun20v7', '21Sep20v7']):
        # Just a hack to keep the colors consistent
        data_err_opts['color'] = colors[idx+1]
        hist.plotratio(h_dict[tag], h_dict['noSmear'], ax=rax, 
                        unc='num', error_opts=data_err_opts, 
                        label=get_label_for_tag(tag), clear=False
                        )

    rax.grid(True)
    if variable != 'ak4_eta1':
        rax.set_ylim(0.5,1.5)
    else:
        rax.set_ylim(0,2)
    rax.set_ylabel('Ratio to no JER')
    rax.legend()
    
    if variable == 'recoil' and relaxed_recoil:
        ylim = ax.get_ylim()
        ax.plot([250, 250], ylim, color='black')
        ax.set_ylim(ylim)

        ylim = rax.get_ylim()
        rax.plot([250, 250], ylim, color='black')
        rax.set_ylim(ylim)

    # Save figure
    if relaxed_recoil:
        outdir = f'./output/three_jer_comparison/{dataset}/relaxed_recoil'
    else:
        outdir = f'./output/three_jer_comparison/{dataset}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}_{year}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    '''Compare Z(vv) distributions for three cases:
    1. No JER smearing applied
    2. MiniAOD-like JER smearing applied (09Jun20v7 skim)
    3. NanoAOD-like JER smearing applied (21Sep20v7 skim)
    '''
    # Read the dataset as a command line argument
    args = parse_cli()
    
    inpath_noSmear = bucoffea_path('./submission/merged_2020-09-17_vbfhinv_noJER_nanoAODv7_deepTau')
    inpath_09Jun20v7 = bucoffea_path('./submission/merged_2020-09-18_vbfhinv_withJER_nanoAODv7_deepTau')
    inpath_21Sep20v7 = bucoffea_path('./submission/merged_2020-09-22_vbfhinv_znunu_vbf_2017_21Sep20v7')
    # inpath_noSmear = bucoffea_path('./submission/merged_2020-09-24_vbfhinv_znunu_09Jun20v7_noJER_relaxed_recoil')
    # inpath_09Jun20v7 = bucoffea_path('./submission/merged_2020-09-24_vbfhinv_znunu_09Jun20v7_relaxed_recoil')
    # inpath_21Sep20v7 = bucoffea_path('./submission/merged_2020-09-24_vbfhinv_znunu_relaxed_recoil')

    acc_dict = {
        'noSmear' : dir_archive(inpath_noSmear),
        '09Jun20v7' : dir_archive(inpath_09Jun20v7),
        '21Sep20v7' : dir_archive(inpath_21Sep20v7)
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    variables = ['mjj', 'ak4_pt0', 'ak4_pt1', 'ak4_eta0', 'ak4_eta1', 'recoil']
    for variable in variables:
        compare_shapes(acc_dict, dataset=args.dataset, variable=variable, relaxed_recoil=args.relaxed_recoil)

if __name__ == '__main__':
    main()