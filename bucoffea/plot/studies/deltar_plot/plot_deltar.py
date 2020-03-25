#!/usr/bin/env python

import os
import sys
import re
import warnings
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from pprint import pprint
from klepto.archives import dir_archive

pjoin = os.path.join

def plot_deltar_dist(acc, regex, tag, outtag):
    '''
    Given the input accumulator, plot the LHE-level deltaR distribution between
    photons and partons.
    ==================
    PARAMETERS:
    ==================
    acc    : Input accumulator containing all the histograms.
    regex  : The regular expression matching the dataset name.
    tag    : Tag for the process.
    outtag : Out tag for naming the output directory.
    '''
    pt_type = 'stat1'
    dist = f'lhe_mindr_g_parton_{pt_type}'
    acc.load(dist)
    h = acc[dist]

    # Rebin deltaR axis
    new_bin = hist.Bin('dr', r'$\Delta R_{\gamma, j}$', 50, 0, 5)
    h = h.rebin('dr', new_bin)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Pick the relevant dataset(s)
    h = h[re.compile(regex)].integrate('dataset')

    # Plot the histogram
    fig, ax = plt.subplots(1,1)
    hist.plot1d(h, ax=ax)

    dataset_name = regex.replace('.*', '_')
    handle, _ = ax.get_legend_handles_labels()
    handle[0].set_label(dataset_name)
    ax.legend()

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{tag}_deltar.pdf')
    fig.savefig(outpath)
    print(f'Figure saved: {outpath}')

    # Return the histogram values (sumw and sumw2) and bin edges
    vals =  h.values(overflow='over', sumw2=True)[()]
    tup = ( h.axis('dr').edges(overflow='over'), vals[0], vals[1])
    return tup

def plot_comparison(vals, datasets, tag, outtag):
    '''
    Given the individual deltaR distributions and two datasets, create the
    deltaR comparison plot for these two datasets.
    =============
    PARAMETERS:
    =============
    vals     : Dictionary containing deltaR distributions as arrays for each dataset.
    datasets : Tags for the datasets to be compared.
    tag      : Tag for the output file.
    outtag   : Tag for the output directory.
    '''
    pretty_label = {
        'gjets_dr_2016'  : 'GJets_DR-0p4_HT_2016',
        'gjets_dr_2017'  : 'GJets_DR-0p4_HT_2017',
        'gjets_ht_2016'  : 'GJets_HT_2016',
        'gjets_ht_2017'  : 'GJets_HT_2017',
        'gjets_nlo_2016' : 'G1Jet_Pt_amcatnlo_2016'
    }

    # Number of datasets to compare 
    ndatasets = len(datasets)

    # Plot ratio pad only if comparing two datasets
    plot_ratio = (ndatasets == 2)

    # Plot the comparison
    rax = None
    if plot_ratio:
        fig, axes = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3,1)}, sharex=True)
        ax, rax = axes
    else:
        fig, ax = plt.subplots(1,1)

    for data in datasets:
        edges, sumw, _ = vals[data]
        ax.step(edges[:-1],sumw,where='post',label=pretty_label[data])
    
    # Do not include the overflow bin
    ax.set_xlim(edges[0], edges[-2])
    ax.set_ylim(0,1e5)
    ax.set_xlabel(r'$\Delta R_{\gamma,j}$')
    ax.set_ylabel('Counts')
    ax.legend()

    ax_ylim = ax.get_ylim()
    # ax.set_xlim(0,2)
    ax.plot([0.4,0.4], ax_ylim, 'r')
    ax.set_ylim(ax_ylim)

    # Calculate and plot ratios in the ratio pad if needed
    if rax:
        sumw = {}
        sumw2 = {}
        for dataset in datasets:
            sumw[dataset], sumw2[dataset] = vals[dataset][1:] 
    
        def calc_unc(*datasets):
            '''Calculate uncertainty on the ratio of two datasets.'''
            data1, data2 = datasets
            return np.hypot(
                np.sqrt(sumw2[data1])/sumw[data1],
                np.sqrt(sumw2[data2])/sumw[data2]
            )
    
        ratio = sumw[datasets[0]] / sumw[datasets[1]]
        unc = calc_unc(*datasets)
        centers = (( edges + np.roll(edges,-1) ) / 2 )[:-1]
    
        # Plot ratio pad 
        rax.errorbar(x=centers, y=ratio, yerr=unc, marker='o', ls='', color='k')
        ax.set_xlabel('')
        rax.set_xlabel(r'$\Delta R_{\gamma,j}$')
        rax.grid(True)
        rax.set_ylim(0.8,1.2)
        rax.set_ylabel('Ratios')
    
        rax_ylim = rax.get_ylim()
        rax.plot([0.4,0.4], rax_ylim, 'r--')
        rax.set_ylim(rax_ylim)
    
        rax.plot(rax.get_xlim(), [1., 1.])
    
    # Save the figure
    outdir = f'./output/{outtag}/comparisons'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{tag}_dr_comparison.pdf')
    fig.savefig(outpath)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
        inpath,
        memsize=1e3
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    tag_regex = {
        'gjets_dr_2016'  : 'GJets_DR-0p4_HT.*2016',
        'gjets_dr_2017'  : 'GJets_DR-0p4_HT.*2017',
        'gjets_ht_2016'  : 'GJets_HT.*2016',
        'gjets_ht_2017'  : 'GJets_HT.*2017',
        'gjets_nlo_2016' : 'G1Jet_Pt-amcatnlo.*2016'
    }

    vals = {}

    # Get individual deltaR distributions for each dataset
    for tag, regex in tag_regex.items():
        vals[tag] = plot_deltar_dist(acc,regex=regex,tag=tag,outtag=outtag)

    # Get the comparison plots
    comparisons = {
        'gjets_dr_16_VS_gjets_ht_16'  : ['gjets_dr_2016', 'gjets_ht_2016'],
        'gjets_dr_16_VS_gjets_dr_17'  : ['gjets_dr_2016', 'gjets_dr_2017'],
        'gjets_dr_17_VS_gjets_ht_17'  : ['gjets_dr_2017', 'gjets_ht_2017'],
        'all' : ['gjets_dr_2016', 'gjets_dr_2017', 'gjets_ht_2016', 'gjets_ht_2017']
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tag, datasets in comparisons.items():
            plot_comparison(vals, datasets=datasets,tag=tag,outtag=outtag)

if __name__ == '__main__':
    main()