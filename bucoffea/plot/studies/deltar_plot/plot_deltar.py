#!/usr/bin/env python

import os
import sys
import re
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
    datasets : Tag for the datasets to be compared (there should be 2 or 3 datasets).
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

    # Labels for ratio legends
    pretty_label_ratio = {
        'gjets_dr_2017 / gjets_ht_2017' : 'DR 2017 / Non-DR 2017',
        'gjets_dr_2017 / gjets_ht_2016' : 'DR 2017 / Non-DR 2016',
        'gjets_dr_2016 / gjets_ht_2017' : 'DR 2016 / Non-DR 2017',
        'gjets_dr_2016 / gjets_dr_2017' : 'DR 2016 / DR 2017',
        'gjets_dr_2016 / gjets_ht_2016' : 'DR 2016 / Non-DR 2016',
        'gjets_ht_2017 / gjets_ht_2016' : 'Non-DR 2017 / Non-DR 2016',
        'gjets_nlo_2016 / gjets_ht_2017' : 'NLO 2016 / Non-DR 2017',
        'gjets_nlo_2016 / gjets_ht_2016' : 'NLO 2016 / Non-DR 2016',
        'gjets_nlo_2016 / gjets_dr_2017' : 'NLO 2016 / DR 2017',
        'gjets_nlo_2016 / gjets_dr_2016' : 'NLO 2016 / DR 2016',
    }

    # Number of datasets to compare (should be 2 or 3)
    ndatasets = len(datasets)

    assert ndatasets == 2 or ndatasets == 3

    # Plot the comparison
    num_ratiopads = int( (ndatasets*(ndatasets-1)) / 2 )
    height_ratios = [3] + [1]*(num_ratiopads)
    figsize = (7,10) if num_ratiopads > 1 else (7,7)
    fig, axes = plt.subplots(num_ratiopads+1, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios}, sharex=True)
    ax = axes[0] # Main plot
    rax = axes[1:] # Ratio pad(s)
    for data in datasets:
        edges, sumw, _ = vals[data]
        ax.step(edges[:-1],sumw,where='post',label=pretty_label[data])
    
    # Do not include the overflow bin
    ax.set_xlim(edges[0], edges[-2])
    ax.set_ylim(0,1e5)
    ax.set_ylabel('Counts')
    ax.legend()

    ax_ylim = ax.get_ylim()
    ax.set_xlim(0,2)
    ax.plot([0.4,0.4], ax_ylim, 'r')
    ax.set_ylim(ax_ylim)

    # Calculate and plot ratios in the ratio pad
    sumw = {}
    sumw2 = {}
    for dataset in datasets:
        sumw[dataset], sumw2[dataset] = vals[dataset][1:] 

    # Ratios between the sum of weights for each dataset
    ratios = {}
    # Uncertainties on each ratio (Gaussian error propagation)
    uncs = {}
    def calc_unc(data1, data2):
        '''Calculate uncertainty on the ratio of two datasets.'''
        return np.hypot(
            np.sqrt(sumw2[data1])/sumw[data1],
            np.sqrt(sumw2[data2])/sumw[data2]
        )

    for idx1 in range(len(datasets)):
        data1 = datasets[idx1]
        for idx2 in range(idx1+1, len(datasets)):
            data2 = datasets[idx2]
            ratios[f'{data1} / {data2}'] = sumw[data1] / sumw[data2]
            uncs[f'{data1} / {data2}'] = calc_unc(data1, data2)

    centers = (( edges + np.roll(edges,-1) ) / 2 )[:-1]

    # Plot ratio pads for each dataset comparison
    for idx, (key, ratio) in enumerate(ratios.items()):
        unc = uncs[key]
        rax[idx].errorbar(x=centers, y=ratio, yerr=unc, marker='o', ls='', color='k', label=pretty_label_ratio[key])
        if len(rax) > 1:
            rax[idx].legend()
        rax[idx].set_xlabel(r'$\Delta R_{\gamma,j}$')
        rax[idx].grid(True)
        rax[idx].set_ylim(0.8,1.2)
        rax[idx].set_ylabel('Ratios')
    
        rax_ylim = rax[idx].get_ylim()
        rax[idx].plot([0.4,0.4], rax_ylim, 'r--')
        rax[idx].set_ylim(rax_ylim)
    
        rax[idx].plot(rax[idx].get_xlim(), [1., 1.])

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
        'gjets_nlo_16_VS_gjets_ht_17' : ['gjets_nlo_2016', 'gjets_ht_2017'],
        'gjets_nlo_16_VS_gjets_dr_16' : ['gjets_nlo_2016', 'gjets_dr_2016'],
        'gjets_nlo_16_VS_gjets_dr_17' : ['gjets_nlo_2016', 'gjets_dr_2017'],
        'gjets_dr_17_VS_gjets_ht_17_VS_gjets_ht_16' : ['gjets_dr_2017', 'gjets_ht_2017', 'gjets_ht_2016']
    }

    for tag, datasets in comparisons.items():
        plot_comparison(vals, datasets=datasets,tag=tag,outtag=outtag)

if __name__ == '__main__':
    main()