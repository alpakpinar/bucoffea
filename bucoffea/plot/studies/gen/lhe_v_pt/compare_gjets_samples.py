#!/usr/bin/env python

import os
import sys
import re
import warnings
import numpy as np
from bucoffea.plot.util import (merge_datasets,
                                merge_extensions,
                                scale_xs_lumi)

from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint
from coffea import hist

pjoin = os.path.join

rebin = {
    'vpt' : hist.Bin('vpt', r'$p_T(V)\ (GeV)$', np.arange(200,1500,50)),
    'mjj' : hist.Bin('mjj', r'$M_{jj}\ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'ak4_pt0' : hist.Bin('jpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,1080,50)) ),
    'ak4_pt1' : hist.Bin('jpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,640,50)) )
}

def compare_two_gjets_samples(acc, samples, outtag, cutlabels, distribution='vpt'):
    '''
    Make a comparison plot for two GJets samples, as a function of the given distribution.
    Distribution will be plotted for different stages in the cutflow.

    Parameter "cutlabels" specifies the point to which cuts are applied as follows:
    cutlabel == "inclusive"   --> No additional VBF cuts on top of DR > 0.4 requirement
    cutlabel == "up_to_<cut>" --> VBF cuts are applied up to (and excluding) <cut> on top of DR > 0.4 requirement
    cutlabel == "all_cuts_applied" --> All VBF cuts are applied on top of DR > 0.4 requirement

    Parameter "cutlabels" must be a list of such cut labels.
    '''
    # Exactly 2 samples must be commpared
    assert len(samples) == 2

    # Extract dataset years
    extract_year = lambda name: name.split('_')[-1]
    years = map(extract_year, samples)

    # Get the distribution
    dist = f'gen_{distribution}_vbf_stat1_withDRreq'

    acc.load(dist)
    h = acc[dist]

    # Rebin, if neccessary
    if distribution in rebin.keys():
        distbin = rebin[distribution]
        h = h.rebin(distbin.name, distbin)

    xaxis = h.axes()[1]
    edges = xaxis.edges(overflow='over')
    centers = xaxis.centers(overflow='over')

    # Dataset merging, rescaling w.r.t. xs and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Mapping from sample names to regular expressions
    # for the relevant dataset names
    sample_to_regex = {s : s.replace(f'{year}',f'.*_{year}') for year, s in zip(years, samples)}

    # Store histograms from each dataset in a dictionary
    histos = {s : h[re.compile(sample_to_regex[s])].integrate('dataset') for s in samples}
    
    # Short legend labels for different datasets
    short_data_labels = {
        'GJets_DR-0p4_HT_2016' : 'DR_2016',
        'GJets_DR-0p4_HT_2017' : 'DR_2017',
        'GJets_HT_2016' : 'NonDR_2016',
        'GJets_HT_2017' : 'NonDR_2017'
    }

    # Plot for all cuts requested
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    for cutlabel in cutlabels:
        for dataset, histo in histos.items():
            h = histo.integrate('cut', cutlabel)
            ax.step(edges[:-1], h.values(overflow='over')[()], where='post', label=f'{short_data_labels[dataset]}_{cutlabel}')

    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(1e0,1e10)
    # Do not include overflow bin
    ax.set_xlim(edges[0], edges[-2])
    ax.set_ylabel('Counts')

    # Store 2016/2017 ratios and uncertainties on the ratios in a dictionary
    ratios_and_uncs = {}

    # Compute the ratios for each cut and the uncertainty on the ratio
    for cutlabel in cutlabels:
        sumw_1, sumw2_1 = histos[samples[0]].integrate('cut', cutlabel).values(overflow='over', sumw2=True)[()]
        sumw_2, sumw2_2 = histos[samples[1]].integrate('cut', cutlabel).values(overflow='over', sumw2=True)[()]
        ratio = sumw_1 / sumw_2
        unc = np.hypot(
            np.sqrt(sumw2_1) / sumw_1,
            np.sqrt(sumw2_2) / sumw_2,
        )

        ratios_and_uncs[cutlabel] = {'ratio' : ratio, 'unc' : unc}

        rax.errorbar(x=centers, y=ratio, yerr=unc, ls='', marker='o', label=cutlabel)
        rax.grid(True)
        rax.set_ylim(0.6, 1.4)
        rax.set_ylabel('2016 / 2017')
        rax.set_xlabel(xaxis.label)
        rax.legend()
    
    # Save figure
    outdir = f'./output/gjets_comparisons/{outtag}/{"_vs_".join(samples)}/{distribution}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    filename = f'comparison_{"_".join(cutlabels)}.pdf'

    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)

    print(f'File saved: {outpath}')

    plt.close()

    # Return dictionary containing ratios and uncertainties on the ratios
    return ratios_and_uncs

def main():
    inpath = sys.argv[1]

    # Get the output tag name for output directory naming
    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    acc = dir_archive(
        inpath,
        serialized=True,
        memsize=1e3,
        compression=0
    )

    acc.load('sumw')
    acc.load('sumw2')

    # Distributions to compare against
    distributions = ['vpt', 
            'mjj', 
            'ak4_pt0', 
            'ak4_pt1',
            'ak4_eta0',
            'ak4_eta1',
            'detajj',
            'dphijj']

    to_compare = [
        ('GJets_DR-0p4_HT_2016', 'GJets_DR-0p4_HT_2017'),
        ('GJets_HT_2016', 'GJets_HT_2017'),
        ('GJets_HT_2016', 'GJets_DR-0p4_HT_2017'),
        ('GJets_HT_2017', 'GJets_DR-0p4_HT_2017'),
        ('GJets_HT_2016', 'GJets_HT_2017', 'GJets_DR-0p4_HT_2017')
    ]

    # Cut labels for different points in cutflow
    cutlabels_list = [
        ('inclusive', 'up_to_leadak4_pt_eta', 'up_to_trailak4_pt_eta'),
        ('inclusive', 'up_to_hemisphere', 'up_to_mindphijr'),
        ('inclusive', 'up_to_detajj', 'up_to_dphijj'),
        ('inclusive', 'all_cuts_applied')
    ]

    ratios_and_uncs = {}

    # Compare GJets DR samples from 2016 and 2017
    # Ignore runtime warnings (due to invalid ratio) for now
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for distribution in distributions:
            ratios_and_uncs[distribution] = {}
            for cutlabels in cutlabels_list:
                d = compare_two_gjets_samples(acc, distribution=distribution, samples=to_compare[0], outtag=outtag, cutlabels=cutlabels)
                ratios_and_uncs[distribution].update(d)

if __name__ == '__main__':
    main()


    

    





