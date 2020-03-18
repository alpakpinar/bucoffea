#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import (merge_datasets,
                                merge_extensions,
                                scale_xs_lumi)

from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint
from coffea import hist

pjoin = os.path.join

def compare_two_gjets_samples(acc, samples, outtag, distribution='vpt', inclusive=True):
    '''
    Compare the specified distribution of several LO GJets samples.
    List of samples is specified in "samples" argument.
    The distribution to be compared is specified in "distribution" argument.

    If inclusive option is set to True, look at distributions
    without VBF cuts applied, otherwise look at distributions
    after VBF cuts are applied.
    '''
    # Extract dataset years
    extract_year = lambda name: name.split('_')[-1]
    years = map(extract_year, samples)

    # Get the distribution
    dist = f'gen_{distribution}_inclusive_stat1_withDRreq' if inclusive else f'gen_{distribution}_vbf_stat1_withDRreq'
    
    acc.load(dist)
    h = acc[dist]

    rebin = {
        'vpt' : hist.Bin('vpt', r'$p_T(V)\ (GeV)$', np.arange(200,1500,50)),
        'mjj' : hist.Bin('mjj', r'$M_{jj}\ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
        'ak4_pt0' : hist.Bin('jpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(0,1000,50)) ),
        'ak4_pt1' : hist.Bin('jpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(0,600,50)) )
    }

    # Rebin, if neccessary
    if distribution in rebin.keys():
        distbin = rebin[distribution]
        h = h.rebin(distbin.name, distbin)

    xaxis = h.axes()[1]

    centers = xaxis.centers(overflow='over')

    # Merging and scaling
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Mapping from sample names to regular expressions
    # for the relevant dataset names
    sample_to_regex = {s : s.replace(f'{year}',f'.*_{year}') for year, s in zip(years, samples)}

    # Store histograms from each dataset in a dictionary
    histos = {s : h[re.compile(sample_to_regex[s])].integrate('dataset') for s in samples}

    # Plot the comparison
    fig, ax = plt.subplots(1,1)

    for dataset, histo in histos.items():
        vals = histo.values(overflow='over')[()]
        ax.plot(centers, vals, label=dataset, marker='o')
    
    ax.legend()

    ax.set_xscale('log')
    ax.set_ylabel('Normalized Counts')

    title = 'Inclusive' if inclusive else 'After VBF Selection'
    ax.set_title(title)
    if distribution not in ['ak4_eta0', 'ak4_eta1']:
        ax.set_xlabel(xaxis.label)
    elif distribution == 'ak4_eta0':
        ax.set_xlabel(r'Leading Jet $\eta$')
    elif distribution == 'ak4_eta1':
        ax.set_xlabel(r'Trailing Jet $\eta$')

    # Save figure
    outdir = f'./output/gjets_comparisons/{outtag}/{distribution}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f"{'_VS_'.join(samples)}_{'inclusive' if inclusive else 'vbf'}.pdf")
    fig.savefig(outpath)

    print(f'File saved: {outpath}')

    plt.close()

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

    for samples in to_compare:
        for distribution in distributions:
            compare_two_gjets_samples(acc, samples=samples, inclusive=True, outtag=outtag, distribution=distribution)
            compare_two_gjets_samples(acc, samples=samples, inclusive=False, outtag=outtag, distribution=distribution)

if __name__ == '__main__':
    main()


    

    





