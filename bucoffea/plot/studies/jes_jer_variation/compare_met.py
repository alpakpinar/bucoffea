#!/usr/bin/env python

#####################
# Script to compare several types of MET in NanoAOD:
# JER smeared MET pt: met_pt_jer
# MET pt with no JER smearing applied: met_pt_nom
# MET pt with JES / JER variations applied

# INPUT: Use with 2020-03-28_vbf_jes_jer_var job
######################

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt

pjoin = os.path.join

def preprocess_histos(h, acc, regex):
    '''Merging, scaling, integrating.'''
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    # Histograms filled only for signal region, integrate it out
    h = h.integrate('region').integrate('dataset', re.compile(regex) )

    # Rebin
    new_met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,500,50)) + list(range(500,1100,100)) )
    h = h.rebin('met', new_met_bin)
    return h

def compare_jer_nom_met(acc, regex, dataset_name, tag, outtag, inclusive=True):
    '''Plot JER smeared and non-JER smeared MET pt.'''
    inc_suffix = '_inc' if inclusive else ''
    acc.load(f'met_jer{inc_suffix}')
    acc.load(f'met_nom{inc_suffix}')


    # Get pre-processed histograms
    histograms = {
        'with JER' : preprocess_histos(acc[f'met_jer{inc_suffix}'], acc, regex),
        'without JER' : preprocess_histos(acc[f'met_nom{inc_suffix}'], acc, regex)
    }

    met_edges = histograms['with JER'].axes()[0].edges()

    # Get normalized histogram values as numpy arrays
    values = {key : histograms[key].values()[()]/np.sum(histograms[key].values()[()]) for key in histograms.keys()}

    # Plot comparison between the two
    fig, ax = plt.subplots(1,1)
    for key, arr in values.items():
        ax.step(met_edges[:-1], arr, label=key, where='post')

    ax.legend(title='Signal Region VBF')
    ax.set_xlim(met_edges[0], met_edges[-2])
    ax.set_xlabel(r'$p_T^{miss}$ (GeV)')
    ax.set_ylabel('Normalized Counts')
    ax.set_title(dataset_name)

    # Save figure
    outdir = f'./output/{outtag}/met_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{tag}_nom_jer_met_comparison.pdf')
    fig.savefig(outpath)
    print(f'Figure saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
        inpath,
        serialized=True,
        compression=0,
        memsize=1e3
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    data_info = {
        'wjets2017' : {'dataset_name' : 'WJetsToLNu_HT_2017', 'regex' : 'WJetsToLNu.*2017'},
        'wjets2018' : {'dataset_name' : 'WJetsToLNu_HT_2018', 'regex' : 'WJetsToLNu.*2018'},
        'zjets2017' : {'dataset_name' : 'ZJetsToNuNu_HT_2017', 'regex' : 'ZJetsToNuNu.*2017'},
        'zjets2018' : {'dataset_name' : 'ZJetsToNuNu_HT_2018', 'regex' : 'ZJetsToNuNu.*2018'},
        'gjets2017' : {'dataset_name' : 'GJets_DR-0p4_HT_2017', 'regex' : 'GJets_DR-0p4.*2017'},
        'gjets2018' : {'dataset_name' : 'GJets_DR-0p4_HT_2018', 'regex' : 'GJets_DR-0p4.*2018'},
    }

    for tag, info in data_info.items():
        compare_jer_nom_met(acc, dataset_name=info['dataset_name'], regex=info['regex'], tag=tag, outtag=outtag)

if __name__ == '__main__':
    main()