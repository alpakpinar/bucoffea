#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
import mplhep as hep
import warnings

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from klepto.archives import dir_archive

pjoin = os.path.join

warnings.filterwarnings('ignore')

binning = {
    'met' : hist.Bin('met', r'$p_T^{miss} \ (GeV)$', list(range(200,500,50)) + list(range(500,1000,100))),
    'ak4_pt0' : hist.Bin('jetpt', r'Leading jet $p_T$ (GeV)', list(range(80,800,20))),
    'ak4_pt1' : hist.Bin('jetpt', r'Trailing jet $p_T$ (GeV)', list(range(40,600,20)) ),
}

xlabels = {
    'met' : r'$p_T^{miss} \ (GeV)$',
    'ak4_pt0' : r'Leading jet $p_T$ (GeV)',
    'ak4_pt1' : r'Trailing jet $p_T$ (GeV)',
}

def preprocess(h, acc, variable, region_regex, trailjetfilter=False):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if not trailjetfilter:
        h = h.integrate('region', 'sr_vbf').integrate('dataset', re.compile(region_regex))
    else:
        h = h.integrate('region', 'sr_vbf_trailJetMask').integrate('dataset', re.compile(region_regex))

    if variable in binning.keys():
        if variable == 'met':
            h = h.rebin('met', binning['met'])
        elif 'ak4_pt' in variable:
            h = h.rebin('jetpt', binning[variable])

    return h

def compare_dists(acc_smear, acc_no_smear, region_regex='ZJetsToNuNu.*2017', variable='met'):
    '''Make plots comparing smeared/unsmeared quantities.'''
    acc_smear.load(variable)
    h_smear = acc_smear[variable]
    acc_no_smear.load(variable)
    h_no_smear = acc_no_smear[variable]

    h_smear = preprocess(h_smear, acc_smear, variable, region_regex)
    h_no_smear = preprocess(h_no_smear, acc_no_smear, variable, region_regex)

    # edges = h_smear.axes()[0].edges()
    centers = h_smear.axes()[0].centers()

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h_smear, ax=ax)
    hist.plot1d(h_no_smear, ax=ax, clear=False)
    
    legend_labels = ['With JER', 'Without JER']
    ax.legend(labels=legend_labels)
    ax.set_xlabel('')

    # Plot ratio
    r = h_smear.values()[()] / h_no_smear.values()[()]
    rax.plot(centers, r, marker='o', ls='', color='k')
    
    rax.grid(True)
    rax.set_ylim(0.7,1.3)
    rax.set_ylabel('With JER / Without')
    rax.set_xlabel(xlabels[variable])

    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{variable}_comp.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath_smear = bucoffea_path('./submission/merged_2020-09-18_vbfhinv_znunu_withJER')
    inpath_no_smear = bucoffea_path('./submission/merged_2020-09-18_vbfhinv_znunu_noJER')
    acc_smear = dir_archive(inpath_smear)
    acc_no_smear = dir_archive(inpath_no_smear)

    acc_smear.load('sumw')
    acc_smear.load('sumw2')
    acc_no_smear.load('sumw')
    acc_no_smear.load('sumw2')

    for var in ['met', 'ak4_pt0', 'ak4_pt1']:
        compare_dists(acc_smear=acc_smear, acc_no_smear=acc_no_smear, variable=var)

if __name__ == '__main__':
    main()