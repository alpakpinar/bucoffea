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

def preprocess(h, acc, relax_cut, year):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    relax_cut_suffix = f'_relaxed_{relax_cut}' if relax_cut is not None else ''
    region = f'sr_vbf{relax_cut_suffix}'
        
    h = h.integrate('region', region).integrate('dataset', re.compile(f'ZJetsToNuNu.*{year}'))

    return h

def compare_jer_nojer_distribution(acc, outtag, relax_cut=None, variable='met', year=2017):
    '''Compare distribution of a variable with/without smearing.'''
    variables = [variable, f'{variable}_jer']
    for var in variables:
        acc.load(var)
    
    # Variable with and without smearing
    h_ns = acc[variable]
    h_ws = acc[f'{variable}_jer']
    
    h_ns = preprocess(h_ns, acc, relax_cut=relax_cut, year=year)
    h_ws = preprocess(h_ws, acc, relax_cut=relax_cut, year=year)

    # Plot the comparison of the two
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h_ns, ax=ax)
    hist.plot1d(h_ws, ax=ax, clear=False)

    labels = ['No JER', 'With JER']
    ax.legend(labels=labels)

    # Plot ratio
    centers = h_ns.axes()[0].centers()
    ratio = h_ws.values()[()] / h_ns.values()[()]

    rax.plot(centers, ratio, marker='o', ls='', color='k')

    rax.grid(True)
    rax.set_ylim(0.6,1.4)
    rax.set_ylabel('With JER / Without')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if relax_cut is None:
        filename = f'{variable}_{year}_jer_vs_nojer.pdf'
    else:
        filename = f'{variable}_{year}_jer_vs_nojer_relaxed_{relax_cut}.pdf'

    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['met', 'ak4_pt0', 'ak4_pt1']
    for year in [2017, 2018]:
        for variable in variables:
            # Plot the comparison of the variables for three cases:
            # 1. Regular SR, 2. Relaxed met cut (met>200), 3. Relaxed trail jet pt cut (pt>20)
            compare_jer_nojer_distribution(acc, outtag, variable=variable, year=year)
            compare_jer_nojer_distribution(acc, outtag, variable=variable, year=year, relax_cut='recoil')
            compare_jer_nojer_distribution(acc, outtag, variable=variable, year=year, relax_cut='trailak4')

if __name__ == '__main__':
    main()