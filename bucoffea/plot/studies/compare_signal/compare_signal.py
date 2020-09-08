#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def calculate_loss(h_withCut, h_noCut):
    '''Calculate the difference in integrals with the cuts'''
    sumw_withCut = h_withCut.values(overflow='over')[()]
    sumw_noCut = h_noCut.values(overflow='over')[()]

    bins = h_withCut.axes()[0].edges(overflow='over') 
    bin_widths = np.diff(bins)

    int_withCut = np.sum(bin_widths * sumw_withCut)
    int_noCut = np.sum(bin_widths * sumw_noCut)

    percent_diff = np.abs(int_withCut - int_noCut) / int_noCut * 100
    return percent_diff

def compare_signal(acc, outtag, variable='mjj', year=2017):
    '''Compare the signal yields with and without cleaning cuts applied.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if variable == 'mjj':
        new_mjj_bin = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
        h = h.rebin('mjj', new_mjj_bin)

    h = h.integrate('dataset', re.compile(f'VBF.*{year}'))[re.compile('^sr_vbf$|^sr_vbf_noCleaningCuts$')]

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, overlay='region', ax=ax, overflow='over')

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e5)
    ax.set_title(f'VBF: {year}')
    
    h_withCut = h.integrate('region', 'sr_vbf')
    h_noCut = h.integrate('region', 'sr_vbf_noCleaningCuts')

    percent_diff = calculate_loss(h_withCut, h_noCut)
    ax.text(0.02, 0.93, f'Overall loss: {percent_diff:.3f}%', fontsize=12, transform=ax.transAxes)

    centers = h_noCut.axes()[0].centers(overflow='over')
    r = h_withCut.values(overflow='over')[()] / h_noCut.values(overflow='over')[()]

    rax.plot(centers, r, ls='', marker='o', color='black')

    rax.grid(True)
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('With cut / without')
    rax.set_ylim(0.8, 1.2)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1,1], color='red')
    rax.set_xlim(xlim)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{year}_signal_comp.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    compare_signal(acc, outtag)

if __name__ == '__main__':
    main()

