#!/usr/bin/env python

import os
import sys
import re
import matplotlib.colors as colors
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

rebin = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', list(range(0,5500,500))),
    'vpt' : hist.Bin('vpt', r'$p_T(V) \ (GeV)$', list(range(0,2200,200))),
}

def plot_vpt(acc, year):
    var = 'gen_vpt'
    acc.load(var)
    h = acc[var]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebinning
    h = h.rebin('vpt', rebin['vpt'])

    h = h.integrate('region', 'sr_vbf_no_veto_all').integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)

    ax.get_legend().remove()

    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e6)

    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'gen_vpt_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_vpt_mjj(acc, year):
    var = 'gen_vpt_mjj'
    acc.load(var)
    h = acc[var]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebinning
    for var in ['mjj', 'vpt']:
        h = h.rebin(var, rebin[var])

    h = h.integrate('region', 'sr_vbf_no_veto_all').integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))

    fig, ax = plt.subplots()
    hist.plot2d(h, ax=ax, xaxis='mjj', patch_opts={'norm' : colors.LogNorm(vmin=1e-3, vmax=1e5)})

    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'gen_vpt_mjj_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    for year in [2017, 2018]:
        plot_vpt(acc, year)
        plot_vpt_mjj(acc, year)

if  __name__ == '__main__':
    main()
