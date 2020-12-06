#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.colors as colors

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def plot_2d(acc, outtag, region):
    '''Plot 2D VecB/VecDPhi distribution.'''
    distribution = 'vecb_dphitkpf'
    acc.load(distribution)
    h = acc[distribution]

    h = h.integrate('region', region)

    # Output directory to save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'VBF_HToInvisible_M125(_PSweights)?_pow_pythia8_{year}'))
        fig, ax = plt.subplots()
        patch_opts = {
            'norm' : colors.LogNorm(vmin=1e-2, vmax=1e2)
        }
        hist.plot2d(_h, ax=ax, xaxis='dphi', patch_opts=patch_opts)

        ax.set_ylim(0,0.3)
        ax.set_xlabel(r'$\Delta\phi(Tk,PF)$')
        ax.set_title(f'VBF H(inv) EE-HF Events: {year}', fontsize=14)

        ax.axvline(1., ymin=0, ymax=1, ls='--', color='black')

        outpath = pjoin(outdir, f'vbf_2d_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def plot_1d(acc, outtag, distribution, region):
    '''Plot 1D distribution of the given variable in the given region, for the signal MC.'''
    acc.load(distribution)
    h = acc[distribution]

    # scale_xs_lumi(h)
    h = h.integrate('region', region)

    # Output directory to save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'VBF_HToInvisible_M125(_PSweights)?_pow_pythia8_{year}'))
        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax)
        ax.get_legend().remove()

        outpath = pjoin(outdir, f'vbf_{distribution}_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    distributions = [
        'vecdphi',
        'vecb',
        'dphitkpf',
    ]

    for distribution in distributions:
        plot_1d(acc, outtag, distribution, region='sr_vbf_no_veto_all_ee_hf')
    
    plot_2d(acc, outtag, region='sr_vbf_no_veto_all_ee_hf')

if __name__ == '__main__':
    main()