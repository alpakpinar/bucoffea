#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive

pjoin = os.path.join

def get_proc_tag(dataset, region, year):
    mapping = {
        'DYJetsToLL' : {
            'cr_2m_vbf' : f'$Z(\\mu\\mu)$ {year}',
            'cr_2e_vbf' : f'$Z(ee)$ {year}',
        },
        'WJetsToLNu' : {
            'sr_vbf' : f'$W(\\ell\\nu)$ {year}',
            'cr_1m_vbf' : f'$W(\\mu\\nu)$ {year}',
            'cr_1e_vbf' : f'$W(e\\nu)$ {year}',
        },
        'ZJetsToNuNu' : {
            'sr_vbf' : f'$Z(\\nu\\nu)$ {year}',
        },
    }

    return mapping[dataset][region]

def plot_btag_weights(h, outtag, dataset, region, year):
    '''Plot the b-tag weight distributions.'''
    h = h.integrate('dataset', re.compile(f'{dataset}.*{year}')).integrate('region', region)

    # Get b-veto weights
    h = h.integrate('weight_type', 'bveto')

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)

    ax.text(0, 1, get_proc_tag(dataset, region, year),
        fontsize=14,
        ha='left',
        va='bottom',
        transform=ax.transAxes
    )

    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e5)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlim(-2,2)
    ax.set_xlabel('b-tag Weight', fontsize=14)

    ax.get_legend().remove()

    # Save figure
    outdir = f'./output/{outtag}/btag_weights'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{dataset}_{region}_btag_weights_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')
    
    # Prepare the histogram once
    acc.load('weights_wide')
    h = acc['weights_wide']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    datasets_regions = {
        'DYJetsToLL' : ['cr_2m_vbf', 'cr_2e_vbf'],
        'WJetsToLNu' : ['cr_1m_vbf', 'cr_1e_vbf', 'sr_vbf'],
        'ZJetsToNuNu' : ['sr_vbf'],
    }

    for dataset, regions in datasets_regions.items():
        for year in [2017, 2018]:
            for region in regions:
                plot_btag_weights(h, outtag, dataset=dataset, region=region, year=year)

if __name__ == '__main__':
    main()