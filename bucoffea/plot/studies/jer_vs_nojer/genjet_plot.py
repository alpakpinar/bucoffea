#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def preprocess(h, acc, region, year):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region).integrate('dataset', f'ZJetsToNuNu_HT_{year}')
    return h

def plot_variable(acc, outtag, variable, region='sr_vbf_trailjeteta', year=2017):
    '''Plot distribution related to matched GEN-jets for trailing jets.'''
    acc.load(variable)
    h = preprocess(acc[variable], acc, region, year)

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)

    ax.get_legend().remove()
    ylim = ax.get_ylim()
    ax.plot([0, 0], ylim, color='red')
    ax.set_ylim(ylim)

    region_to_title = {
        'sr_vbf' : 'All trailing jets',
        'sr_vbf_trailjeteta' : r'Trailing jets with $2.4 < |\eta| < 2.8$',
    }

    ax.set_title(region_to_title[region])

    # Save figure
    outdir = f'./output/genjet/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{region}_{variable}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = [
        'deltapt',
        'matched_genjet_idx',
        'matched_genjet_pt'
    ]

    for region in ['sr_vbf', 'sr_vbf_trailjeteta']:
        for variable in variables:
            try:
                plot_variable(acc, outtag, variable=variable, region=region)
            except KeyError:
                print(f'Variable {variable} not found, skipping')
                continue

if __name__ == '__main__':
    main()
