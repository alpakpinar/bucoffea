#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

def dataset_to_title(dataset):
    mapping = {
        'ZJetsToNuNu' : r'QCD $Z(\nu\nu)$',
        'WJetsToLNu' : r'QCD $W(\ell\nu)$',
        'VBF_HToInv' : r'VBF $H(inv)$',
    }
    return mapping[dataset]

def preprocess(h, acc, dataset, year, region='sr_vbf'):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Integrate over the relevant dataset and region
    h = h.integrate('region', region).integrate('dataset', re.compile(f'{dataset}.*{year}'))

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)

    return h

def plot_btag_weight_variations(acc, outtag, dataset, year, region='sr_vbf'):
    '''For the given dataset, plot the b-tag weight variations as a function of mjj.'''
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    distributions = {
        'nominal' : 'mjj',
        'btag_up' : 'mjj_bveto_up',
        'btag_down' : 'mjj_bveto_down',
    }
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
    }

    histos = {}
    fig, ax, rax = fig_ratio()
    for tag, distribution in distributions.items():
        acc.load(distribution)
        histos[tag] = preprocess(acc[distribution], acc, dataset, year, region)

        hist.plot1d(histos[tag], ax=ax, clear=False)

        # Plot the ratios w.r.t. nominal
        if tag == 'nominal':
            continue
        hist.plotratio(histos[tag], histos['nominal'], 
            ax=rax,
            clear=False,
            error_opts=data_err_opts
            )

    ax.legend(labels=[
        'Nominal',
        'b-weight Up',
        'b-weight Down',
    ])
    ax.set_title(dataset_to_title(dataset), fontsize=14)

    ax.text(1., 0., year,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    rax.set_ylabel('Ratio to Nominal')
    rax.set_ylim(0.8,1.2)
    rax.grid(True)
    rax.axhline(1, xmin=0, xmax=1, color='black')

    # Save figure
    outpath = pjoin(outdir, f'{dataset}_{year}_bweight_variations.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    datasets = [
        'ZJetsToNuNu',
        'WJetsToLNu',
        'VBF_HToInv'
    ]

    for year in [2017, 2018]:
        for dataset in datasets:
            plot_btag_weight_variations(acc, outtag, dataset=dataset, year=year)

if __name__ == '__main__':
    main()