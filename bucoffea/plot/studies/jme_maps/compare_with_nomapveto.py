#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.colors as colors

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj': hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

def preprocess(h, acc, distribution='mjj', region='sr_vbf'):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)

    if distribution in REBIN.keys():
        new_ax = REBIN[distribution]
        h = h.rebin(h.axis(new_ax.name), new_ax)

    return h

def compare_with_nomapveto(acc_dict, year, distribution='mjj', region='sr_vbf', dataset='MET'):
    '''Compare the cases where the JME hot/cold map based veto is applied and not applied.'''
    outdir = './output/compare_with_nomapveto'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    h_dict = {}
    for key, acc in acc_dict.items():
        acc.load(distribution)
        h_dict[key] = preprocess( acc[distribution], acc, distribution, region )
        h_dict[key] = h_dict[key].integrate('dataset', re.compile(f'{dataset}.*{year}'))
        
    fig, ax, rax = fig_ratio()
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    hist.plot1d(h_dict['nomapveto'], ax=ax, error_opts=data_err_opts)
    hist.plot1d(h_dict['withmapveto'], ax=ax, clear=False)

    labels = [
        'With JME veto',
        'No JME veto',
    ]

    ax.legend(labels=labels)
    ax.set_title(f'{dataset} Dataset: {year}', fontsize=14)

    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e7)

    # Plot ratio of the two
    hist.plotratio(
        h_dict['nomapveto'],
        h_dict['withmapveto'],
        ax=rax,
        unc='num',
        error_opts=data_err_opts
        )

    rax.grid(True)
    rax.set_ylim(0,2)
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('Without veto / with')

    rax.axhline(1, xmin=0, xmax=1, color='red')

    outpath = pjoin(outdir, f'{dataset}_{distribution}_{region}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    acc_dict = {
        'nomapveto' : dir_archive( bucoffea_path('submission/merged_2020-12-08_vbfhinv_nohornveto') ),
        'withmapveto' : dir_archive( bucoffea_path('submission/merged_2020-12-09_vbfhinv_jmemaps_test_v2') ),
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    for year in [2017, 2018]:
        for dataset in ['MET', 'VBF']:
            compare_with_nomapveto(acc_dict, year=year, dataset=dataset)

if __name__ == '__main__':
    main()