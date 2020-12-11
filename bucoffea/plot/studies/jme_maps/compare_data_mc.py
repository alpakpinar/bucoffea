#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

xlabels = {
    'mjj' : r'$M_{jj} \ (GeV)$',
    'detajj' : r'$\Delta\eta_{jj}$',
}

def preprocess(h, acc, distribution, region):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution in REBIN.keys():
        new_ax = REBIN[distribution]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('region', region)

    return h

def compare_data_mc(acc_dict, year, categories, distribution='mjj', region='sr_vbf'):
    '''Compare data/MC ratio for the given region for several JME map vetoes.'''
    h_dict = {}
    for category in categories:
        acc_dict[category].load(distribution)
        h_dict[category] = preprocess(acc_dict[category][distribution], acc_dict[category], distribution, region)

    data = f'MET_{year}'
    mc = re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|DYJetsToLL_M-50_HT_MLM.*|WJetsToLNu.*HT.*).*{year}')

    fig, ax = plt.subplots()

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
    }

    for category, h in h_dict.items():
        h_data = h[data].integrate('dataset')
        h_mc = h[mc].integrate('dataset')

        hist.plotratio(h_data, h_mc, 
                ax=ax, 
                unc='num', 
                clear=False, 
                error_opts=data_err_opts,
                label=category
                )

    ax.set_xlabel(xlabels[distribution], fontsize=14)
    ax.set_ylabel('Data / MC', fontsize=14)
    ax.set_ylim(0.8,1.5)
    ax.legend()
    ax.set_title(f'Data/MC in SR: {year}', fontsize=14)

    ax.axhline(1, xmin=0, xmax=1, color='k', lw=2)

    # Save figure
    outdir = './output/data_mc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'data_mc_{region}_{distribution}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    acc_dict = {
        'noveto' : dir_archive( bucoffea_path('submission/merged_2020-12-03_vbfhinv_run_02Dec20') ),
        'veto_hotTowers_dR0' : dir_archive( bucoffea_path('submission/merged_2020-12-10_vbfhinv_veto_hotTowers_dR0') ),
        'veto_hotTowers_dR2' : dir_archive( bucoffea_path('submission/merged_2020-12-10_vbfhinv_veto_hotTowers_dR2') ),
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    for year in [2017, 2018]:
        for distribution in ['mjj', 'detajj']:
            compare_data_mc(acc_dict, 
                    year=year, 
                    categories=acc_dict.keys(),
                    distribution=distribution
                    )

if __name__ == '__main__':
    main()