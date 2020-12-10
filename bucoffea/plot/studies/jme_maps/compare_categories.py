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
    'mjj': hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500.])
}

def preprocess(h, acc, distribution='mjj', region='sr_vbf'):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution in REBIN.keys():
        new_ax = REBIN[distribution]
        h = h.rebin(h.axis(new_ax.name), new_ax)

    return h

def compare_categories(acc, outtag, year, dr_tag, categories, distribution='mjj', dataset='MET'):
    '''Compare different JME veto categories in the given dataset.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)
    h = h.integrate('dataset', re.compile(f'{dataset}.*{year}'))
    
    # For each JME category, get the relevant signal region and plot the yields
    fig, ax, rax = fig_ratio()
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    legend_labels = []
    centers = h.axis('mjj').centers(overflow='over')

    sumw = {}

    for category in categories:
        _h = h.integrate('region', f'sr_vbf{category}')
        if category == '':
            hist.plot1d(_h, ax=ax, overflow='over', error_opts=data_err_opts)
            legend_labels.append('Nominal')
        else:
            hist.plot1d(_h, ax=ax, overflow='over', clear=False)
            legend_labels.append(category.replace('_', '', 1))

        sumw[category] = _h.values()[()]

    ax.legend(labels=legend_labels)

    dataset_tags = {
        'MET' : 'Data',
        'VBF' : r'VBF $H(inv)$ Signal',
    }

    ax.text(0., 1., dataset_tags[dataset],
        fontsize=14,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    ax.text(1., 1., year,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    # Plot ratios
    sumw_nom = sumw['']
    data_err_opts.pop('color')
    
    for category in categories:
        if category == '':
            continue
        r = sumw[category] / sumw_nom

        rax.plot(centers, r, **data_err_opts)

    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('Ratio to Nominal')
    rax.grid(True)
    rax.set_ylim(0.5,1.5)

    rax.axhline(1, xmin=0, xmax=1, color='red')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'comparison_{dr_tag}_{distribution}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # List of veto combinations to compare with the baseline (no-veto, corresponds to '' here)
    jme_categories = [
        ('', '_hotTowers_dR0', '_coldTowers_dR0', '_hotAndColdTowers_dR0'),
        ('', '_hotTowers_dR2', '_coldTowers_dR2', '_hotAndColdTowers_dR2'),
    ]

    dr_tags = [
        'dR0', 'dR2'
    ]

    for year in [2017, 2018]:
        for idx, categories in enumerate(jme_categories):
            compare_categories(acc, outtag,
                    year=year,
                    categories=categories,
                    distribution='mjj',
                    dr_tag=dr_tags[idx]
                    )

if __name__ == '__main__':
    main()