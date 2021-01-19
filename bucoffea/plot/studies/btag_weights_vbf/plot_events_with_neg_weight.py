#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from klepto.archives import dir_archive

pjoin = os.path.join

def get_xlabel(distribution):
    mapping = {
        'bjets_pt_fake' : r'GEN-jet $p_T \ (GeV)$',
        'bjets_eta_fake' : r'Jet $\eta$',
        'bjets_jetflav_fake' : r'Matched GEN-jet flavor',
    }
    
    return mapping[distribution]

def get_plot_tag(dataset, region, year):
    mapping = {
        'DY.*' : {
            'cr_2m_vbf' : f'$Z(\\mu\\mu) \\ {year}$',
            'cr_2e_vbf' : f'$Z(ee) \\ {year}$',
        },
        'WJets.*' : {
            'cr_1m_vbf' : f'$W(\\mu\\nu) \\ {year}$',
            'cr_1e_vbf' : f'$W(e\\nu) \\ {year}$',
        },
    }

    for dataset_regex, tags in mapping.items():
        if re.match(dataset_regex, dataset):
            return tags[region]

    raise RuntimeError(f'Please check: {dataset}, {region}')

def plot_bjet_distribution(acc, outtag, distribution, region, dataset):
    '''Plot the given b-jet distribution, pt or eta.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('dataset', re.compile(f'{dataset}.*{year}'))
        fig, ax = plt.subplots()

        hist.plot1d(_h, ax=ax)
        ax.set_yscale('log')
        ax.set_ylim(1e-2,1e6)
        ax.set_xlabel(get_xlabel(distribution))

        ax.get_legend().remove()

        ax.text(0., 1., get_plot_tag(dataset, region, year),
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., r'Events with $w < 0$',
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'{dataset}_{region}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved:{outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    datasets_regions = {
        'DYJetsToLL' : ['cr_2m_vbf', 'cr_2e_vbf'],
        'WJetsToLNu' : ['cr_1m_vbf', 'cr_1e_vbf'],
    }

    distributions = [
        'bjets_pt_fake',
        'bjets_eta_fake',
        'bjets_jetflav_fake',
    ]

    for dataset, regions in datasets_regions.items():
        for region in regions:
            for distribution in distributions:
                plot_bjet_distribution(acc, outtag,
                    distribution=distribution,
                    region=region,
                    dataset=dataset
                )

if __name__ == '__main__':
    main()