#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def get_title(region, year):
    mapping = {
        'cr_1m_vbf' : r'MET {} Dataset: $1\mu$ CR',
        'cr_1e_vbf' : r'EGamma {} Dataset: $1e$ CR',
        'cr_g_vbf' : r'EGamma {} Dataset: $\gamma$ CR',
    }

    for regex, titletemp in mapping.items():
        if re.match(regex, region):
            return titletemp.format(year)

    raise RuntimeError(f'Could not find title for: {region}, {year}')

def get_new_legend_label(oldlabel):
    '''Get prettier legend labels for the comparison plot.'''
    newlabels = {
        'cr_1m_vbf_large_pu' : r'$30 \leq N_{PV} \leq 60$',
        'cr_1m_vbf_small_pu' : r'$N_{PV} \leq 20$',
    }

    for label, newlabel in newlabels.items():
        if re.match(label, oldlabel):
            return newlabel
    
    raise RuntimeError(f'Could not find legend label for: {oldlabel}')

def compare_low_pu_high_pu(acc, outtag, region='cr_1m_vbf', distribution='mjj'):
    '''For the given variable, compare the distribution at low PU and high PU (in data).'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500.,2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)
            
    for year in [2017, 2018]:
        _h = h.integrate('dataset', f'MET_{year}')[re.compile('cr_1m.*_(small|large)_pu')]

        fig, ax, rax = fig_ratio()
        hist.plot1d(_h, ax=ax, overlay='region')

        ax.set_yscale('log')
        ax.set_ylim(1e0,1e6)
        ax.set_xlabel('')

        # Handle legend labels
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            newlabel = get_new_legend_label(label)
            handle.set_label(newlabel)

        ax.legend(title='Pileup', handles=handles)

        ax.set_title( 
            get_title(region, year),
            fontsize=14
            )

        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
            'color':'k',
        }

        # Plot ratio
        hist.plotratio(
            _h.integrate('region', re.compile(f'{region}.*large_pu')),
            _h.integrate('region', re.compile(f'{region}.*small_pu')),
            ax=rax,
            unc='num',
            error_opts=data_err_opts
        )

        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
        rax.set_ylabel('High PU / Low PU')
        rax.set_ylim(0,2)
        rax.grid(True)

        rax.axhline(1, xmin=0, xmax=1, color='red')

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'met_data_comp_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for region in ['cr_1m_vbf']:
        compare_low_pu_high_pu(acc, outtag, region=region)

if __name__ == '__main__':
    main()