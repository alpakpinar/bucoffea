#!/usr/bin/env python

import os
import sys
import re
import argparse
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
}

def preprocess(h, acc, distribution):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        h = h.rebin('mjj', REBIN['mjj'])

    return h

def compare_z_over_w_ratio(acc, outtag, channel='muons', distribution='mjj', proc='qcd'):
    '''Compare Z / W ratio for two cases.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        dataset_regex_for_ratio = {
            'qcd' : {
                'zjets' : re.compile(f'DYJetsToLL.*{year}'),
                'wjets' : re.compile(f'WJetsToLNu.*{year}'),
            },
            'ewk' : {
                'zjets' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}'),
                'wjets' : re.compile(f'EWKW2Jets.*{year}'),
            },
        }

        fig, ax, rax = fig_ratio()

        regions = {
            'muons' : ['cr_2m_vbf', 'cr_1m_vbf'],
            'electrons' : ['cr_2e_vbf', 'cr_1e_vbf'],
        }

        regions_to_look = regions[channel]

        # Numerator and denominator histograms
        h_num = h.integrate('region', regions_to_look[0]).integrate('dataset', dataset_regex_for_ratio[proc]['zjets'])
        h_num_nodphijjcut = h.integrate('region', f'{regions_to_look[0]}_nodphijjcut').integrate('dataset', dataset_regex_for_ratio[proc]['zjets'])

        h_den = h.integrate('region', regions_to_look[1]).integrate('dataset', dataset_regex_for_ratio[proc]['wjets'])
        h_den_nodphijjcut = h.integrate('region', f'{regions_to_look[1]}_nodphijjcut').integrate('dataset', dataset_regex_for_ratio[proc]['wjets'])

        # Plot the two ratios
        hist.plotratio(h_num, h_den, 
            ax=ax, 
            unc='num',
            error_opts=data_err_opts,
            label=r'With $\Delta\phi_{jj}$ cut',
            )

        hist.plotratio(h_num_nodphijjcut, h_den_nodphijjcut, 
            ax=ax, 
            unc='num',
            error_opts=data_err_opts,
            label=r'No $\Delta\phi_{jj}$ cut',
            clear=False
            )

        ax.set_xlim(200,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel('Ratio')
        ax.legend()

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        ax.yaxis.set_ticks_position('both')

        textlabels = {
            'muons' : r'{} $Z(\mu\mu) \ / \ W(\mu\nu)$'.format(proc.upper()),
            'electrons' : r'{} $Z(ee) \ / \ W(e\nu)$'.format(proc.upper()),
        }

        ax.text(0., 1., textlabels[channel],
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

        # Plot the ratio of the two cases
        r_withcut = h_num.values()[()] / h_den.values()[()]
        r_nocut = h_num_nodphijjcut.values()[()] / h_den_nodphijjcut.values()[()]

        centers = h_num.axis('mjj').centers()
        rr = r_nocut / r_withcut
        rax.plot(centers, rr, ls='', marker='o', color='k')

        rax.grid(True)
        rax.set_ylim(0.5,1.5)
        rax.set_ylabel(r'$\Delta\phi_{jj}$ Inclusive / $\Delta\phi_{jj} < 1.5$')
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')

        # Save figure
        outpath = pjoin(outdir, f'{proc}_z_over_w_{channel}_with_without_dphijjcut_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for channel in ['electrons', 'muons']:
        for proc in ['qcd', 'ewk']:
            compare_z_over_w_ratio(acc, outtag, proc=proc, channel=channel)

if __name__ == '__main__':
    main()
