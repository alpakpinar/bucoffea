#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

def plot_z_over_w(acc, outtag, channel='muons', distribution='mjj'):
    '''Plot Z/W ratio from the accumulator itself.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        h = h.rebin('mjj', REBIN['mjj'])

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        dataset_regex = {
            'qcdZ' : re.compile(f'DYJetsToLL.*{year}'),
            'ewkZ' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}'),
            'qcdW' : re.compile(f'WJetsToLNu.*{year}'),
            'ewkW' : re.compile(f'EWKW2Jets.*{year}'),
        }
        
        regions = {
            'muons' : ('2m', '1m'),
            'electrons' : ('2e', '1e'),
        }

        h_qcdz = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', dataset_regex['qcdZ'])
        h_ewkz = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', dataset_regex['ewkZ'])
        h_qcdw = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', dataset_regex['qcdW'])
        h_ewkw = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', dataset_regex['ewkW'])
        
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        fig, ax = plt.subplots()
        hist.plotratio(h_qcdz, h_qcdw, 
            ax=ax, 
            unc='num',
            label='QCD',
            error_opts=data_err_opts
            )

        hist.plotratio(h_ewkz, h_ewkw, 
            ax=ax,
            unc='num',
            label='EWK',
            error_opts=data_err_opts,
            clear=False
            )

        h_qcdz.add(h_ewkz)
        h_qcdw.add(h_ewkw)

        hist.plotratio(h_qcdz, h_qcdw,
            ax=ax,
            unc='num',
            label='QCD + EWK',
            error_opts=data_err_opts,
            clear=False
        )

        ylabels = {
            'muons' : r'$Z(\mu\mu) \ / \ W(\mu\nu)$',
            'electrons' : r'$Z(ee) \ / \ W(e\nu)$',
        }

        ax.set_xlim(0,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel(ylabels[channel])
        ax.legend()

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        ax.yaxis.set_ticks_position('both')

        ax.text(0., 1., year,
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_{channel}_{year}.pdf')
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
        plot_z_over_w(acc, outtag, channel=channel)

if __name__ == '__main__':
    main()