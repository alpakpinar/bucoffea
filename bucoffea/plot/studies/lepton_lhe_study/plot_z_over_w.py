#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive

pjoin = os.path.join

def get_region(dataset, channel):
    mapping = {
        'DY' : {'electrons' : 'cr_2e_lhe_vbf', 'muons' : 'cr_2m_lhe_vbf'},
        'W'  : {'electrons' : 'cr_1e_lhe_vbf', 'muons' : 'cr_1m_lhe_vbf'},
    }
    return mapping[dataset][channel]

def get_plot_tag(channel, year):
    mapping = {
        'electrons' : f'$Z(ee)$ / $W(e\\nu)$ {year}',
        'muons' : f'$Z(\\mu\\mu)$ / $W(\\mu\\nu)$ {year}',
    }
    return mapping[channel]

def plot_z_over_w(acc, outtag, channel='electrons'):
    '''Plot Z/W ratio, where events are selected using regular jet cuts + LHE-level lepton cuts.'''
    distribution = 'mjj'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }
    
    for year in [2017, 2018]:
        h_z = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}')).integrate('region', get_region('DY', channel))
        h_w = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}')).integrate('region', get_region('W', channel))

        # Plot the ratio
        fig, ax = plt.subplots()
        hist.plotratio(h_z, h_w, 
            ax=ax,
            unc='num',
            error_opts=data_err_opts
            )

        ax.text(0., 1., get_plot_tag(channel, year),
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.set_xlim(200,5000)
        ax.set_ylim(0,0.2)

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_{channel}_from_lhe_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved:{outpath}')

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