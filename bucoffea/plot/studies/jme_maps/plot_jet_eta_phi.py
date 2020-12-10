#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import matplotlib.colors as colors

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join


def plot_jet_eta_phi(acc, outtag, distribution, runperiods):
    '''From data or MC, plot the eta phi distribution for the leading or trailing jet.'''
    acc.load(distribution)
    h = acc[distribution]

    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    # h = merge_datasets(h)

    h = h.integrate('region', 'sr_vbf')

    for runperiod in runperiods:
        _h = h.integrate('dataset', re.compile(f'MET.*{runperiod}'))

        # Plot eta/phi map
        fig, ax = plt.subplots()

        patch_opts = {
            'norm' : colors.LogNorm(1e-1, 1e2)
        }
        
        hist.plot2d(_h, xaxis='jeteta', ax=ax, patch_opts=patch_opts)
    
        if distribution == 'ak4_eta0_phi0':
            ax.set_xlabel(r'Leading jet $\eta$')
            ax.set_ylabel(r'Leading jet $\phi$')
        else:
            ax.set_xlabel(r'Trailing jet $\eta$')
            ax.set_ylabel(r'Trailing jet $\phi$')

        ax.set_title(f'MET Dataset: {runperiod}', fontsize=14)

        # Save figure
        outpath = pjoin(outdir, f'met_{runperiod}_{distribution}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # Run periods for MET dataset
    runperiods = {
       2017: ['2017B', '2017C', '2017D', '2017E', '2017F'],
       2018: ['2018A', '2018B', '2018C', '2018D'],
    }
    
    for year in [2017, 2018]:
        for distribution in ['ak4_eta0_phi0', 'ak4_eta1_phi1']:
            plot_jet_eta_phi(acc, outtag, distribution=distribution, runperiods=runperiods[year])

if __name__ == '__main__':
    main()
