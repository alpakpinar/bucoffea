#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def compare_eff(acc):
    '''Calculate the efficiency of neutral EM fraction cut as a function of the jet eta, plot the efficiency for data and MC.'''
    acc.load('ak4_eta0')
    h = acc['ak4_eta0']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant dataset and regions
    h_data = h.integrate('dataset', 'EGamma_2017')[re.compile('.*EmEF.*')]
    h_mc = h.integrate('dataset', re.compile('GJets_DR-0p4.*2017'))[re.compile('.*EmEF.*')]

    # Get the event yields with and without the fraction cut applied
    h_data_withCut = h_data.integrate('region', 'cr_g_withEmEF')
    h_data_withoutCut = h_data.integrate('region', 'cr_g_noEmEF')

    h_mc_withCut = h_mc.integrate('region', 'cr_g_withEmEF')
    h_mc_withoutCut = h_mc.integrate('region', 'cr_g_noEmEF')

    # Calculate and plot efficiencies for data and MC
    fig, ax = plt.subplots()
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    hist.plotratio(h_data_withCut, h_data_withoutCut, ax=ax, error_opts=data_err_opts, label='Single Photon')
    hist.plotratio(h_mc_withCut, h_mc_withoutCut, ax=ax, error_opts=data_err_opts, clear=False, label='GJets_DR-0p4')

    ax.set_xlabel(r'Jet $\eta$')
    ax.set_ylabel('Efficiency')
    ax.set_ylim(0.8,1.1)
    ax.grid(True)
    ax.legend()

    # Save figure
    fig.savefig('test.pdf')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(
        inpath,
        memsize=1e3,
        serialized=True,
        compression=0
    )

    acc.load('sumw')
    acc.load('sumw2')

    compare_eff(acc)

if __name__ == '__main__':
    main()