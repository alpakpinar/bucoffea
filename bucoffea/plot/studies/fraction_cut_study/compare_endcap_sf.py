#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from coffea import hist
from matplotlib import pyplot as plt
from pprint import pprint
from compare_eff import ratio_unc

pjoin = os.path.join

warnings.filterwarnings('ignore')

def compare_endcap_sf(f, outdir, year):
    '''Make a comparison plot for the endcap SFs for the given year.'''
    # Read the values from the input ROOT file
    h_ak40_in_endcap = f[f'jetsf_ak40_in_endcap_{year}'].values
    h_ak40_in_pos_endcap = f[f'jetsf_ak40_in_pos_endcap_{year}'].values
    h_ak40_in_neg_endcap = f[f'jetsf_ak40_in_neg_endcap_{year}'].values

    xcenters = 0.5 * np.sum(f[f'jetsf_ak40_in_endcap_{year}'].bins, axis=1)    

    # Read the stat errors
    h_ak40_in_endcap_statUp = f[f'jetsf_ak40_in_endcap_{year}_statUp'].values - h_ak40_in_endcap
    h_ak40_in_endcap_statDown = f[f'jetsf_ak40_in_endcap_{year}_statDown'].values - h_ak40_in_endcap
    h_ak40_in_endcap_err = np.vstack([
        np.abs(h_ak40_in_endcap_statUp),
        np.abs(h_ak40_in_endcap_statDown)
    ])

    h_ak40_in_pos_endcap_statUp = f[f'jetsf_ak40_in_pos_endcap_{year}_statUp'].values - h_ak40_in_pos_endcap
    h_ak40_in_pos_endcap_statDown = f[f'jetsf_ak40_in_pos_endcap_{year}_statDown'].values - h_ak40_in_pos_endcap
    h_ak40_in_pos_endcap_err = np.vstack([
        np.abs(h_ak40_in_pos_endcap_statUp),
        np.abs(h_ak40_in_pos_endcap_statDown),
    ])

    h_ak40_in_neg_endcap_statUp = f[f'jetsf_ak40_in_neg_endcap_{year}_statUp'].values - h_ak40_in_neg_endcap
    h_ak40_in_neg_endcap_statDown = f[f'jetsf_ak40_in_neg_endcap_{year}_statDown'].values - h_ak40_in_neg_endcap
    h_ak40_in_neg_endcap_err = np.vstack([
        np.abs(h_ak40_in_neg_endcap_statUp),
        np.abs(h_ak40_in_neg_endcap_statDown),
    ])

    # Plot comparison
    opts = {
        'marker' : 'o',
    }
    fig, ax = plt.subplots()
    ax.errorbar(xcenters, y=h_ak40_in_endcap, yerr=h_ak40_in_endcap_err, label=r'$2.5 < |\eta| < 3.0$', **opts)
    ax.errorbar(xcenters, y=h_ak40_in_pos_endcap, yerr=h_ak40_in_pos_endcap_err, label=r'$2.5 < \eta < 3.0$', **opts)
    ax.errorbar(xcenters, y=h_ak40_in_neg_endcap, yerr=h_ak40_in_neg_endcap_err, label=r'$-3.0 < \eta < -2.5$', **opts)

    ax.legend(title=r'Jet $\eta$')
    ax.set_ylim(0.7,1.3)
    ax.set_ylabel('Data / MC SF')
    ax.set_xlabel(r'Jet $p_T \ (GeV)$')

    ax.set_title(f'Data / MC SFs: {year}')

    # Save figure
    outpath = pjoin(outdir, f'sf_comp_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    # Path to the root file
    inpath = sys.argv[1]
    f = uproot.open(inpath)

    # Output path to save plots
    outdir = os.path.dirname(os.path.dirname(inpath))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        compare_endcap_sf(f, outdir, year=year)

if __name__ == '__main__':
    main()