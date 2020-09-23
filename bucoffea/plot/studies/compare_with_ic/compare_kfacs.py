#!/usr/bin/env python

# =========================================
# Compare Z(vv) k-factors between BU and IC
# =========================================

import os
import sys
import re
import uproot
import numpy as np
from bucoffea.helpers.paths import bucoffea_path
from matplotlib import pyplot as plt

pjoin = os.path.join

def plot_ratio_of_kfacs(bu_kfac_file, ic_kfac_file):
    h_bu = uproot.open(bu_kfac_file)['kfactors_shape']['kfactor_vbf']
    h_ic = uproot.open(ic_kfac_file)['kfactors_shape']['kfactor_vbf']

    yedges, xedges = h_bu.edges
    kfacs_bu = h_bu.values 
    kfacs_ic = h_ic.values 

    # Calculate the ratio of k-factors
    r = kfacs_bu / kfacs_ic

    # Plot the ratio
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(xedges, yedges, r)
    fig.colorbar(pc, ax=ax, label='BU / IC')

    xcenters = ( (xedges + np.roll(xedges,-1))/2 )[:-1]
    ycenters = ( (yedges + np.roll(yedges,-1))/2 )[:-1]

    for ix in range(len(xcenters)):
        for iy in range(len(ycenters)):
            if ix < 2:
                ratio_print = f'{r[iy, ix]:.2f}'
            else:
                ratio_print = f'{r[iy, ix]:.3f}'
            ax.text(
                xcenters[ix],
                ycenters[iy],
                ratio_print,
                ha='center',
                va='center',
                fontsize=6,
                color='black'
            )

    ax.set_title(r'QCD $Z(\nu\nu)$ NLO k-factors')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel(r'$p_T (V) \ (GeV)$')

    # Save figure
    outdir = f'./output/kfac_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'znunu_kfac_comparison.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    # The paths to root files containing k-factors
    bu_kfac_file = bucoffea_path('./data/sf/theory/2Dkfactor_VBF_znn.root')
    ic_kfac_file = bucoffea_path('./data/sf/theory/ic/2Dkfactor_VBF_znn.root')

    plot_ratio_of_kfacs(bu_kfac_file, ic_kfac_file)

if __name__ == '__main__':
    main()