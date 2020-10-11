#!/usr/bin/env python

# =========================================
# Compare Z(vv) k-factors between BU and IC
# =========================================

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

bu_procs = {
    'zjet' : 'dy',
    'wjet' : 'wjet'
}

titles = {
    'zjet' : 'DY',
    'wjet' : 'QCD W'
}

def plot_ratio_of_kfacs(bu_kfac_file, ic_kfac_file, proc):
    h_bu = uproot.open(bu_kfac_file)[f'2d_{bu_procs[proc]}_vbf']
    h_ic = uproot.open(ic_kfac_file)['kfactors_shape']['kfactor_vbf']

    # Get the values with p_T(V) > 200 GeV and mjj > 200 GeV for comparison
    xedges = h_bu.edges[0][1:]
    yedges = h_bu.edges[1][6:]
    kfacs_bu = h_bu.values[1:, 6:] 
    kfacs_ic = h_ic.values.T[1:, 6:]  # Take the transpose to be consistent with BU   

    # Calculate the ratio of k-factors
    r = kfacs_bu / kfacs_ic

    # Plot the ratio
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(xedges, yedges, r.T)
    fig.colorbar(pc, ax=ax, label='BU / IC')

    xcenters = ( (xedges + np.roll(xedges,-1))/2 )[:-1]
    ycenters = ( (yedges + np.roll(yedges,-1))/2 )[:-1]

    for ix in range(len(xcenters)):
        for iy in range(len(ycenters)):
            ax.text(
                xcenters[ix],
                ycenters[iy],
                f'{r[ix, iy]:.3f}',
                ha='center',
                va='center',
                fontsize=6,
                color='black'
            )

    ax.set_title(f'{titles[proc]} NLO k-factors')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel(r'$p_T (V) \ (GeV)$')

    # Save figure
    outdir = f'./output/kfac_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{proc}_kfac_comparison.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    # The paths to root files containing k-factors
    proc = sys.argv[1]
    inpath_bu = './inputs/kfactor_comp'
    inpath_ic = './inputs/kfactor_comp/ic'
    bu_kfac_file = pjoin(inpath_bu, '2017_gen_v_pt_qcd_sf.root')
    ic_kfac_file = pjoin(inpath_ic, f'2Dkfactor_VBF_{proc}.root')

    plot_ratio_of_kfacs(bu_kfac_file, ic_kfac_file, proc)

if __name__ == '__main__':
    main()