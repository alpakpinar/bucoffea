#!/usr/bin/env python

# =========================================
# Compare Z(vv) k-factors between BU and IC
# =========================================

import os
import sys
import re
import uproot
import numpy as np
import matplotlib.ticker
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

def get_input_files(proc):
    '''Get input root files from both sides, containing k-facs'''
    inpath_bu = './inputs/kfactor_comp'
    inpath_ic = './inputs/kfactor_comp/ic'
    ic_kfac_file = pjoin(inpath_ic, f'2Dkfactor_VBF_{proc}.root')
    # For BU, get the file paths with original BU binning and IC binning
    bu_kfac_file_bu_binning = pjoin(inpath_bu, f'2017_gen_v_pt_qcd_sf_bu_binning.root')
    bu_kfac_file_ic_binning = pjoin(inpath_bu, f'2017_gen_v_pt_qcd_sf_ic_binning.root')

    return bu_kfac_file_bu_binning, bu_kfac_file_ic_binning, ic_kfac_file

def compare_kfacs_at_high_mjj(bu_kfac_file, ic_kfac_file, proc):
    '''For the given process, compare the k-factors applied to highest mjj bin between BU & IC'''
    h_bu = uproot.open(bu_kfac_file)[f'2d_{bu_procs[proc]}_vbf']
    h_ic = uproot.open(ic_kfac_file)['kfactors_shape']['kfactor_vbf']

    vpt_edges = h_bu.edges[1]
    vpt_centers = ( (vpt_edges + np.roll(vpt_edges,-1))/2 )[:-1]

    # Get all values for p_T(V) > 200 GeV
    vpt_mask = vpt_centers > 200
    vpt_centers = vpt_centers[vpt_mask]

    # Get the k-factors at the high-mjj end
    bu_kfacs = h_bu.values[-1,:][vpt_mask]
    ic_kfacs = h_ic.values[:,-1][vpt_mask]

    # Plot comparison of the two k-factors
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    ax.plot(vpt_centers, bu_kfacs, label='BU', marker='o')
    ax.plot(vpt_centers, ic_kfacs, label='IC', marker='o')

    ax.legend()
    ax.set_title(r'{}: k-factors @ highest $M_{{jj}}$'.format(titles[proc]))

    r = bu_kfacs / ic_kfacs
    rax.plot(vpt_centers, r, marker='o', color='k', ls='')

    rax.grid(True)
    rax.set_xlabel(r'$p_T (V) \ (GeV)$')

    rax.set_ylim(0.85,1.15)
    rax.set_ylabel('BU / IC')

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.], color='red')
    rax.set_xlim(xlim)
    
    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = './output/kfac_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{proc}_high_mjj_kfac_comparison.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')


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
    bu_kfac_file_bu_binning, bu_kfac_file_ic_binning, ic_kfac_file = get_input_files(proc)

    # For direct comparison (i.e. ratio plot), use the one with IC binning from both sides
    plot_ratio_of_kfacs(bu_kfac_file_ic_binning, ic_kfac_file, proc)

    compare_kfacs_at_high_mjj(bu_kfac_file_bu_binning, ic_kfac_file, proc)

if __name__ == '__main__':
    main()