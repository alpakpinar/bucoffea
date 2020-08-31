#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

# Script to check smoothing of JES/JER shape uncertainties

def plot_smoothed(rootfile, variation='jer'):
    f = uproot.open(rootfile)
    h_jerUp_smooth, edges = f[f'ZJetsToNuNu2017_{variation}Up_smoothed'].numpy()
    h_jerDown_smooth, edges = f[f'ZJetsToNuNu2017_{variation}Down_smoothed'].numpy()

    h_jerUp, edges = f[f'ZJetsToNuNu2017_{variation}Up'].numpy()
    h_jerDown, edges = f[f'ZJetsToNuNu2017_{variation}Down'].numpy()

    centers = ( (edges + np.roll(edges,-1)) / 2)[:-1]

    fig, ax = plt.subplots()
    ax.plot(centers, h_jerUp, 'o', label='Up')
    ax.plot(centers, h_jerUp_smooth, label='Up (smooth)')
    ax.plot(centers, h_jerDown, 'o', label='Down')
    ax.plot(centers, h_jerDown_smooth, label='Down (smooth)')

    xlim = ax.get_xlim()
    ax.plot(xlim, [1., 1.], color='k')
    ax.set_xlim(xlim)

    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.legend()

    # Save figure
    outdir = './output/smoothing_checks'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(pjoin(outdir, f'{variation}_smoothing_comp.pdf'))

def main():
    inpath = './output/merged_2020-08-30_vbfhinv_splitJECuncs_25Aug20_noJER/splitJEC/vbf/root'
    rootfile = pjoin(inpath, 'vbf_shape_jes_uncs.root')

    variations = ['jer', 'jesTotal', 'jesAbsolute']

    for variation in variations:
        plot_smoothed(rootfile, variation)

if __name__ == '__main__':
    main()
