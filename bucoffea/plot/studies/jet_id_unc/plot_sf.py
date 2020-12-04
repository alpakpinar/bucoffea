#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt
from bucoffea.helpers.paths import bucoffea_path

pjoin = os.path.join

def plot_sf(infile):
    for year in [2017, 2018]:
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        
        pos_sf_nom = infile[f'sf_pos_endcap_{year}'].values
        pos_sf_up = infile[f'sf_pos_endcap_{year}_up'].values
        pos_sf_down = infile[f'sf_pos_endcap_{year}_down'].values

        neg_sf_nom = infile[f'sf_neg_endcap_{year}'].values
        neg_sf_up = infile[f'sf_neg_endcap_{year}_up'].values
        neg_sf_down = infile[f'sf_neg_endcap_{year}_down'].values

        centers = 0.5 * np.sum(infile[f'sf_pos_endcap_{year}'].bins, axis=1)

        axes[0].plot(centers, pos_sf_nom, label='Nominal', marker='o')
        axes[0].plot(centers, pos_sf_up, label='SF up', marker='o')
        axes[0].plot(centers, pos_sf_down, label='SF down', marker='o')

        axes[0].set_title(r'$2.5 < \eta < 3.0$')
        axes[0].set_xlabel(r'$M_{jj} \ (GeV)$')
        axes[0].legend()

        axes[1].plot(centers, neg_sf_nom, label='Nominal', marker='o')
        axes[1].plot(centers, neg_sf_up, label='SF up', marker='o')
        axes[1].plot(centers, neg_sf_down, label='SF down', marker='o')

        axes[1].set_title(r'$-3.0 < \eta < -2.5$')
        axes[1].set_xlabel(r'$M_{jj} \ (GeV)$')
        axes[1].legend()

        outdir = './output/jet_sf_plots'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'jet_sf_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    infilepath = bucoffea_path('data/sf/ak4/jet_id_sf_for_endcaps.root')
    infile = uproot.open(infilepath)

    plot_sf(infile)

if __name__ == '__main__':
    main()