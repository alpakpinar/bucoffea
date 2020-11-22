#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from bucoffea.plot.util import fig_ratio

pjoin = os.path.join

def plot_uncertainty(fin, outdir, uncertainty='jesTotal', endcap='pos'):
    '''For the given systematic uncertainty source, plot the nominal SF and the uncertainties.'''
    for year in [2017, 2018]:
        sf_nom = fin[f'sf_{endcap}_endcap_{year}'].values
        centers = 0.5 * np.sum(fin[f'sf_{endcap}_endcap_{year}'].bins, axis=1)

        # Read the stat uncertainties on the nominal SF
        sf_nom_statUp = fin[f'sf_{endcap}_endcap_{year}_statUp'].values
        sf_nom_statDown = fin[f'sf_{endcap}_endcap_{year}_statDown'].values

        sf_nom_statErr = np.vstack([
            np.abs(sf_nom_statUp - sf_nom),
            np.abs(sf_nom_statDown - sf_nom)
        ])

        # Read the systematic uncertainties, provided in the function call
        sf_sysUp = fin[f'sf_{endcap}_endcap_{year}_{uncertainty}Up'].values
        sf_sysDown = fin[f'sf_{endcap}_endcap_{year}_{uncertainty}Down'].values

        # Plot the nominal SF + variations
        fig, ax, rax = fig_ratio()
        ax.errorbar(centers, y=sf_nom, yerr=sf_nom_statErr, marker='o', label='Nominal')
        ax.plot(centers, sf_sysUp, marker='o', label=f'{uncertainty} Up')
        ax.plot(centers, sf_sysDown, marker='o', label=f'{uncertainty} Down')

        ax.set_ylabel('Data / MC SF')
        ax.set_ylim(0.8,1.2)
        ax.set_title(f'{uncertainty} Uncertainties: {year}', fontsize=14)

        if endcap == 'pos':
            figtext = r'$2.5 < \eta < 3.0$'
        elif endcap == 'neg':
            figtext = r'$-3.0 < \eta < -2.5$'

        ax.text(1., 1., figtext,
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
            )

        ax.legend(title='Uncertainties')

        # On the ratio pad, plot nominal / varied
        r_sysUp = sf_nom / sf_sysUp
        r_sysUp_err = sf_nom_statErr / sf_sysUp

        r_sysDown = sf_nom / sf_sysDown
        r_sysDown_err = sf_nom_statErr / sf_sysDown

        rax.errorbar(centers, y=r_sysUp, yerr=r_sysUp_err, marker='o', ls='', color='C1')
        rax.errorbar(centers, y=r_sysDown, yerr=r_sysDown_err, marker='o', ls='', color='C2')

        rax.set_xlabel(r'Jet $p_T \ (GeV)$')
        rax.set_ylabel('Nominal / Varied')
        rax.set_ylim(0.96,1.04)
        rax.grid(True)

        rax.axhline(1, xmin=0, xmax=1, color='black')

        rax.axhline(0.99, xmin=0, xmax=1, ls='--', color='black', label=r'$1\%$')
        rax.axhline(1.01, xmin=0, xmax=1, ls='--', color='black')

        rax.legend()

        loc = MultipleLocator(0.02)
        rax.yaxis.set_major_locator(loc)

        # Save figure
        outpath = pjoin(outdir, f'{uncertainty}_{endcap}_endcap_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    # Path to input ROOT file containing SF and its variations
    inpath = sys.argv[1]
    fin = uproot.open(inpath)

    # Output directory to save plots
    outdir = pjoin(os.path.dirname(os.path.dirname(inpath)), 'final_uncs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    uncertainties = [
        'jesTotal',
        'jer',
        'pileup',
        'prefire'
    ]

    for endcap in ['pos', 'neg']:
        for uncertainty in uncertainties:
            plot_uncertainty(fin, outdir, uncertainty=uncertainty, endcap=endcap)

if __name__ == '__main__':
    main()
