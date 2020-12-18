#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import uproot
import mplhep as hep

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from bucoffea.plot.util import fig_ratio

pjoin = os.path.join

def region_to_title(region):
    mapping = {
        'singlemu' : r'$1\mu$ CR',
        'singleel' : r'$1e$ CR',
        'dimuon' : r'$2\mu$ CR',
        'dielec' : r'$2e$ CR',
        'photon' : r'$\gamma$ CR',
    }

    return mapping[region]

def compute_ratio_and_errors(histos):
    '''Given the dictionary of histograms, compute data/data and bkg/bkg ratios and the errors.'''
    # Calculate and plot data/data and bkg/bkg ratios
    r_bkg = histos['hotAndColdVeto']['total_bkg'].values / histos['noVeto']['total_bkg'].values
    r_data = histos['hotAndColdVeto']['data'].yvalues / histos['noVeto']['data'].yvalues

    # Errors on the ratio
    r_bkg_err = np.sqrt(histos['hotAndColdVeto']['total_bkg'].variances) / histos['noVeto']['total_bkg'].values

    r_data_err = np.vstack((
        histos['hotAndColdVeto']['data'].yerrorslow / histos['noVeto']['data'].yvalues,
        histos['hotAndColdVeto']['data'].yerrorshigh / histos['noVeto']['data'].yvalues,
    ))

    return r_bkg, r_bkg_err, r_data, r_data_err

def plot_ratios(files, region, year):
    '''Plot prefit ratios of veto vs. no-veto for the specified control region and year.'''
    histos = {}
    for tag, f in files.items():
        histos[tag] = {
            'total_bkg' : f['shapes_prefit'][f'vbf_{year}_{region}']['total_background'],
            'data' : f['shapes_prefit'][f'vbf_{year}_{region}']['data'],
        }

    edges = histos['hotAndColdVeto']['total_bkg'].edges
    centers = 0.5 * np.sum(histos['hotAndColdVeto']['total_bkg'].bins, axis=1)

    r_bkg, r_bkg_err, r_data, r_data_err = compute_ratio_and_errors(histos)

    # Plot!
    fig, ax, rax = fig_ratio()
    hep.histplot(r_data, edges, 
            ax=ax, 
            yerr=r_data_err, 
            histtype='errorbar', 
            label='Data/Data',
            **{'color': 'k', 'markersize': 12.}
            )
    
    hep.histplot(r_bkg, edges, 
            ax=ax, 
            yerr=r_bkg_err, 
            histtype='step',
            label='Total bkg/Total bkg'
            )

    ax.legend(title='Ratios')

    ax.set_title('With Veto / No Veto', fontsize=14)
    ax.set_ylim(0,1.5)
    ax.set_ylabel('Ratio', fontsize=14)

    ax.text(1., 1., year,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    ax.text(0., 1., region_to_title(region),
        fontsize=14,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    # Plot ratio of data/MC
    rr = r_data / r_bkg
    rr_err = r_data_err / r_bkg
    
    data_err_opts = {
        'color' : 'k',
        'linestyle' : '',
        'marker' : 'o',
    }

    rax.errorbar(centers, rr, yerr=rr_err, **data_err_opts)

    rax.set_xlabel(r'$M_{jj} \ (GeV)$', fontsize=14)
    rax.set_ylabel('Data / MC', fontsize=14)
    rax.grid(True)
    rax.set_ylim(0.8,1.2)

    loc = MultipleLocator(0.1)
    rax.yaxis.set_major_locator(loc)

    rax.axhline(1, xmin=0, xmax=1, color='red')

    # Save figure
    outdir = f'./output/ratios'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'ratios_{region}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = './input/fitdiag'
    inputfiles = {
        'noVeto' : uproot.open( pjoin(inpath, 'fitDiagnostics_vbf_combined_noVeto.root') ),
        'hotAndColdVeto' : uproot.open( pjoin(inpath, 'fitDiagnostics_vbf_combined_hotAndColdVeto.root') )
    }

    regions = [
        'singlemu',
        'singleel',
        'dimuon',
        'dielec',
        'photon'
    ]

    for year in [2017, 2018]:
        for region in regions:
            plot_ratios(inputfiles, region=region, year=year)

if __name__ == '__main__':
    main()