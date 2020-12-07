#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from bucoffea.plot.util import fig_ratio

pjoin = os.path.join

def plot_postfit_ratios(infile, outtag, region_pairs):
    '''Plot template ratios for postfit, given the input fit diagnostics file.'''
    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shapedir = infile['shapes_fit_s']

    for year in [2017, 2018]:
        for region1, region2 in region_pairs:
            hdir1 = shapedir[f'vbf_{year}_{region1}']
            hdir2 = shapedir[f'vbf_{year}_{region2}']

            h_data = {
                region1: hdir1['data'],
                region2: hdir2['data'],
            }

            h_mc = {
                region1: hdir1['total_background'],
                region2: hdir2['total_background'],
            }

            # Compute data/data and MC/MC ratios
            r_data = h_data[region1].yvalues / h_data[region2].yvalues
            r_mc = h_mc[region1].values / h_mc[region2].values

            centers = 0.5 * np.sum(h_mc[region1].bins, axis=1)
            edges = h_mc[region1].edges

            # Error in data/data ratio
            yerr_data = np.vstack((
                h_data[region1].yerrorslow,
                h_data[region1].yerrorshigh
            )) / h_data[region2].yvalues

            # Total uncertainty on MC/MC ratios
            vars_mc = np.sqrt(h_mc[region1].variances)
            r_mc_up = (h_mc[region1].values + vars_mc) / h_mc[region2].values
            r_mc_down = (h_mc[region1].values - vars_mc) / h_mc[region2].values

            yerr_mc = np.vstack((
                r_mc_up,
                r_mc_down
            ))

            fig, ax, rax = fig_ratio()
        
            data_err_opts = {
                'marker' : 'o',
                'linestyle' : '',
                'color' : 'k',
            }
    
            # Plot data and MC ratios together        
            ax.errorbar(centers, r_data, yerr=yerr_data, label='Data/Data', **data_err_opts)
            hep.histplot(r_mc, edges, ax=ax, label='MC/MC')
    
            fill_opts = {
                'color' : 'gray',
                'alpha' : 0.5,
                'step' : 'post',
                'label' : 'Total Unc. on MC'
            }
    
            r_mc_down = np.r_[yerr_mc[0, :], yerr_mc[0, -1]]
            r_mc_up = np.r_[yerr_mc[1, :], yerr_mc[1, -1]]
    
            ax.fill_between(edges, 
                    y1=r_mc_down,
                    y2=r_mc_up,
                    **fill_opts
                    )

            # TODO: Continue here --> Ratio pad & save figure
            # fig.savefig(f'test_{year}.pdf')
            plt.close(fig)

def main():
    # Path to input fit diagnostics file
    inpath = sys.argv[1]
    infile = uproot.open(inpath)

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    regions = [
        'singlemu',
        'singleel',
        'dimuon',
        'dielec',
        'photon'
    ]
    
    # Pairs of regions for which we're going to take the ratio
    region_pairs = [
        ('dielec', 'dimuon')
    ]

    plot_postfit_ratios(infile, outtag, region_pairs)

if __name__ == '__main__':
    main()