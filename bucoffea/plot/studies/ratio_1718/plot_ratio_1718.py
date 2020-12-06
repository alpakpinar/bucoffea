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

def region_to_title(region):
    mapping = {
        'singlemu': r'$1\mu$ CR',
        'singleel': r'$1e$ CR',
        'dimuon': r'$2\mu$ CR',
        'dielec': r'$2e$ CR',
        'photon': r'$\gamma$ CR',
    }

    try:
        title = f'{mapping[region]}: 2017/2018'
        return title
    except KeyError:
        raise ValueError(f'Could not find title for region: {region}')

def plot_ratio_1718(infile, regions):
    '''Plot data-to-data and MC-to-MC ratios from the fit diagnostics file (prefit).'''
    shapedir = infile['shapes_prefit']
    
    # Output directory to save plots
    outdir = f'./output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for region in regions:
        datadir_2017 = shapedir[f'vbf_2017_{region}']
        datadir_2018 = shapedir[f'vbf_2018_{region}']

        # Read data and total MC for two regions
        h_data = {
            '2017' : datadir_2017['data'],
            '2018' : datadir_2018['data']
        }

        h_mc = {
            '2017' : datadir_2017['total_background'],
            '2018' : datadir_2018['total_background']
        }

        # Data-to-data and MC-to-MC ratios
        r_data = h_data['2017'].yvalues / h_data['2018'].yvalues
        r_mc = h_mc['2017'].values / h_mc['2018'].values

        centers = 0.5 * np.sum(h_mc['2017'].bins, axis=1)
        edges = h_mc['2017'].edges

        # Error in data/data ratio
        yerr_data = np.vstack( (h_data['2017'].yerrorslow, h_data['2017'].yerrorshigh) ) / h_data['2018'].yvalues

        # Total uncertainty on MC ratios
        vars_mc = np.sqrt(h_mc['2017'].variances) 
        r_mc_up = (h_mc['2017'].values + vars_mc) / h_mc['2018'].values
        r_mc_down = (h_mc['2017'].values - vars_mc) / h_mc['2018'].values

        rerr_mc = np.vstack( (r_mc_down, r_mc_up) )

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

        r_mc_down = np.r_[rerr_mc[0, :], rerr_mc[0, -1]]
        r_mc_up = np.r_[rerr_mc[1, :], rerr_mc[1, -1]]

        ax.fill_between(edges, 
                y1=r_mc_down,
                y2=r_mc_up,
                **fill_opts
                )

        ax.legend(title='Ratios')
        ax.set_ylabel('Ratio')
        ax.set_title( region_to_title(region) )

        # Plot ratio of ratios in the ratio pad
        r = r_data / r_mc

        r_err = yerr_data / r_mc

        rax.errorbar(centers, r, yerr=r_err, **data_err_opts)
        rax.axhline(1, xmin=0, xmax=1, color='red')

        # MC uncertainty
        rr_mc_up = r_mc_up[:-1] / r_mc 
        rr_mc_down = r_mc_down[:-1] / r_mc 

        rax.fill_between(edges,
            y1=np.r_[rr_mc_down, rr_mc_down[-1]],
            y2=np.r_[rr_mc_up, rr_mc_up[-1]],
            **fill_opts
            )

        rax.grid(True)
        rax.set_ylim(0.5,1.5)
        rax.set_xlim(200., 5000.)
        rax.set_ylabel('Data/MC')
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')

        loc = MultipleLocator(0.5)
        rax.yaxis.set_major_locator(loc)

        outpath = pjoin(outdir, f'2017_2018_ratio_{region}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    # Path to input fit diagnostics file
    inpath = sys.argv[1]
    infile = uproot.open(inpath)

    regions = [
        'singlemu',
        'singleel',
        'dimuon',
        'dielec',
        'photon'
    ]

    plot_ratio_1718(infile, regions=regions)

if __name__ == '__main__':
    main()