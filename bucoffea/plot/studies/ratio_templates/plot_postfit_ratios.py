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

def get_lumi(year):
    if year == 2017:
        return 41.5
    elif year == 2018:
        return 59.7

def get_ylabel_from_regionpair(region1, region2):
    mapping = {
        'dielec' : r'$Z(ee)$',
        'dimuon' : r'$Z(\mu\mu)$',
        'singleel' : r'$W(e\nu)$',
        'singlemu' : r'$W(\mu\nu)$',
        'photon' : r'$\gamma$ + jets',
    }

    try:
        ylabel = f'{mapping[region1]} / {mapping[region2]}'
        return ylabel
    except KeyError:
        raise ValueError(f'Could not set up y-label for regions: {region1}, {region2}')

def plot_postfit_ratios(infile, region_pairs):
    '''Plot template ratios for postfit, given the input fit diagnostics file.'''
    # Output directory to save plots
    outdir = f'./output/'
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

            ax.legend(title='Ratios')
            ax.set_ylabel( get_ylabel_from_regionpair(region1, region2) )

            ax.set_title('Post-fit Ratio', fontsize=14)

            ax.text(0.1, 1., year,
                fontsize=14,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
            )

            ax.text(1., 1., r'${} \ fb^{{-1}}$ (13 TeV)'.format(get_lumi(year)),
                fontsize=14,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
            )

            # Plot ratio of ratios
            r = r_data / r_mc
            r_err = yerr_data/ r_mc
            
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

            # Save figure
            outpath = pjoin(outdir, f'ratio_{region1}_{region2}_{year}.pdf')
            fig.savefig(outpath)
            plt.close(fig)

            print(f'File saved: {outpath}')

def main():
    # Path to input fit diagnostics file
    inpath = sys.argv[1]
    infile = uproot.open(inpath)

    # Pairs of regions for which we're going to take the ratio
    region_pairs = [
        ('dielec', 'dimuon'),
        ('singleel', 'singlemu'),
        ('singleel', 'photon'),
        ('singlemu', 'photon'),
        ('dielec', 'singleel'),
        ('dimuon', 'singlemu'),
        ('dielec', 'photon'),
        ('dimuon', 'photon'),
    ]

    plot_postfit_ratios(infile, region_pairs)

if __name__ == '__main__':
    main()