#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from matplotlib import pyplot as plt

pjoin = os.path.join

def plot_uncertainties(f, outdir):
    '''Plot up and down uncertainties as a function of jet pt and eta.'''
    for year in [2017,2018]:
        h = f[f'sf_{year}']
        sf = h.values
        pt_edges = h.edges[0]
        eta_edges = h.edges[1]
        pt_centers = 0.5 * np.sum(h.bins[0], axis=1)
        eta_centers = 0.5 * np.sum(h.bins[1], axis=1)
        
        for variation in ['up', 'down']:
            sf_var = f[f'sf_{year}_{variation}'].values
            # Calculate percent difference
            percent_diff = (sf_var - sf) / sf * 100

            # Plot!
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(pt_edges, eta_edges, percent_diff.T)
            cb = fig.colorbar(pc, ax=ax)
            cb.set_label('% Uncertainty')

            ax.set_title(f'{year}: Prefire {variation.capitalize()}')
            ax.set_xlabel(r'Jet $p_T \ (GeV)$')
            ax.set_ylabel(r'Jet $\eta$')

            opts = {
                'horizontalalignment' : 'center',
                'verticalalignment' : 'center'
            }
            for ix, xcenter in enumerate(pt_centers):
                for iy, ycenter in enumerate(eta_centers):
                    ax.text(xcenter, ycenter, f'{percent_diff[ix, iy]:.2f}', **opts)
        

            outpath = pjoin(outdir, f'sf_uncs_{year}_{variation}.pdf')
            fig.savefig(outpath)
            plt.close(fig)

            print(f'File saved: {outpath}')

def main():
    # ROOT file containing the SF and uncertainties
    input_root_path = sys.argv[1]
    f = uproot.open(input_root_path)

    # Output path to save plots into
    outdir = pjoin(os.path.dirname(input_root_path), 'uncertainties')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plot_uncertainties(f, outdir)

if __name__ == '__main__':
    main()
