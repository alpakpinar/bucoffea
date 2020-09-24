#!/usr/bin/env python

import os
import sys
import re
import uproot
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

def get_input_trees(region):
    '''Get the relevant input trees from both sides for pre-fit shape comparison.'''
    rootdir = './inputs/prefit_shapes'
    ic_file = pjoin(rootdir, f'IC_VBF_shapes_{region}.root')
    bu_file = pjoin(rootdir, f'BU_VBF_shapes_{region}.root')

    return ic_file, bu_file

def get_procs(region):
    '''Get list of processes for the given region.'''
    region_to_procs = {
        'SR' : ['data_obs', 'VBFHtoInv', 'DY', 'EWKW', 'EWKZll', 'EWKZNUNU', 'TOP', 'VV', 'WJETS', 'ZJETS'],
        'Wenu' : ['data_obs', 'EWKW', 'WJETS'],
        'Wmunu' : ['data_obs', 'EWKW', 'WJETS'],
        'Zee' : ['data_obs', 'EWKZll', 'ZJETS'],
        'Zmumu' : ['data_obs', 'EWKZll', 'ZJETS']
    }

    return region_to_procs[region]

def compare_prefit_shapes(ic_file, bu_file, region, year=2017):
    '''Compare pre-fit shapes as a function of mjj.'''
    processes = get_procs(region)

    print('*'*20)
    print(f'Region: {region}')
    print('*'*20)

    for process in processes:
        print(f'Process: {process}')
        directory = f'{region}VBF' if region != 'SR' else 'SR'
        h_ic = ic_file[directory][process]
        h_bu = bu_file[process]

        # Binning must be the same between the two
        assert((h_ic.edges == h_bu.edges).all())
        edges = h_ic.edges
        
        ic_vals = h_ic.values
        bu_vals = h_bu.values

        # Plot comparison of values
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        hep.histplot(ic_vals, edges, ax=ax, label='IC')
        hep.histplot(bu_vals, edges, ax=ax, label='BU')

        ax.legend()

        # Plot ratio
        ratio = bu_vals / ic_vals
        centers = ( (edges + np.roll(edges,-1))/2 )[:-1]

        rax.plot(centers, ratio, marker='o', ls='')

        rax.grid(True)
        rax.set_ylim(0.8,1.2)
        rax.set_ylabel('BU / IC')
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')

        # Save figure
        outdir = './output/prefit_shape_comparison'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        outpath = pjoin(outdir, f'{process}_comp.pdf')
        fig.savefig(outpath)
        print(f'MSG% File saved: {outpath}')

        plt.close(fig)

def main():
    region = sys.argv[1]
    ic_file, bu_file = get_input_trees(region)

    compare_prefit_shapes(ic_file, bu_file, region='SR')

if __name__ == '__main__':
    main()

