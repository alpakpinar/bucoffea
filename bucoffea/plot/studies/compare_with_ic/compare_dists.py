#!/usr/bin/env python

import os
import sys
import re
import uproot
import mplhep as hep
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt

pjoin = os.path.join

def get_input_files(noWeight):
    rootdir = './inputs/sync/23Sep20'
    ic_files = {
        'withWeight' : pjoin(rootdir, 'Histos_Nominal_ZJETS_2017_UPDATED.root'),
        'noWeight' : pjoin(rootdir, 'Histos_Nominal_ZJETS_2017_UPDATED_NOWEIGHT.root')
    }

    bu_file = pjoin(rootdir, 'Histos_Nominal_ZJETS_2017_BU.root')

    ic_file = ic_files['noWeight'] if noWeight else ic_files['withWeight']
    
    return ic_file, bu_file

def compare_distributions(data, variable='mjj', year=2017, noWeight=False):
    '''Compare the pre-fit templates from BU and IC.'''
    ic_file, bu_file = get_input_files(noWeight=noWeight)
    print('*'*20)
    print(f'With weights: {not noWeight}')
    print(f'BU file: {bu_file}')
    print(f'IC file: {ic_file}')
    
    f_ic = uproot.open(ic_file)
    sr_ic = f_ic['SR']['MTR']

    h_ic = sr_ic['h_SR_MTR_diCleanJet_M']

    f_bu = uproot.open(bu_file)
    if noWeight:
        h_bu = f_bu['mjj_nowgt']
    else:
        h_bu = f_bu['mjj']

    # Plot the comparison
    # assert (h_bu.edges == h_ic.edges)
    bins = h_bu.edges

    bu_vals = h_bu.values
    ic_vals = h_ic.values

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(bu_vals, bins=bins, label='BU', ax=ax)
    hep.histplot(ic_vals, bins=bins, label='IC', ax=ax)
    
    ax.set_ylabel('Counts')
    ax.legend()
    
    data_to_title = {
        'znunu' : r'QCD $Z(\nu\nu)$ {}', 
        'zmumu' : r'QCD $Z(\mu\mu)$ {}', 
        'zee' : r'QCD $Z(ee)$ {}', 
        'wlnu' : r'QCD $W(\ell\nu)$ {}', 
        'vbf' : r'VBF $H(inv)$ {}' 
    }

    title = data_to_title[data].format(year)
    ax.set_title(title)

    # Plot ratio
    ratio = bu_vals / ic_vals
    centers = ((bins + np.roll(bins, -1))/2 )[:-1]
    rax.plot(centers, ratio, marker='o', ls='', color='black')
    
    rax.set_ylabel('BU / IC')
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylim(0.9,1.1)
    rax.grid(True)

    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = './output/sync_23Sep20'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if noWeight:
        filename = f'{variable}_{year}_comp_nowgt.pdf'
    else:
        filename = f'{variable}_{year}_comp.pdf'
        
    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    procs = ['znunu', 'wlnu', 'zmumu', 'zee', 'wmunu', 'wenu']

    compare_distributions(data='znunu', noWeight=False)
    compare_distributions(data='znunu', noWeight=True)

if __name__ == '__main__':
    main()
