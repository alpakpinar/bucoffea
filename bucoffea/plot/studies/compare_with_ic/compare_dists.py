#!/usr/bin/env python

import os
import sys
import re
import uproot
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

def compare_znunu_dists(ic_file, bu_file):
    f_ic = uproot.open(ic_file)

def plot_distributions(ic_file, bu_file, year, proc='znunu', jer=False):
    '''Plot comparison of Z(vv) background template from BU and IC.'''
    f_ic = uproot.open(ic_file)['shapes_prefit']
    edges = f_ic['MTR_2017_SR']['ZJETS'].edges
    centers = ((edges + np.roll(edges, -1))/2 )[:-1]

    proc_to_vals_ic = {
        'znunu' : f_ic[f'MTR_{year}_SR']['ZJETS'].values,
        'wlnu' : f_ic[f'MTR_{year}_SR']['WJETS'].values,
        'zmumu' : f_ic[f'MTR_{year}_ZMUMU']['DY'].values,
        'zee' : f_ic[f'MTR_{year}_ZEE']['DY'].values,
        'wmunu' : f_ic[f'MTR_{year}_WMUNU']['WJETS'].values,
        'wenu' : f_ic[f'MTR_{year}_WENU']['WJETS'].values
    }
    ic_vals = proc_to_vals_ic[proc]

    f_bu = uproot.open(bu_file)
    proc_to_vals_bu = {
        'znunu' : f_bu[f'sr_qcd_znunu_{year}'].values,
        'wlnu' : f_bu[f'sr_qcd_wlnu_{year}'].values,
        'zmumu' : f_bu[f'cr_qcd_zmumu_{year}'].values,
        'zee' : f_bu[f'cr_qcd_zee_{year}'].values,
        'wmunu' : f_bu[f'cr_qcd_wmunu_{year}'].values,
        'wenu' : f_bu[f'cr_qcd_wenu_{year}'].values
    }

    proc_to_title = {
        'znunu' : r'QCD $Z(\nu\nu)$',
        'wlnu' : r'QCD $W(\ell\nu)$',
        'zmumu' : r'QCD $Z(\mu\mu)$',
        'zee' : r'QCD $Z(ee)$',
        'wmunu' : r'QCD $W(\mu\nu)$',
        'wenu' : r'QCD $W(e\nu)$'
    }

    bu_vals = proc_to_vals_bu[proc]

    bin_widths = np.diff(edges)
    bu_vals /= bin_widths 

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(ic_vals, edges, ax=ax, label='IC')
    hep.histplot(bu_vals, edges, ax=ax, label='BU')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e5)
    ax.legend()

    if jer:
        title = f'JER Applied: {proc_to_title[proc]}'
    else:
        title = f'JER Not Applied: {proc_to_title[proc]}'

    ax.set_title(title)

    ratio = ic_vals / bu_vals
    rax.plot(centers, ratio, marker='o', ls='', color='k')

    rax.grid(True)
    rax.set_ylim(0.5,1.5)
    rax.set_ylabel('IC / BU')

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.,], color='red')
    rax.set_xlim(xlim)
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')

    # Save figure
    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if jer:
        outpath = pjoin(outdir, f'bu_ic_comparison_{proc}_mjj_{year}_withJER.pdf')
    else:
        outpath = pjoin(outdir, f'bu_ic_comparison_{proc}_mjj_{year}_withoutJER.pdf')
    fig.savefig(outpath)

    print(f'MSG% File saved: {outpath}')

def main():
    ic_file_withjer = './inputs/fitDiagnosticsCRonlyFit.root'
    ic_file_withoutjer = './inputs/fitDiagnosticsCRonlyFit_noJER.root'

    bu_file_withjer = './inputs/bu_input_withJER.root'
    bu_file_nojer = './inputs/bu_input_noJER.root'

    procs = ['znunu', 'wlnu', 'zmumu', 'zee', 'wmunu', 'wenu']

    for year in [2017, 2018]:
        for proc in procs:
            plot_distributions(ic_file_withjer, bu_file=bu_file_withjer, year=year, proc=proc, jer=True)
            plot_distributions(ic_file_withoutjer, bu_file=bu_file_nojer, year=year, proc=proc, jer=False)

if __name__ == '__main__':
    main()
