#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(0,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt0_over_met' : hist.Bin('jmet', r'Leading Jet $p_T$ / $MET$', 25, 0, 2),
    'dphi_ak40_met' : hist.Bin('dphi', r'$\Delta \phi(ak4_0, MET)$', 25, 0, 3.5)
}

XLABELS = {
    'mjj' : r'$M_{jj} \ (GeV)$',
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
    'ak4_pt0' : r'Leading Jet $p_T$',
    'ak4_pt1' : r'Trailing Jet $p_T$'
}

def plot_2d(acc, outtag, region):
    '''Plot 2D histogram of the two cut variables for data in SR.'''
    variable = 'dphi_ak40_met_ak4_pt0_over_met'
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin the two variables
    h = h.rebin('jmet', REBIN['ak4_pt0_over_met'])
    h = h.rebin('dphi', REBIN['dphi_ak40_met'])

    # Get the MET dataset and the relevant region
    h = h.integrate('dataset', 'MET_2017').integrate('region', region)

    fig, ax = plt.subplots()
    hist.plot2d(h, xaxis='dphi', ax=ax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, [0.7, 0.7], color='red', lw=2)
    ax.plot(xlim, [1.3, 1.3], color='red', lw=2)
    ax.plot([2.9, 2.9], ylim, color='red', lw=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Save figure
    outdir = f'./output/{outtag}/2d'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{region}_dphi_pt_over_met_2d.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_quantity(acc, outtag, variable, region):
    '''Plot the given quantity for data in given region.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        if 'ak4_pt' in variable:
            h = h.rebin('jetpt', REBIN[variable])
        else:
            h = h.rebin(variable, REBIN[variable])

    # Get the MET dataset and the relevant region
    h = h.integrate('dataset', 'MET_2017').integrate('region', region)

    # Plot the distribution
    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overflow='over')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e5)
    ax.get_legend().remove()

    if variable == 'ak4_pt0_over_met':
        ylim = ax.get_ylim()
        ax.plot([0.7,0.7], ylim, 'k--')
        ax.plot([1.3,1.3], ylim, 'k--')
        ax.set_ylim(ylim)
        print(f'Region: {region}')
        centers = h.axes()[0].centers()
        print(centers)
        mask = (centers > 0.7) & (centers < 1.3)
        total_events = sum(h.values()[()])
        print(h.values()[()][mask])
        passing_events = sum(h.values()[()][mask])

        print(f'Passing events: {passing_events}')
        print(f'Total events: {total_events}')

    elif variable == 'dphi_ak40_met':
        ylim = ax.get_ylim()
        ax.plot([2.9,2.9], ylim, 'k--')
        ax.set_ylim(ylim)

    # Get the relevant title, according to region
    titles = {
        'sr_vbf_leadak4_ee' : r'$2.9 < |\eta| < 3.3$',
        'sr_vbf_leadak4_ee_pt' : r'$2.9 < |\eta| < 3.3$ & $p_T > 100$'
    }
    title = titles[region]
    ax.set_title(title)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}_{region}_data.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['ak4_pt0_over_met', 'ak4_pt0', 'ak4_eta0', 'dphi_ak40_met']
    regions = ['sr_vbf_leadak4_ee_pt', 'sr_vbf_leadak4_ee']
    for region in regions:
        for variable in variables:
            plot_quantity(acc, outtag, variable, region)

        plot_2d(acc, outtag, region=region)

if __name__ == '__main__':
    main()