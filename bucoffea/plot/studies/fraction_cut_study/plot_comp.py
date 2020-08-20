#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'ak4_pt0' : hist.Bin('jetpt', r'Jet $p_T$ (GeV)', 25, 0, 1000),
    'met' : hist.Bin('met', r'$p_T^{miss}$ (GeV)', list(range(0,500,50)))
}

XLABELS = {
    'ak4_pt0' : r'Jet $p_T$ (GeV)',
    'ak4_eta0' : r'Jet $\eta$',
    'ak4_nef0' : 'Jet Neutral EM Fraction',
    'met' : r'$p_T^{miss}$ (GeV)'
}

def preprocess(h, acc, variable):
    '''Preprocessing for histograms: scaling + rebinning.'''
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        if variable == 'ak4_pt0':
            h = h.rebin('jetpt', REBIN[variable])
        elif variable == 'met':
            h = h.rebin('met', REBIN[variable])
    
    return h

def make_plot(h, outtag, mode='data', variable='ak4_pt0'):
    '''Make the comparison plot for data or MC and save it.'''
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region')

    ax.set_xlabel('')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e8)
    titles = {
        'data' : 'Single Photon 2017',
        'mc' : 'GJets 2017',
    }
    ax.set_title(titles[mode])

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k'
    }
    # Plot the ratio
    h_num = h.integrate('region', 'cr_g_withEmEF')
    h_denom = h.integrate('region', 'cr_g_noEmEF')
    hist.plotratio(h_num, h_denom, ax=rax, error_opts=data_err_opts, unc='num')

    rax.set_xlabel(XLABELS[variable])
    rax.set_ylabel('With cut / without cut')
    rax.grid(True)
    rax.set_ylim(0.8, 1.2)

    # Save the figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outname = f'{variable}_comp_{mode}.pdf'
    outpath = pjoin(outdir, outname)
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_comparison(acc, outtag, variable='ak4_pt0'):
    '''Plot spectrum for the given variable, with and without the EM fraction cut.'''
    acc.load(variable)
    h = acc[variable]
    h = preprocess(h, acc, variable)

    # Get data and MC (EGamma)
    h_data = h.integrate('dataset', 'EGamma_2017')[re.compile('.*EmEF.*')]
    h_mc = h.integrate('dataset', re.compile('GJets_DR-0p4.*2017'))[re.compile('.*EmEF.*')]

    # Make the plots for data and MC
    make_plot(h_data, outtag, mode='data', variable=variable)
    make_plot(h_mc, outtag, mode='mc', variable=variable)

def plot_data_mc_comparison(acc, outtag, variable='ak4_pt0'):
    '''Plot data/MC comparison for the given variable, without the EM fraction cut.'''
    acc.load(variable)
    h = acc[variable]
    h = preprocess(h, acc, variable)

    # Get the region without EM fraction cut applied
    h = h.integrate('region', 'cr_g_noEmEF')

    # Plot the two histograms, and their ratio
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='dataset')

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e8)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k'
    }
    h_num = h.integrate('dataset', 'EGamma_2017')
    h_denom = h.integrate('dataset', re.compile('GJets_DR-0p4.*2017'))
    hist.plotratio(h_num, h_denom, ax=rax, error_opts=data_err_opts, unc='num')

    rax.set_xlabel(XLABELS[variable])
    rax.set_ylabel('Data / MC')
    rax.set_ylim(0.8,1.2)
    rax.grid(True)

    if variable == 'ak4_nef0':
        ylim = rax.get_ylim()
        rax.plot([0.7, 0.7], ylim, color='red')
        rax.set_ylim(ylim)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1, 1], color='red')
    rax.set_xlim(xlim)

    # Save the figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outname = f'data_mc_comp_{variable}.pdf'
    outpath = pjoin(outdir, outname)
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(
        inpath,
        serialized=True,
        memsize=1e3,
        compression=0
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    # Variables to plot
    variables = ['ak4_pt0', 'ak4_eta0', 'ak4_nef0']

    for variable in variables:
        plot_comparison(acc, outtag, variable=variable)
        plot_data_mc_comparison(acc, outtag, variable=variable)

if __name__ == '__main__':
    main()