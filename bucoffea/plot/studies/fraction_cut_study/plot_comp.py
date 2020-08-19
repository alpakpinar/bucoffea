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
    'ak4_pt0' : hist.Bin('jetpt', r'Jet $p_T$ (GeV)', 25, 0, 1000)
}

XLABEL = {
    'ak4_pt0' : r'Jet $p_T$ (GeV)'
}

def make_plot(h, outtag, mode='data', variable='ak4_pt0'):
    '''Make the comparison plot for data or MC and save it.'''
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region')

    ax.set_xlabel('')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e8)
    titles = {
        'data' : 'EGamma 2017',
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
    hist.plotratio(h_num, h_denom, ax=rax, error_opts=data_err_opts)

    rax.set_xlabel(XLABEL[variable])
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

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        h = h.rebin('jetpt', REBIN[variable])

    # Get data and MC (EGamma)
    h_data = h.integrate('dataset', 'EGamma_2017')[re.compile('.*EmEF.*')]
    h_mc = h.integrate('dataset', re.compile('GJets_DR-0p4.*2017'))[re.compile('.*EmEF.*')]

    # Make the plots for data and MC
    make_plot(h_data, outtag, mode='data', variable=variable)
    make_plot(h_mc, outtag, mode='mc', variable=variable)

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

    plot_comparison(acc, outtag)

if __name__ == '__main__':
    main()