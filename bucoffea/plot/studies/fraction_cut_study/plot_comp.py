#!/usr/bin/env python

import os
import sys
import re
import argparse
import warnings
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

REBIN = {
    'ak4_pt0' : hist.Bin('jetpt', r'Jet $p_T$ (GeV)', 20, 0, 1000),
    'met' : hist.Bin('met', r'$p_T^{miss}$ (GeV)', list(range(0,60,10))),
    'muon_pt0' : hist.Bin('pt',r'Leading muon $p_{T}$ (GeV)',list(range(0,600,20)))
}

XLABELS = {
    'ak4_pt0' : r'Jet $p_T$ (GeV)',
    'ak4_eta0' : r'Jet $\eta$',
    'ak4_nef0' : 'Jet Neutral EM Fraction',
    'met' : r'$p_T^{miss}$ (GeV)',
    'muon_pt0' : r'Leading $\mu \ p_T \ (GeV)$',
    'z_pt_over_jet_pt' : r'$p_T^Z / p_T^j - 1$',
    'dphi_z_jet' : r'$\Delta\phi(Z,j)$'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to merged coffea files.')
    parser.add_argument('--years', help='Years to plot.', nargs='*', type=int, default=[2017, 2018])
    parser.add_argument('--specs', help='Regex to specify one or more of the special regions: regular, tight, tightBalCut, tightMassCut', default='regular')
    parser.add_argument('--plot_data_mc', help='If this is specified, data/MC plots will be plotted.', action='store_true')
    parser.add_argument('--distribution', help='Regex to plot for data/MC comparison plots.', default='.*')
    args = parser.parse_args()
    return args

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
        elif variable == 'muon_pt0':
            h = h.rebin('pt', REBIN[variable])

    return h

def make_plot(h, outtag, mode='data', region='cr_2m', variable='ak4_pt0', spec='regular', year=2017):
    '''Make the comparison plot for data or MC and save it.'''
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    spec_suffix = f'_{spec}' if spec != 'regular' else ''

    # Get the relevant regions
    h = h[re.compile(f'^.*EmEF{spec_suffix}$')]

    hist.plot1d(h, ax=ax, overlay='region')

    ax.set_xlabel('')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e8)
    titles = {
        'data' : {'cr_g': f'Single Photon {year}', 'cr_2m': f'Single Muon {year}'},
        'mc' : {'cr_g': f'GJets {year}', 'cr_2m': f'DY {year}'}
    }
    ax.set_title(titles[mode][region])

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k'
    }
    # Plot the ratio
    h_num = h.integrate('region', re.compile(f'.*withEmEF{spec_suffix}'))
    h_den = h.integrate('region', re.compile(f'.*noEmEF{spec_suffix}'))
    hist.plotratio(h_num, h_den, ax=rax, error_opts=data_err_opts, unc='num')

    rax.set_xlabel(XLABELS[variable])
    rax.set_ylabel('With cut / without cut')
    rax.grid(True)
    if year == 2017:
        rax.set_ylim(0.8, 1.2)
    elif year == 2018:
        rax.set_ylim(0.7, 1.3)

    # Save the figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outname = f'{variable}_comp_{region}_{mode}{spec_suffix}_{year}.pdf'
    outpath = pjoin(outdir, outname)
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_comparison(acc, outtag, variable='ak4_pt0', region='cr_2m', spec='regular', year=2017):
    '''Plot spectrum for the given variable, with and without the EM fraction cut.'''
    acc.load(variable)
    h = acc[variable]
    h = preprocess(h, acc, variable)

    # Get data and MC, for GJets or Zmumu events
    if region == 'cr_g':
        h_data = h.integrate('dataset', f'EGamma_{year}')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile(f'GJets_DR-0p4.*{year}'))[re.compile('.*EmEF.*')]
    elif region == 'cr_2m':
        h_data = h.integrate('dataset', f'SingleMuon_{year}')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))[re.compile('.*EmEF.*')]

    # Make the plots for data and MC
    make_plot(h_data, outtag, mode='data', variable=variable, region=region, spec=spec, year=year)
    make_plot(h_mc, outtag, mode='mc', variable=variable, region=region, spec=spec, year=year)

def plot_data_mc_comparison(acc, outtag, variable='ak4_pt0', mode='before_cut', region='cr_2m', spec='regular', year=2017):
    '''Plot data/MC comparison for the given variable, with or without the EM fraction cut.'''
    acc.load(variable)
    h = acc[variable]
    h = preprocess(h, acc, variable)

    spec_suffix = f'_{spec}' if spec != 'regular' else ''

    # Get the relevant region
    if mode == 'before_cut':
        h = h.integrate('region', f'{region}_noEmEF{spec_suffix}')
    elif mode == 'after_cut':
        h = h.integrate('region', f'{region}_withEmEF{spec_suffix}')

    h = h[re.compile(f'.*{year}')]

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
    if region == 'cr_g':
        h_num = h.integrate('dataset', f'EGamma_{year}')
        h_denom = h.integrate('dataset', re.compile(f'GJets_DR-0p4.*{year}'))
    elif region == 'cr_2m':
        h_num = h.integrate('dataset', f'SingleMuon_{year}')
        h_denom = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))
        
    hist.plotratio(h_num, h_denom, ax=rax, error_opts=data_err_opts, unc='num')

    rax.set_xlabel(XLABELS[variable])
    rax.set_ylabel('Data / MC')
    rax.set_ylim(0.4,1.6)
    rax.grid(True)

    if variable == 'ak4_nef0':
        ylim = rax.get_ylim()
        rax.plot([0.7, 0.7], ylim, color='red')
        rax.set_ylim(ylim)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1, 1], color='red')
    rax.set_xlim(xlim)

    # Save the figure
    outdir = f'./output/{outtag}/data_mc_comp'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outname = f'data_mc_comp_{variable}_{region}_{mode}{spec_suffix}_{year}.pdf'
    outpath = pjoin(outdir, outname)
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    args = parse_cli()
    inpath = args.inpath
    
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

    # Determine the type of events from the submission title
    if 'gjets' in outtag:
        region = 'cr_g'
    else:
        region = 'cr_2m'

    # Variables to plot
    variables = ['ak4_pt0', 'ak4_eta0', 'ak4_nef0', 'muon_pt0', 'met', 'dphi_z_jet']

    all_specs = [
        'regular',
        'tight',
        'tightBalCut',
        'tightMassCut',
        'very_tight'
        ]

    for year in args.years:
        for variable in variables:
            if not re.match(args.distribution, variable):
                continue
            for spec in all_specs:
                if not re.match(args.specs, spec):
                    continue
                # Plot comparison with the normal pt balance cut and the tighter one
                plot_comparison(acc, outtag, variable=variable, region=region, spec=spec, year=year)
            # Plot data/MC comparison plots before and after the EM fraction cut, if requested
                if args.plot_data_mc:
                    plot_data_mc_comparison(acc, outtag, variable=variable, mode='before_cut', region=region, spec=spec, year=year)
                    plot_data_mc_comparison(acc, outtag, variable=variable, mode='after_cut', region=region, spec=spec, year=year)

if __name__ == '__main__':
    main()
