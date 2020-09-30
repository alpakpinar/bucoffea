#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to merged coffea files.')
    parser.add_argument('--years', help='Years to plot.', nargs='*', type=int, default=[2017, 2018])
    parser.add_argument('--specs', help='Regex for specific region types: regular, tight, tightBalCut, tightMassCut', default='regular')
    args = parser.parse_args()
    return args

def preprocess(h, acc, region, year):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant dataset and regions
    if region == 'cr_g':
        h_data = h.integrate('dataset', f'EGamma_{year}')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile(f'GJets_DR-0p4.*{year}'))[re.compile('.*EmEF.*')]
    elif region == 'cr_2m':
        h_data = h.integrate('dataset', f'SingleMuon_{year}')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))[re.compile('.*EmEF.*')]

    return h_data, h_mc

def do_coarse_rebinning_for_2d(h):
    '''Make coarser binning for 2D SF histogram.'''
    coarse_binnings = {
        'jetpt' : hist.Bin('jetpt', r'Jet $p_T \ (GeV)$', list(range(100,400,100)) + [400, 600, 1000])
    }

    h = h.rebin('jetpt', coarse_binnings['jetpt'])
    return h

def compare_eff(acc, outtag, region='cr_2m', spec='regular', year=2017):
    '''Calculate the efficiency of neutral EM fraction cut as a function of the jet eta, plot the efficiency for data and MC.'''
    acc.load('ak4_eta0')
    h = acc['ak4_eta0']

    h_data, h_mc = preprocess(h, acc, region, year)

    # Get the event yields with and without the fraction cut applied
    spec_suffix = f'_{spec}' if spec != 'regular' else ''

    h_data_withCut = h_data.integrate('region', f'{region}_withEmEF{spec_suffix}')
    h_data_withoutCut = h_data.integrate('region', f'{region}_noEmEF{spec_suffix}')

    h_mc_withCut = h_mc.integrate('region', f'{region}_withEmEF{spec_suffix}')
    h_mc_withoutCut = h_mc.integrate('region', f'{region}_noEmEF{spec_suffix}')

    # Calculate and plot efficiencies for data and MC
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    labels = {
        'data' : {'cr_2m' : 'Single Muon', 'cr_g' : 'Single Photon'},
        'mc' : {'cr_2m' : 'DY', 'cr_g' : 'GJets'}
    }

    hist.plotratio(h_data_withCut, h_data_withoutCut, ax=ax, error_opts=data_err_opts, label=labels['data'][region])
    hist.plotratio(h_mc_withCut, h_mc_withoutCut, ax=ax, error_opts=data_err_opts, clear=False, label=labels['mc'][region])

    ax.set_ylabel('Efficiency')
    if year == 2017:
        ax.set_ylim(0.8,1.1)
    else:
        ax.set_ylim(0.6,1.1)
    ax.grid(True)
    ax.legend()

    # Plot the double ratio on the ratio pad (scale factor)
    ratio_num = h_data_withCut.values()[()] / h_data_withoutCut.values()[()] 
    ratio_denom = h_mc_withCut.values()[()] / h_mc_withoutCut.values()[()] 
    dratio = ratio_num / ratio_denom
    centers = h_data_withCut.axes()[0].centers()

    rax.plot(centers, dratio, marker='o', ls='', color='k')
    rax.set_xlabel(r'Jet $\eta$')
    rax.set_ylabel('Data / MC SF')
    rax.set_ylim(0.8,1.2)
    rax.grid(True)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'eff_comparison_data_mc_{region}{spec_suffix}_{year}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def get_2d_sf(acc, outtag, region='cr_2m', year=2017):
    '''Get 2D scale factor as a function of jet pt and eta, for the efficiency of the neutral EM fraction cut.'''
    variable = 'ak4_pt0_eta0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, region, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning_for_2d(h_data)
    h_mc = do_coarse_rebinning_for_2d(h_mc)

    h_data_withCut = h_data.integrate('region', f'{region}_withEmEF')
    h_data_withoutCut = h_data.integrate('region', f'{region}_noEmEF')

    h_mc_withCut = h_mc.integrate('region', f'{region}_withEmEF')
    h_mc_withoutCut = h_mc.integrate('region', f'{region}_noEmEF')

    data_eff = h_data_withCut.values()[()] / h_data_withoutCut.values()[()]
    mc_eff = h_mc_withCut.values()[()] / h_mc_withoutCut.values()[()]

    pt_ax = h_data_withCut.axis('jetpt')
    eta_ax = h_data_withCut.axis('jeteta')

    xedges, xcenters = pt_ax.edges(), pt_ax.centers()
    yedges, ycenters = eta_ax.edges(), eta_ax.centers()

    # Scale factor: Data / MC efficiency
    sf = data_eff / mc_eff

    # Guard against NaN values
    sf[np.isnan(sf) | np.isinf(sf)] = 1.

    # Plot the 2D scale factor
    fig, ax = plt.subplots()
    opts = {'cmap' : 'viridis'}
    pc = ax.pcolormesh(xedges, yedges, sf.T, **opts)
    fig.colorbar(pc, ax=ax, label='Data/MC SF')

    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel(r'Jet $\eta$')

    ax.set_title(f'{year} SF')

    opts = {
        'horizontalalignment' : 'center',
        'verticalalignment' : 'center'
    }
    for ix, xcenter in enumerate(xcenters):
        for iy, ycenter in enumerate(ycenters):
            ax.text(xcenter, ycenter, f'{sf[ix, iy]:.2f}', **opts)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{region}_2d_sf_{year}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

    plt.close(fig)

    # Save the scale factors into ROOT file
    rootpath = pjoin(outdir, 'eff_sf.root')
    f = uproot.recreate(rootpath)

    f['jet_eff_scale_fac'] = (sf, xedges, yedges)
    print(f'MSG% SF saved to ROOT file: {rootpath}')

def main():
    args = parse_cli()
    inpath = args.inpath

    acc = dir_archive(
        inpath,
        memsize=1e3,
        serialized=True,
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

    all_specs = [
        'regular',
        'tight',
        'tightBalCut',
        'tightMassCut',
        'very_tight'
    ]

    # Plot efficiency comparison plots both with the regular pt balance cut, and the tighter one (<0.1)
    for year in args.years:
        for spec in all_specs:
            if not re.match(args.specs, spec):
                continue
            compare_eff(acc, outtag, region=region, spec=spec, year=year)

        # Calculate 2D scale factor and save to a root file
        get_2d_sf(acc, outtag, region=region, year=year)

if __name__ == '__main__':
    main()
