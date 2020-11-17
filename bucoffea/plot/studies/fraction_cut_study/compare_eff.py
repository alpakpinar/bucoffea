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

def ratio_unc(num, den, numunc, denunc):
    sf = num / den
    lo = numunc[0] / den
    high = numunc[1] / den
    return np.abs(np.vstack((lo, high)) - sf)

def do_coarse_rebinning_for_2d(h):
    '''Make coarser binning for 2D SF histogram.'''
    coarse_binnings = {
        'jetpt' : hist.Bin('jetpt', r'Jet $p_T \ (GeV)$', [40,80,120,160,200,300])
    }

    h = h.rebin('jetpt', coarse_binnings['jetpt'])
    return h

def calculate_efficiency(h_data, h_mc, region='cr_2m'):
    '''Calculate the efficiencies in data and MC and return the values.'''
    h_data_withCut = h_data.integrate('region', f'{region}_withEmEF')
    h_data_withoutCut = h_data.integrate('region', f'{region}_noEmEF')

    h_mc_withCut = h_mc.integrate('region', f'{region}_withEmEF')
    h_mc_withoutCut = h_mc.integrate('region', f'{region}_noEmEF')

    data_num = h_data_withCut.values(overflow='over')[()]
    data_den = h_data_withoutCut.values(overflow='over')[()]

    mc_num = h_mc_withCut.values(overflow='over')[()]
    mc_den = h_mc_withoutCut.values(overflow='over')[()]

    data_eff = data_num / data_den
    mc_eff = mc_num / mc_den

    # Calculate the uncertainties in data eff and MC eff
    clopper_pearson_interval = hist.clopper_pearson_interval
    data_eff_unc = clopper_pearson_interval(data_num, data_den)
    mc_eff_unc = clopper_pearson_interval(mc_num, mc_den)

    return data_eff, mc_eff, data_eff_unc, mc_eff_unc

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

def get_varied_sf(data_eff, h_mc, region='cr_2m'):
    '''Get the varied scale factor by varying the prefire weights.'''
    h_mc_withCut_prefireUp = h_mc.integrate('region', f'{region}_withEmEF_prefireUp')
    h_mc_withCut_prefireDown = h_mc.integrate('region', f'{region}_withEmEF_prefireDown')

    h_mc_withoutCut_prefireUp = h_mc.integrate('region', f'{region}_noEmEF_prefireUp')
    h_mc_withoutCut_prefireDown = h_mc.integrate('region', f'{region}_noEmEF_prefireDown')

    mc_eff_prefireUp = h_mc_withCut_prefireUp.values(overflow='over')[()] / h_mc_withoutCut_prefireUp.values(overflow='over')[()]
    mc_eff_prefireDown = h_mc_withCut_prefireDown.values(overflow='over')[()] / h_mc_withoutCut_prefireDown.values(overflow='over')[()]

    sf_prefireUp = data_eff / mc_eff_prefireUp
    sf_prefireDown = data_eff / mc_eff_prefireDown

    return sf_prefireUp, sf_prefireDown

def plot_2d_eff(eff, outtag, xedges, yedges, xcenters, ycenters, year=2017, type='data'):
    '''Plot 2D efficiencies for data or MC.'''
    fig, ax = plt.subplots()
    opts = {'cmap' : 'viridis'}
    pc = ax.pcolormesh(xedges, yedges, eff.T, **opts)
    fig.colorbar(pc, ax=ax, label=f'Efficiency')

    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel(r'Jet $\eta$')

    ax.set_title(f'{year} Efficiency: {type.upper()}')

    opts = {
        'horizontalalignment' : 'center',
        'verticalalignment' : 'center'
    }
    for ix, xcenter in enumerate(xcenters[2:], 2):
        for iy, ycenter in enumerate(ycenters):
            if not np.isnan(eff[ix,iy]):
                ax.text(xcenter, ycenter, f'{eff[ix, iy]:.2f}', **opts)
            else:
                continue

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{type}_2d_eff_{year}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

    plt.close(fig)

def plot_sf_for_endcap(acc, outtag, region='cr_2m', year=2017):
    '''Plot 1D SF as a function of jet pt for endcap jets.'''
    variable = 'ak4_pt0_eta0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, region, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning_for_2d(h_data)
    h_mc = do_coarse_rebinning_for_2d(h_mc)

    h_data_pos_endcap = h_data.integrate('jeteta', slice(2.5,3.0))
    h_data_neg_endcap = h_data.integrate('jeteta', slice(-3.0,-2.5))
    h_mc_pos_endcap = h_mc.integrate('jeteta', slice(2.5,3.0))
    h_mc_neg_endcap = h_mc.integrate('jeteta', slice(-3.0,-2.5))

    # Calculate efficiencies for the two endcap regions
    data_eff_pos_endcap, mc_eff_pos_endcap, data_eff_unc_pos_endcap, mc_eff_unc_pos_endcap = calculate_efficiency(h_data_pos_endcap, h_mc_pos_endcap)
    data_eff_neg_endcap, mc_eff_neg_endcap, data_eff_unc_neg_endcap, mc_eff_unc_neg_endcap = calculate_efficiency(h_data_neg_endcap, h_mc_neg_endcap)

    # Calculate SF
    sf_pos_endcap = data_eff_pos_endcap / mc_eff_pos_endcap
    sf_neg_endcap = data_eff_neg_endcap / mc_eff_neg_endcap

    # Calculate the error on SF
    sf_err_pos_endcap = ratio_unc(
        data_eff_pos_endcap,
        mc_eff_pos_endcap,
        data_eff_unc_pos_endcap,
        mc_eff_unc_pos_endcap
    )

    sf_err_neg_endcap = ratio_unc(
        data_eff_neg_endcap,
        mc_eff_neg_endcap,
        data_eff_unc_neg_endcap,
        mc_eff_unc_neg_endcap
    )

    # Guard against NaN values
    sf_pos_endcap[np.isnan(sf_pos_endcap) | np.isinf(sf_pos_endcap)] = 1.
    sf_neg_endcap[np.isnan(sf_neg_endcap) | np.isinf(sf_neg_endcap)] = 1.

    pt_ax = h_data.axis('jetpt')
    xcenters = pt_ax.centers(overflow='over')

    # Plot the 1D SF for the two regions (for now, do not plot errorbars until we figure it out)
    fig, ax = plt.subplots()
    ax.errorbar(xcenters, sf_pos_endcap, yerr=sf_err_pos_endcap, marker='o', label=r'$2.5 < \eta < 3.0$')
    ax.errorbar(xcenters, sf_neg_endcap, yerr=sf_err_neg_endcap, marker='o', label=r'$-2.5 < \eta < -3.0$')

    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel('Data/MC SF')
    ax.set_ylim(0.7,1.3)
    ax.legend()

    ax.set_title(f'Jet SFs in Endcap: {year}')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'sf_1d_{year}.pdf')

    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def get_2d_sf(acc, outtag, rootfile, region='cr_2m', year=2017):
    '''Get 2D scale factor as a function of jet pt and eta, for the efficiency of the neutral EM fraction cut.'''
    variable = 'ak4_pt0_eta0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, region, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning_for_2d(h_data)
    h_mc = do_coarse_rebinning_for_2d(h_mc)

    data_eff, mc_eff, data_eff_unc, mc_eff_unc = calculate_efficiency(h_data, h_mc)

    pt_ax = h_data.axis('jetpt')
    eta_ax = h_data.axis('jeteta')

    xedges, xcenters = pt_ax.edges(overflow='over'), pt_ax.centers(overflow='over')
    yedges, ycenters = eta_ax.edges(overflow='over'), eta_ax.centers(overflow='over')

    # Plot the efficiencies first
    plot_2d_eff(data_eff, outtag, xedges, yedges, xcenters, ycenters, year, type='data')
    plot_2d_eff(mc_eff, outtag, xedges, yedges, xcenters, ycenters, year, type='mc')

    # Scale factor: Data / MC efficiency
    sf = data_eff / mc_eff

    # Uncertainties on the scale factors: Calculated by the variation of prefire weights
    sf_up, sf_down = get_varied_sf(data_eff, h_mc, region=region)

    # Guard against NaN values
    sf[np.isnan(sf) | np.isinf(sf)] = 1.
    sf_up[np.isnan(sf_up) | np.isinf(sf_up)] = 1.
    sf_down[np.isnan(sf_down) | np.isinf(sf_down)] = 1.

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

    # Save the scale factors into ROOT file with the up/down uncertainties
    rootfile[f'sf_{year}'] = (sf, xedges, yedges)
    rootfile[f'sf_{year}_up'] = (sf_up, xedges, yedges)
    rootfile[f'sf_{year}_down'] = (sf_down, xedges, yedges)

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

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    rootpath = pjoin(outdir, 'jet_em_frac_eff_sf.root')
    rootfile = uproot.recreate(rootpath)

    # Plot efficiency comparison plots both with the regular pt balance cut, and the tighter one (<0.1)
    for year in args.years:
        for spec in all_specs:
            if not re.match(args.specs, spec):
                continue
            compare_eff(acc, outtag, region=region, spec=spec, year=year)

        # Calculate 2D scale factor and save to a root file
        get_2d_sf(acc, outtag, rootfile, region=region, year=year)

        # Calculate 1D SF as a function of jet pt for the endcap jets only
        plot_sf_for_endcap(acc, outtag, region=region, year=year)

if __name__ == '__main__':
    main()
