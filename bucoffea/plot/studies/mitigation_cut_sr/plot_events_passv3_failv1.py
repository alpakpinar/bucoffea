#!/usr/bin/env python

import os
import sys
import re
import warnings
import argparse
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

XLABELS = {
    'ak4_eta0' : r'Leading jet $\eta$',
    'ak4_eta1' : r'Trailing jet $\eta$',
    'ak4_phi0' : r'Leading jet $\phi$',
    'ak4_phi1' : r'Trailing jet $\phi$',
    'ak4_nef0' : r'Leading jet neutral EM fraction',
    'ak4_nef1' : r'Trailing jet neutral EM fraction',
    'ak4_nhf0' : r'Leading jet neutral hadron fraction',
    'ak4_nhf1' : r'Trailing jet neutral hadron fraction'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--plot_spec', help='Plot distributions for events with specific jets, with very low/high fractions.', action='store_true')
    parser.add_argument('--plot_eta_phi', help='Plot 2D distributions for jet eta and phi.', action='store_true')
    args = parser.parse_args()
    return args

def prepare_histogram(h, acc):
    '''Pre-processing the histogram.'''
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant region
    h = h.integrate('region', 'sr_vbf_passv3_failv1').integrate('dataset', 'MET_2017')

    return h

def plot_events(acc, outtag, variable):
    '''Plot events which pass the v3 selection but fail the v1 selection.'''
    acc.load(variable)
    h = acc[variable]

    h = prepare_histogram(h, acc)

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)
    ax.get_legend().remove()
    ax.set_title('Events passing EEv3, failing EEv1')
    ax.set_xlabel(XLABELS[variable])

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_jet_eta_phi(acc, outtag):
    '''Plot 2D histogram of jet eta and phi, for events that pass v3 selection but fail v1.'''
    acc.load('ak4_eta0_phi0')
    h = acc['ak4_eta0_phi0']
    
    h = prepare_histogram(h, acc)

    # Rebinning, use coarser axes for jet eta and phi
    new_bins = {
        'jeteta' : hist.Bin('jeteta', r'Leading Jet $\eta$', 25, -5, 5),
        'jetphi' : hist.Bin('jetphi', r'Leading Jet $\phi$', 25, -np.pi, np.pi),
    }
    for axis in ['jeteta', 'jetphi']:
        h = h.rebin(axis, new_bins[axis])

    fig, ax = plt.subplots()
    vals = h.values()[()]
    patch_opts = {'norm' : colors.LogNorm(vmin=1e-1, vmax=vals.max())}
    hist.plot2d(h, ax=ax, xaxis='jeteta', patch_opts=patch_opts)
    ax.set_title(f'Events passing EEv3, failing EEv1')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'jet_eta_phi.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_events_specific_jets(acc, outtag, variable, spec='lowEmEF'):
    '''Plot events which pass the v3 selection but fail the v1 selection.
    Leading jet in these events also has either very low or very high fractions.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant region
    h = h.integrate('region', f'sr_vbf_passv3_failv1_{spec}').integrate('dataset', 'MET_2017')

    spec_to_title = {
        'lowEmEF' : r'$NeEmEF < 0.1$',
        'highEmEF' : r'$NeEmEF > 0.9$',
        'lowHEF' : r'$NeHEF < 0.1$',
        'highHEF' : r'$NeHEF > 0.9$'
    }

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)
    ax.get_legend().remove()
    ax.set_title(f'Events passing EEv3, failing EEv1 {spec_to_title[spec]}')
    ax.set_xlabel(XLABELS[variable])

    # Save figure
    outdir = f'./output/{outtag}/{spec}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    args = parse_cli()
    inpath = args.inpath
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['ak4_eta0', 'ak4_eta1', 'ak4_phi0', 'ak4_phi1', 'ak4_nef0', 'ak4_nef1', 'ak4_nhf0', 'ak4_nhf1']
    for variable in variables:
        plot_events(acc, outtag, variable=variable)

        # If requested, plot events where jets have either very low or very high energy fractions
        # (Specified by "spec": e.g. low_EmEF refers to jets with neutral EM energy fraction < 0.1)
        if args.plot_spec:
            specs = ['lowEmEF', 'highEmEF', 'lowHEF', 'highHEF']
            for spec in specs:
                plot_events_specific_jets(acc, outtag, variable, spec=spec)

    # 2D eta/phi histogram
    if args.plot_eta_phi:
        plot_jet_eta_phi(acc, outtag)

if __name__ == '__main__':
    main()
