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

def compare_eff(acc, outtag, region='cr_2m'):
    '''Calculate the efficiency of neutral EM fraction cut as a function of the jet eta, plot the efficiency for data and MC.'''
    acc.load('ak4_eta0')
    h = acc['ak4_eta0']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant dataset and regions
    if region == 'cr_g':
        h_data = h.integrate('dataset', 'EGamma_2017')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile('GJets_DR-0p4.*2017'))[re.compile('.*EmEF.*')]
    elif region == 'cr_2m':
        h_data = h.integrate('dataset', 'DoubleMuon_2017')[re.compile('.*EmEF.*')]
        h_mc = h.integrate('dataset', re.compile('DYJetsToLL.*2017'))[re.compile('.*EmEF.*')]

    # Get the event yields with and without the fraction cut applied
    h_data_withCut = h_data.integrate('region', f'{region}_withEmEF')
    h_data_withoutCut = h_data.integrate('region', f'{region}_noEmEF')

    h_mc_withCut = h_mc.integrate('region', f'{region}_withEmEF')
    h_mc_withoutCut = h_mc.integrate('region', f'{region}_noEmEF')

    # Calculate and plot efficiencies for data and MC
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    labels = {
        'data' : {'cr_2m' : 'Double Muon', 'cr_g' : 'Single Photon'},
        'mc' : {'cr_2m' : 'DY', 'cr_g' : 'GJets'}
    }

    hist.plotratio(h_data_withCut, h_data_withoutCut, ax=ax, error_opts=data_err_opts, label=labels['data'][region])
    hist.plotratio(h_mc_withCut, h_mc_withoutCut, ax=ax, error_opts=data_err_opts, clear=False, label=labels['mc'][region])

    ax.set_ylabel('Efficiency')
    ax.set_ylim(0.8,1.1)
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
    
    outpath = pjoin(outdir, f'eff_comparison_data_mc_{region}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
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
    if 'zmumu' in outtag:
        region = 'cr_2m'
    else:
        region = 'cr_g'

    compare_eff(acc, outtag, region=region)

if __name__ == '__main__':
    main()