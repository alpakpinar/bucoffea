#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
import matplotlib.ticker
import argparse
from pprint import pprint

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_19Feb20', help='Path to merged coffea files for 19Feb20 skim.')
    parser.add_argument('path_05Jun20v5', help='Path to merged coffea files for 05Jun20v5 skim.')
    parser.add_argument('--processes', help='The processes to compare.', nargs='*')
    args = parser.parse_args()
    
    return args

def compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process='ZJetsToNuNu', year=2017, region='cr_baseline_vbf'):
    '''Compare smeared MET pt distribution between the two accumulators.'''
    # Load in the T1Smear and T1 MET (met and met_nom, respectively)
    acc_19Feb20.load('met')
    acc_19Feb20.load('met_nom')
    acc_05Jun20v5.load('met')
    acc_05Jun20v5.load('met_nom')

    h_19Feb20       = acc_19Feb20['met']
    h_19Feb20_nom   = acc_19Feb20['met_nom']
    h_05Jun20v5     = acc_05Jun20v5['met']
    h_05Jun20v5_nom = acc_05Jun20v5['met_nom']

    def preprocess(h,acc,T1MET=False):
        h = merge_extensions(h, acc, reweight_pu=False)
        scale_xs_lumi(h)
        h = merge_datasets(h)

        # Get the relevant dataset + region
        h = h.integrate('dataset', re.compile(f'{process}.*{year}')).integrate('region', region)

        # Rebin
        if region == 'cr_baseline_vbf':
            if T1MET:
                met_bin = hist.Bin('met_nom',r'$p_{T}^{miss}$ (GeV)',list(range(100,520,20)))
            else:
                met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(100,520,20)))
        elif 'cr_2m' in region:
            if T1MET:
                met_bin = hist.Bin('met_nom',r'$p_{T}^{miss}$ (GeV)',list(range(0,400,20)))
            else:
                met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,400,20)))
        else:
            if T1MET:
                met_bin = hist.Bin('met_nom',r'$p_{T}^{miss}$ (GeV)',list(range(250,550,20)))
            else:
                met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(250,550,20)))
            
        # Rebin MET (T1 or T1Smear)
        if T1MET:
            h = h.rebin('met_nom', met_bin)
        else:
            h = h.rebin('met', met_bin)

        return h

    # Preprocess histograms
    h_19Feb20   = preprocess(h_19Feb20, acc_19Feb20)
    h_05Jun20v5 = preprocess(h_05Jun20v5, acc_05Jun20v5)
    h_19Feb20_nom   = preprocess(h_19Feb20_nom, acc_19Feb20)
    h_05Jun20v5_nom = preprocess(h_05Jun20v5_nom, acc_05Jun20v5)

    # Plot the comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    # Plot T1Smear MET
    hist.plot1d(h_19Feb20, ax=ax)
    hist.plot1d(h_05Jun20v5, ax=ax, clear=False)
    
    # Plot T1 MET (just from 05Jun20v5 for now)
    hist.plot1d(h_05Jun20v5_nom, ax=ax, clear=False)

    ax.set_xlabel('')
    ax.set_yscale('log')
    if 'baseline' in region:
        ax.set_ylim(1e-1, 1e8)
    else:
        ax.set_ylim(1e-1, 1e4)

    # Set the title for the figure
    region_tags = {
        'cr_baseline' : 'Baseline Selections',
        'sr_vbf' : 'Signal region (VBF)',
        'cr_2m_vbf' : 'Dimuon control region (VBF)',
        'cr_2m_baseline' : r'Baseline Selections, $Z(\mu \mu)$'
    }
    
    region_tag = region_tags[region]
    ax.set_title(f'{process} {year}: {region_tag}')
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    ax.legend(labels=['Without JER fix', 'With JER fix', 'T1 MET'])

    # Plot ratio
    hist.plotratio(h_05Jun20v5, h_19Feb20, ax=rax, error_opts=data_err_opts, unc='num')
    rax.grid(True)
    rax.set_ylabel('With JER fix / Without')
    if 'cr_2m' in region:
        rax.set_ylim(0.6,1.4)
    else:
        rax.set_ylim(0.8,1.2)

    loc = matplotlib.ticker.MultipleLocator(base=0.1)
    rax.yaxis.set_major_locator(loc)

    xlim  = rax.get_xlim()
    rax.plot(xlim, [1., 1.], 'r--')
    rax.set_xlim(xlim)
    
    # Save figure
    if process == 'DYJetsToLL':
        outdir = f'./output/comparison_with_dy'
    elif process == 'ZJetsToNuNu':
        outdir = f'./output/comparison_with_znunu'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{process}_{year}_{region}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    args = parse_cli()
    inpath_19Feb20   = args.path_19Feb20
    inpath_05Jun20v5 = args.path_05Jun20v5
    if args.processes:
        processes = args.processes
    # By defualt, do the comparison for Znunu and DY processes
    else:
        processes = ['ZJetsToNuNu', 'DYJetsToLL']

    acc_19Feb20   = dir_archive(inpath_19Feb20, serialized=True, memsize=1e3, compression=0)
    acc_05Jun20v5 = dir_archive(inpath_05Jun20v5, serialized=True, memsize=1e3, compression=0)

    acc_19Feb20.load('sumw')
    acc_05Jun20v5.load('sumw')

    # Region: cr_baseline_vbf --> Region with minimal baseline selections
    # MET pt > 100 + leading jet pt/eta cuts 
    for process in processes:
        if process == 'ZJetsToNuNu':
            regions_to_test = ['cr_baseline', 'sr_vbf']
        elif process == 'DYJetsToLL':
            regions_to_test = ['cr_baseline', 'cr_2m_baseline']

        compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2017, region=regions_to_test[0])
        compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2018, region=regions_to_test[0])
        compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2017, region=regions_to_test[1])
        compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2018, region=regions_to_test[1])

if __name__ == '__main__':
    main()

