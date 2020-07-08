#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
import matplotlib.ticker
from pprint import pprint

pjoin = os.path.join

def compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process='ZJetsToNuNu', year=2017, region='cr_baseline_vbf'):
    '''Compare smeared MET pt distribution between the two accumulators.'''
    acc_19Feb20.load('met')
    acc_05Jun20v5.load('met')

    h_19Feb20   = acc_19Feb20['met']
    h_05Jun20v5 = acc_05Jun20v5['met']

    def preprocess(h,acc):
        h = merge_extensions(h, acc, reweight_pu=False)
        scale_xs_lumi(h)
        h = merge_datasets(h)

        # Get the relevant dataset + region
        h = h.integrate('dataset', re.compile(f'{process}.*{year}')).integrate('region', region)

        # Rebin
        if region == 'cr_baseline_vbf':
            met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(100,520,20)))
        else:
            met_bin = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(250,550,20)))
            
        h = h.rebin('met', met_bin)

        return h

    # Preprocess histograms
    h_19Feb20   = preprocess(h_19Feb20, acc_19Feb20)
    h_05Jun20v5 = preprocess(h_05Jun20v5, acc_05Jun20v5)

    # Plot the comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h_19Feb20, ax=ax)
    hist.plot1d(h_05Jun20v5, ax=ax, clear=False)

    ax.set_xlabel('')
    # Set the title for the figure
    if region == 'cr_baseline_vbf':
        region_tag = 'Baseline Selections' 
    elif region == 'sr_vbf':
        region_tag = 'Signal region (VBF)' 
    elif region == 'cr_2m_vbf':
        region_tag = 'Dimuon control region (VBF)' 
    ax.set_title(f'{process} {year}: {region_tag}')
    
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

    ax.legend(labels=['Without JER fix', 'With JER fix'])

    # Plot ratio
    hist.plotratio(h_05Jun20v5, h_19Feb20, ax=rax, error_opts=data_err_opts, unc='num')
    rax.grid(True)
    rax.set_ylabel('With JER fix / Without')
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
    inpath_19Feb20, inpath_05Jun20v5 = sys.argv[1:]

    if 'dy' in inpath_19Feb20:
        process='DYJetsToLL'
        regions_to_test = ['cr_baseline_vbf', 'cr_2m_vbf']
    elif 'znunu' in inpath_19Feb20:
        process='ZJetsToNuNu'
        regions_to_test = ['cr_baseline_vbf', 'sr_vbf']

    acc_19Feb20   = dir_archive(inpath_19Feb20, serialized=True, memsize=1e3, compression=0)
    acc_05Jun20v5 = dir_archive(inpath_05Jun20v5, serialized=True, memsize=1e3, compression=0)

    acc_19Feb20.load('sumw')
    acc_05Jun20v5.load('sumw')

    # Region: cr_baseline_vbf --> Region with minimal baseline selections
    # MET pt > 100 + leading jet pt/eta cuts 
    compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2017, region=regions_to_test[0])
    compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2018, region=regions_to_test[0])
    compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2017, region=regions_to_test[1])
    compare_met_pt_jer(acc_19Feb20, acc_05Jun20v5, process=process, year=2018, region=regions_to_test[1])

if __name__ == '__main__':
    main()

