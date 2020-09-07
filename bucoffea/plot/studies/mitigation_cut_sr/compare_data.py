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
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) ),
}

XLABELS = {
    'mjj' : r'$M_{jj} \ (GeV)$',
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
    'ak4_pt0' : r'Leading Jet $p_T$',
    'ak4_pt1' : r'Trailing Jet $p_T$'
}

def compare_data(acc, outtag, variable='ak4_eta0'):
    '''Compare the MET dataset in SR between two versions of mitigation cuts.'''
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

    # Get the MET dataset
    h = h.integrate('dataset', 'MET_2017')[re.compile('^sr_vbf((?!(veto_all|leadak4)).)*$')]

    # Plot comparison
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region', overflow='over')
    
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1,1e5)
    ax.set_title('MET 2017')

    if 'ak4_eta' in variable:
        ylim = ax.get_ylim()
        ax.plot([-3.4, -3.4], ylim, '--', color='black')
        ax.plot([-2.8, -2.8], ylim, '--', color='black')
        ax.plot([3.4, 3.4], ylim, '--', color='black')
        ax.plot([2.8, 2.8], ylim, '--', color='black')

    # Calculate and plot ratio
    h_sr_vbf = h.integrate('region', 'sr_vbf')
    h_eemitigation_v1 = h.integrate('region', 'sr_vbf_eemitigationv1')
    h_eemitigation_v2 = h.integrate('region', 'sr_vbf_eemitigationv2')
    h_eemitigation_v3 = h.integrate('region', 'sr_vbf_eemitigationv3')

    centers = h_sr_vbf.axes()[0].centers(overflow='over')
    r_eemitigation_v1 = h_eemitigation_v1.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    r_eemitigation_v2 = h_eemitigation_v2.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    r_eemitigation_v3 = h_eemitigation_v3.values(overflow='over')[()] / h_sr_vbf.values(overflow='over')[()]
    rax.plot(centers, r_eemitigation_v1, ls='', marker='o', label='EEv1')
    rax.plot(centers, r_eemitigation_v2, ls='', marker='o', label='EEv2')
    rax.plot(centers, r_eemitigation_v3, ls='', marker='o', label='EEv3')
    
    rax.grid(True)
    rax.set_ylim(0.8,1.2)
    rax.set_ylabel('Ratio to nominal SR')
    rax.set_xlabel(XLABELS[variable])
    rax.legend(ncol=3)

    if 'ak4_eta' in variable:
        ylim = rax.get_ylim()
        rax.plot([-2.8, -2.8], ylim, '--', color='black')
        rax.plot([-3.4, -3.4], ylim, '--', color='black')
        rax.plot([3.4, 3.4], ylim, '--', color='black')
        rax.plot([2.8, 2.8], ylim, '--', color='black')

    loc = matplotlib.ticker.MultipleLocator(base=0.05)
    rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'data_comp_{variable}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['ak4_eta0', 'ak4_eta1', 'ak4_pt0', 'ak4_pt1'] 
    for variable in variables:
        compare_data(acc, outtag, variable=variable)

if __name__ == '__main__':
    main()
