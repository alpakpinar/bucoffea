#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from coffea import hist
from textwrap import dedent

pjoin = os.path.join

TAG_TO_TITLE = {
    'vbf_hinv' : 'VBF_HToInvisible',
    'qcd'      : 'QCD_HT'
}

def calculate_eff(dist):
    '''Given the distributions for each region,
       calculate the total number of predictions
       for each additional cut.'''
    # Store the predictions for each region
    predictions = {}
    for region, dist in dist.items():
        region_name = region[0]
        predictions[region_name] = np.sum(dist)

    # Calculate the efficiencies with
    # respect to SR with no additional cuts
    efficiencies = {}
    sr_prediction = predictions['sr_vbf']
    for region, prediction in predictions.items():
        efficiencies[region] = prediction*100/sr_prediction
    
    return efficiencies

def dpfcalo_comp(acc, out_tag, regex, tag):
    '''Given the input accumulator, plot mjj distribution
       with 4 cut cases:
       -- Regular VBF cuts
       -- Regular VBF cuts + dpfcalo cut
       -- Regular VBF cuts + chf/nhf cuts on leading jet
       -- Regular VBF cuts + chf/nhf cuts on leading jet pair

       Saves the output histogram as a .pdf on output/ directory.
       '''
    acc.load('mjj')
    h = acc['mjj']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    new_mjj_bin = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
    h = h.rebin('mjj', new_mjj_bin)

    region_regex = re.compile('sr_vbf.*')
    for year in [2017,2018]:
        h_ = h.integrate('dataset', re.compile(regex + f'_{year}'))
        
        # Get efficiency for each region
        dist = h_[region_regex].values()
        efficiencies = calculate_eff(dist)

        fig, ax = plt.subplots(1,1)
        hist.plot1d(h_[region_regex], ax=ax, overlay='region', binwnorm=True)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e6)
        title = TAG_TO_TITLE[tag] + f'_{year}'
        ax.set_title(title)
        ax.set_ylabel('Counts / Bin Width')

        eff_header = 'Efficiencies w.r.t SR:'

        eff_text = dedent(
        f'''
        DPFCalo cut: {efficiencies["sr_vbf_dpfcalo"]:.2f}%
        CHF/NHF cut (lead-j): {efficiencies["sr_vbf_leadjet_tight"]:.2f}%
        CHF/NHF cut (lead-jj): {efficiencies["sr_vbf_leadjetpair_tight"]:.2f}%
        ''')
        ax.text(0.05, 0.95, eff_header, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, weight='bold') 
        ax.text(0.05, 0.95, eff_text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes) 
        
        # Save the figure
        outdir = f'./output/{out_tag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f'dpfcalo_comparison_{year}.pdf')
        fig.savefig(outpath)

        print(f'Figure saved at {outpath}')

def plot_dpfcalo_dist(acc, out_tag, regex, tag):
    '''Given the input accumulator, plot dpfcalo
       distribution after regular VBF SR cuts.'''
    acc.load('dpfcalo')
    h = acc['dpfcalo']
    
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', 'sr_vbf')
    for year in [2017,2018]:
        h_ = h.integrate('dataset', re.compile(regex +f'_{year}'))

        # Plot the distribution
        fig, ax  = plt.subplots(1,1)
        hist.plot1d(h_, ax=ax, binwnorm=True)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e6)
        title = TAG_TO_TITLE[tag] + f'_{year}'
        ax.set_title(title)
        ax.set_ylabel('Counts / Bin Width')

        # Vertical line at dpfcalo=0.5
        ax.plot([0.5, 0.5], [1e-3, 1e6], 'r')

        # Save the figure
        outdir = f'./output/{out_tag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f'dpfcalo_dist_{year}.pdf')
        fig.savefig(outpath)
        
        print(f'Figure saved at {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(
                        inpath,
                        serialized=True,
                        compression=0,
                        memsize=1e3
                        )

    acc.load('sumw')
    acc.load('sumw2')
    
    # Get the output tag for outdir naming
    if inpath.endswith('/'):
        out_tag = inpath.split('/')[-2]
    else:
        out_tag = inpath.split('/')[-1]

    dpfcalo_comp(acc, out_tag=out_tag, regex='VBF_HToInv.*', tag='vbf_hinv')
    plot_dpfcalo_dist(acc, out_tag=out_tag, regex='VBF_HToInv.*', tag='vbf_hinv')

if __name__ == '__main__':
    main()
