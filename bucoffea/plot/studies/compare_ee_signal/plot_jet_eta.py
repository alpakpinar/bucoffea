#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from matplotlib import pyplot as plt
from coffea import hist
from klepto.archives import dir_archive

pjoin = os.path.join

def plot_jet_eta(acc, outtag, year=2017, jet='leading'):
    '''Plot jet eta distribution for jets with neutral EM fraction > 0.7'''
    if jet == 'leading':
        acc.load('ak4_eta0')
        h = acc['ak4_eta0']
    elif jet == 'trailing':
        acc.load('ak4_eta1')
        h = acc['ak4_eta1']
    
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if jet == 'leading':
        h = h.integrate('region', 'sr_vbf_ak40_largeEmEF')
    elif jet == 'trailing':
        h = h.integrate('region', 'sr_vbf_ak41_largeEmEF')

    h = h.integrate('dataset', re.compile(f'VBF.*{year}'))

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)
    if jet == 'leading':
        ax.set_title(f'VBF {year}: Leading jet with EM frac > 0.7')
        ax.set_xlabel(r'Leading jet $\eta$')
    elif jet == 'trailing':
        ax.set_title(f'VBF {year}: Traling jet with EM frac > 0.7')
        ax.set_xlabel(r'Trailing jet $\eta$')
    
    # Remove unused legend
    ax.get_legend().remove()
    
    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{jet}_jet_eta_check_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        plot_jet_eta(acc, outtag, year=year, jet='leading')
        plot_jet_eta(acc, outtag, year=year, jet='trailing')

if __name__ == '__main__':
    main()