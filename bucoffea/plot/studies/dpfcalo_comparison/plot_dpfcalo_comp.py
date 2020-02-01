#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from coffea import hist

def dpfcalo_comp(acc, regex, tag):
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
        print(h_[region_regex].values())
        fig, ax = plt.subplots(1,1)
        hist.plot1d(h_[region_regex], ax=ax, overlay='region', binwnorm=True)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e6)
        ax.set_title(f'QCD_HT_{year}')
        ax.set_ylabel('Normalized Counts')
        if not os.path.exists('./output'):
            os.mkdir('output')
        fig.savefig(f'./output/{tag}_dpfcalo_comparison_{year}.pdf')

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

    dpfcalo_comp(acc, regex='VBF_HToInv.*', tag='vbf_hinv')

if __name__ == '__main__':
    main()
