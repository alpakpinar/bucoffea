#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt

def plot_recoil(acc, dataset):
    '''Plot the recoil histogram for events with
       2000 < mjj < 2750 GeV, from the given dataset.'''
    acc.load('recoil_mjj2000_2750')
    dist = acc['recoil_mjj2000_2750']

    # Merge extensions/datasets, scale w.r.t
    # x-section and lumi
    dist = merge_extensions(dist, acc, reweight_pu=False)
    scale_xs_lumi(dist)
    dist = merge_datasets(dist)

    # Choose the region (photon CR) and dataset
    dist = dist.integrate('region', 'cr_g_vbf').integrate('dataset', dataset)

    fig, ax = plt.subplots(1,1,figsize=(7,5))
    hist.plot1d(dist, ax=ax, binwnorm=True)
    filepath = f'./output/recoil_mjj2000_2750_{dataset}.pdf'
    fig.savefig(filepath)
    print(f'Histogram saved in {filepath}')        

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
                      inpath,
                      serialized=True,
                      compression=0,
                      memsize=1e3
                    )

    plot_recoil(acc, dataset='EGamma_2017')

if __name__ == '__main__':
    main()
    
    
