#!/usr/bin/env python

import os 
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt

def lhe_deltar_plot(acc, dataset):
    '''Given the accumulator and the dataset name, plot the 
       LHE deltaR distribution.
       ========================
       PARAMETERS:
       ========================
       acc: Accumulator containing the histograms.
       dataset: The dataset name.
    '''
    # Get the delta_r distribution 
    # from the accumulator
    acc.load('delta_r')
    dist = acc['delta_r']

    # Merge datasets/extensions and
    # scale the histogram w.r.t x-section/lumi    
    dist = merge_extensions(dist, acc, reweight_pu=False)
    scale_xs_lumi(dist)
    dist = merge_datasets(dist)
    
    # Choose the dataset/region
    dist = dist.integrate('region', 'cr_g_vbf').integrate('dataset', dataset)

    # Create output directory if it doesn't exists
    if not os.path.exists('./output'):
        os.mkdir('output')
    
    # Plot and save the histogram 
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    hist.plot1d(dist, ax=ax)
    ax.set_title(f'{dataset}')
    filepath = f'./output/lhe_deltar_{dataset}.pdf' 
    fig.savefig(filepath)
    print(f'Histogram saved in {filepath}')

    # Calculate the percentage of entries
    # with deltaR < 0.4
    smaller_0_4 = np.sum(dist.values()[()][:4])
    total = np.sum(dist.values()[()])
    percentage = (smaller_0_4/total)*100

    print('Percentage of entries with deltaR < 0.4 : %.3f%%' % percentage)


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

    lhe_deltar_plot(acc, dataset='GJets_HT_MLM_2017')

if __name__ == '__main__':
    main()

     
