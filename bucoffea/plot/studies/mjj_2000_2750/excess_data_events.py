#!/usr/bin/env python

import re
import argparse
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt
from plot_mjj2000_2750 import REBIN

def excess_num_evts(acc, distribution, region):
    '''Given the accumulator, distribution and region,
       calculate the number of excess events in data,
       compared to MC.'''
    acc.load(distribution)
    h = acc[distribution]

    # Define data and MC samples for 2017
    data = 'EGamma_2017'
    mc = re.compile('(GJets_(HT|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*2017')

    # Merge extensions/datasets, scale w.r.t
    # x-section and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Choose the region (photon CR with mjj cut) and dataset
    h = h.integrate('region', region)

    # Rebin the histogram if necessary 
    if distribution in REBIN.keys():
        h = h.rebin(REBIN[distribution].name, REBIN[distribution])

    data_counts = h[data].integrate('dataset').values()[()]
    mc_counts = h[mc].integrate('dataset').values()[()]

    diff_counts = data_counts - mc_counts
    print('Data:')
    print(data_counts)
    print('MC:')
    print(mc_counts)
    print('Data/MC:')
    print(data_counts/mc_counts)
    print('Total data - Total MC = %.3f' % np.sum(diff_counts))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', help='Path containing the merged directory.')
    parser.add_argument('-r', '--region', help='The region to investigate.')
    parser.add_argument('-d', '--distribution', help='The distribution to investigate.')
    args = parser.parse_args()

    acc = dir_archive(
                        args.inpath,
                        serialized=True,
                        compression=0,
                        memsize=1e3
                        )

    acc.load('sumw')
    acc.load('sumw2')
    
    excess_num_evts(acc, 
                    distribution=args.distribution,
                    region=args.region
                    )

if __name__ == '__main__':
    main()
