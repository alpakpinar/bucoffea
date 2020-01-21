#!/usr/bin/env python

import os
import re
import argparse
import numpy as np
from tabulate import tabulate
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt
from plot_mjj2000_2750 import REBIN, get_tag

pjoin = os.path.join

# Set print options for numpy
print_options = {
    'precision' : 3,
    'suppress' : True
}
np.set_printoptions(**print_options)

def excess_num_evts(acc, distribution, region, tag, mjjcut, binning='coarse'):
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
        h = h.rebin(REBIN[distribution][binning].name, REBIN[distribution][binning])

    data = h[data].integrate('dataset')
    mc = h[mc].integrate('dataset')

    bin_centers = data.axes()[0].centers()

    data_counts = data.values()[()]
    mc_counts = mc.values()[()]

    ratio = data_counts/mc_counts
    diff = data_counts - mc_counts

    stack = np.column_stack(
        (bin_centers,
        data_counts,
        mc_counts,
        diff,
        ratio
        )
    )

    headers = [f'{distribution.capitalize()}', 'Data', 'MC', 'Data-MC', 'Data/MC']

    # Tabulate the values
    table = tabulate(stack, headers, tablefmt='fancy_grid', numalign='right', floatfmt=('.1f','.1f','.3f','.3f','.3f'))

    # Dump the table into a .txt file
    outdir = f'./output/{tag}/data_mc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if mjjcut:
        filename = f'data_mc_{distribution}_mjj2000_2750.txt'
    else:
        filename = f'data_mc_{distribution}.txt'
    outpath = pjoin(outdir, filename) 
    with open(f'{outpath}', 'w+') as f:
        f.write(table)

    print(f'Output saved in {outpath}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', help='Path containing the merged directory.')
    parser.add_argument('-r', '--region', help='The region to investigate.')
    parser.add_argument('-d', '--distribution', help='The distribution to investigate.')
    args = parser.parse_args()

    tag = get_tag(args.inpath)

    mjjcut = 'mjj2000_2750' in tag

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
                    region=args.region,
                    tag=tag,
                    mjjcut=mjjcut
                    )

if __name__ == '__main__':
    main()
