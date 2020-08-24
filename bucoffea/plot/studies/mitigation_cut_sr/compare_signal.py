#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from coffea import hist
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
}

def compare_signal(acc, variable='mjj'):
    '''Compare the signal distribution with two different noise mitigation cuts applied.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        h = h.rebin(variable, REBIN[variable])

    # Get the signal dataset
    h = h.integrate('dataset', re.compile('VBF_HToInv.*2017'))[re.compile('^sr_vbf((?!veto).)*$')]
    pprint(h.values())

    # Plot comparison
    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='region')
    fig.savefig('test.pdf')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    compare_signal(acc)

if __name__ == '__main__':
    main()