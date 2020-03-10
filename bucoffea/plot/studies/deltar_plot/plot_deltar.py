#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from pprint import pprint
from klepto.archives import dir_archive

pjoin = os.path.join

def plot_deltar_dist(acc, regex, tag, outtag):
    '''
    Given the input accumulator, plot the LHE-level deltaR distribution between
    photons and partons.
    ==================
    PARAMETERS:
    ==================
    acc    : Input accumulator containing all the histograms.
    regex  : The regular expression matching the dataset name.
    tag    : Tag for the process.
    outtag : Out tag for naming the output directory.
    '''
    pt_type = 'stat1'
    dist = f'lhe_mindr_g_parton_{pt_type}'
    acc.load(dist)
    h = acc[dist]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Pick the relevant dataset(s)
    h = h[re.compile(regex)].integrate('dataset')

    # Plot the histogram
    fig, ax = plt.subplots(1,1)
    hist.plot1d(h, ax=ax)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{tag}_deltar.pdf')
    fig.savefig(outpath)
    print(f'Figure saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
        inpath,
        memsize=1e3
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    tag_regex = {
        'gjets_dr_2016'  : 'GJets_DR-0p4.*2016',
        'gjets_dr_2017'  : 'GJets_DR-0p4.*2017',
        'gjets_ht_2017'  : 'GJets_HT.*2017',
        'gjets_nlo_2016' : 'G1Jet.*2016'
    }

    for tag, regex in tag_regex.items():
        plot_deltar_dist(acc,regex=regex,tag=tag,outtag=outtag)

if __name__ == '__main__':
    main()