#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from matplotlib import pyplot as plt

pjoin = os.path.join

def plot_leading_jet_pt_dist(acc, outtag, year=2017, region='norecoil'):
    '''Plot leading jet pt distribution to check if the cuts are properly applied.'''
    acc.load('ak4_pt0')
    h = acc['ak4_pt0']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('dataset', re.compile(f'DYJets.*{year}')).integrate('region', f'cr_2e_j_{region}')

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)

    outdir = f'./output/{outtag}/jet_pt_test'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'jet_pt_{region}_{year}.pdf')
    fig.savefig(outpath)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        for region in ['norecoil', 'norecoil_nojpt', 'norecoil_jptv2']:
            plot_leading_jet_pt_dist(acc, outtag, year, region)

if __name__ == '__main__':
    main()