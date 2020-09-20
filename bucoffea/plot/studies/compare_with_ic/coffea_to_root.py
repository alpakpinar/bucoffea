#!/usr/bin/env python

import os
import sys
import re
import uproot
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

# Convert existing coffea files to root histograms and save them
def convert_to_root_for_sync(acc):
    outfilename = './inputs/sync/znunu_bu.root'
    outfile = uproot.recreate(outfilename)

    variables = ['ak4_eta0', 'ak4_eta1', 'mjj', 'recoil', 'ak4_pt0', 'ak4_pt1']
    for var in variables:
        print(f'Variable: {var}')
        acc.load(var)
        h = acc[var]

        h = merge_extensions(h, acc, reweight_pu=False)
        scale_xs_lumi(h)
        h = merge_datasets(h)

        h_znunu = h.integrate('dataset', f'ZJetsToNuNu_HT_2017').integrate('region', 'sr_vbf')
        th1 = hist.export1d(h_znunu)

        outfile[var] = th1

def convert_to_root(acc, jer=False):
    acc.load('mjj')
    h = acc['mjj']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    mjj_bins = hist.Bin('mjj', r'$M_{jj} \ (GeV)$',[200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000])
    h = h.rebin('mjj', mjj_bins)

    # Create output ROOT file
    if jer:
        outfilename = f'./inputs/bu_input_withJER.root'
    else:
        outfilename = f'./inputs/bu_input_noJER.root'
    outfile = uproot.recreate(outfilename)
    for year in [2017, 2018]:
        h_znunu = h.integrate('dataset', re.compile(f'ZJetsToNuNu.*{year}')).integrate('region', 'sr_vbf')
        h_wlnu = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}')).integrate('region', 'sr_vbf')

        h_zmumu = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}')).integrate('region', 'cr_2m_vbf')
        h_zee = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}')).integrate('region', 'cr_2e_vbf')

        h_wmunu = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}')).integrate('region', 'cr_1m_vbf')
        h_wenu = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}')).integrate('region', 'cr_1e_vbf')

        th1_znunu = hist.export1d(h_znunu)
        th1_wlnu = hist.export1d(h_wlnu)
        th1_zmumu = hist.export1d(h_zmumu)
        th1_zee = hist.export1d(h_zee)
        th1_wmunu = hist.export1d(h_wmunu)
        th1_wenu = hist.export1d(h_wenu)

        outfile[f'sr_qcd_znunu_{year}'] = th1_znunu
        outfile[f'sr_qcd_wlnu_{year}'] = th1_wlnu
        outfile[f'cr_qcd_zmumu_{year}'] = th1_zmumu
        outfile[f'cr_qcd_zee_{year}'] = th1_zee
        outfile[f'cr_qcd_wmunu_{year}'] = th1_wmunu
        outfile[f'cr_qcd_wenu_{year}'] = th1_wenu
    
        print(f'MSG% TH1F histogram saved in: {outfilename}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    # convert_to_root(acc, jer='withJER' in inpath)
    convert_to_root_for_sync(acc)

if __name__ == '__main__':
    main()
