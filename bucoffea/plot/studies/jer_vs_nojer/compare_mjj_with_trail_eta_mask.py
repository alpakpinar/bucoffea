#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np
import warnings

from matplotlib import pyplot as plt
from coffea import hist
from bucoffea.plot.util import merge_extensions, merge_datasets, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from klepto.archives import dir_archive

pjoin = os.path.join

warnings.filterwarnings('ignore')

mjj_binning = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(0,3000,100)))

def preprocess(h, acc):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.rebin('mjj', mjj_binning)

    return h

def compare_mjj_spectrum(acc_withSmear, acc_noSmear, year=2017, trailjet_filter=False):
    '''Compare mjj spectra with and without smearing.'''
    acc_withSmear.load('mjj')
    h_ws = acc_withSmear['mjj']
    acc_noSmear.load('mjj')
    h_ns = acc_noSmear['mjj']

    h_ws = preprocess(h_ws, acc_withSmear)
    h_ns = preprocess(h_ns, acc_noSmear)

    sr_data_regex = re.compile(f'MET.*{year}')
    sr_mc_regex = re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}')

    if trailjet_filter:
        region = 'sr_vbf_trailJetMask'
    else:
        region = 'sr_vbf'

    h_ws_data = h_ws.integrate('dataset', sr_data_regex).integrate('region', region)
    h_ns_data = h_ns.integrate('dataset', sr_data_regex).integrate('region', region)
    h_ws_mc = h_ws.integrate('dataset', sr_mc_regex).integrate('region', region)
    h_ns_mc = h_ns.integrate('dataset', sr_mc_regex).integrate('region', region)

    centers = h_ws_data.axes()[0].centers()

    # Get data / MC values
    r_ws = h_ws_data.values()[()] / h_ws_mc.values()[()]
    r_ns = h_ns_data.values()[()] / h_ns_mc.values()[()]

    # Plot the comparison
    fig, ax = plt.subplots()
    ax.plot(centers, r_ws, marker='o', label='With JER')
    ax.plot(centers, r_ns, marker='o', label='No JER')

    ax.legend()
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_ylabel('Data / MC')
    ax.set_ylim(0,2)
    ax.grid(True)

    if trailjet_filter:
        title = r'With trailing jet veto: $2.4 < |\eta| < 2.8$'
    else:
        title = r'Without trailing jet veto: $2.4 < |\eta| < 2.8$'
    ax.set_title(title)

    # Save figure
    outdir = './output/trail_jet_eta_mask'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if trailjet_filter:
        filename = f'mjj_comp_with_trailjet_filter_{year}.pdf'
    else:
        filename = f'mjj_comp_without_trailjet_filter_{year}.pdf'

    outpath = pjoin(outdir, filename)
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath_withSmear = bucoffea_path('./submission/merged_2020-09-18_vbfhinv_withJER_nanoAODv7_deepTau')
    inpath_noSmear = bucoffea_path('./submission/merged_2020-09-18_vbfhinv_nanoAODv7_tree')

    acc_withSmear = dir_archive(inpath_withSmear)
    acc_noSmear = dir_archive(inpath_noSmear)

    acc_withSmear.load('sumw')
    acc_withSmear.load('sumw2')
    acc_noSmear.load('sumw')
    acc_noSmear.load('sumw2')

    compare_mjj_spectrum(acc_withSmear, acc_noSmear, trailjet_filter=True)
    compare_mjj_spectrum(acc_withSmear, acc_noSmear, trailjet_filter=False)

if __name__ == '__main__':
    main()