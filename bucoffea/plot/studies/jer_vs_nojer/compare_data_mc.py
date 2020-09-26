#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from collections import OrderedDict
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

np.seterr(divide='ignore', invalid='ignore')

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

binnings = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
    'recoil' : hist.Bin('recoil','Recoil (GeV)', recoil_bins_2016),
    'recoil_relaxed' : hist.Bin('recoil','Recoil (GeV)', list(range(160,250,30)) + recoil_bins_2016),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) )
}

pretty_xlabels = {
    'mjj' : r'$M_{jj} \ (GeV)$',
    'recoil' : 'Recoil (GeV)',
    'ak4_pt0' : r'Leading jet $p_T \ (GeV)$',
    'ak4_pt1' : r'Trailing jet $p_T \ (GeV)$',
    'ak4_eta0' : r'Leading jet $\eta$',
    'ak4_eta1' : r'Trailing jet $\eta$',
}

pretty_titles_for_region = {
    'sr_vbf' : 'VBF Signal Region',
    'cr_1m_vbf' : r'VBF $1\mu$ CR Region',
    'cr_1e_vbf' : r'VBF $1e$ CR Region',
    'cr_2m_vbf' : r'VBF $2\mu$ CR Region',
    'cr_2e_vbf' : r'VBF $2e$ CR Region',
    'cr_g_vbf' : r'VBF $\gamma$ CR Region'
}

def get_label_for_tag(tag):
    mapping = {
        'noSmear' : 'No JER',
        '09Jun20v7' : 'MiniAOD-like JER',
        '21Sep20v7' : 'NanoAOD-like JER'
    }
    return mapping[tag]

def do_rebinning(h, variable):
    if variable == 'mjj':
        h = h.rebin('mjj', binnings['mjj'])
    elif variable == 'recoil':
        h = h.rebin('recoil', binnings['recoil'])
    elif 'ak4_pt' in variable:
        h = h.rebin('jetpt', binnings[variable])

    return h

def preprocess(h, acc, variable, year, region):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    
    # Regexes to match for data and MC
    data = {
        'sr_vbf' : f'MET_{year}',
        'cr_1m_vbf' : f'MET_{year}',
        'cr_2m_vbf' : f'MET_{year}',
        'cr_1e_vbf' : f'EGamma_{year}',
        'cr_2e_vbf' : f'EGamma_{year}',
        'cr_g_vbf' : f'EGamma_{year}',
    }

    mc = {
        'sr_vbf' : re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1m_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EWKZ.*ZToLL.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_g_vbf' : re.compile(f'(GJets_(DR-0p4|SM).*|QCD_data.*|WJetsToLNu.*HT.*).*{year}'),
    }

    h_data = h.integrate('region', region).integrate('dataset', re.compile(data[region]))
    h_mc = h.integrate('region', region).integrate('dataset', re.compile(mc[region]))

    if variable in binnings.keys():
        h_data = do_rebinning(h_data, variable)
        h_mc = do_rebinning(h_mc, variable)

    return {'data' : h_data, 'mc' : h_mc}

def compare_data_mc(acc_dict, variable='ak4_eta0', year=2017, region='sr_vbf'):
    '''
    For the given region+year, compare data/MC behavior for three different smearing types:
    1. No smearing applied
    2. MiniAOD-like smearing applied
    3. NanoAOD-like smearing applied
    '''
    h_dict = OrderedDict()
    for tag, acc in acc_dict.items():
        acc.load(variable)
        # Get the pre-processed histograms
        h_dict[tag] = preprocess(acc[variable], acc, variable, year, region)

    # Calculate data/MC for three cases
    legend_labels = []
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    for tag, hs in h_dict.items():
        h_data = hs['data']
        h_mc = hs['mc']

        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.
        }

        hist.plotratio(h_data, h_mc, ax=ax, label=get_label_for_tag(tag), clear=False, unc='num', error_opts=data_err_opts)
        legend_labels.append(get_label_for_tag(tag))

        # Calculate % difference from 1.0
        data_mc_ratio = h_data.values()[()] / h_mc.values()[()]
        percent_diff = (data_mc_ratio - 1.0) / 1.0 
        centers = h_data.axes()[0].centers()
        rax.plot(centers, percent_diff, marker='o', ls='', label=get_label_for_tag(tag))

    ax.legend(labels=legend_labels)
    ax.grid(True)
    ax.set_ylabel('Data / MC')
    ax.set_xlabel('')
    ax.set_ylim(0,2)
    ax.set_title(pretty_titles_for_region[region])

    xlim = ax.get_xlim()
    ax.plot(xlim, [1,1], color='black')
    ax.set_xlim(xlim)

    rax.grid(True)
    rax.set_ylim(-1,1)
    rax.set_ylabel(r'% diff from 1.0')
    rax.set_xlabel(pretty_xlabels[variable])
    rax.legend(ncol=3)

    # Save figure
    outdir = f'./output/data_mc_comparisons/{region}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}_{year}.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')
    plt.close(fig)

def main():
    '''Compare Z(vv) distributions for three cases:
    1. No JER smearing applied
    2. MiniAOD-like JER smearing applied (09Jun20v7 skim)
    3. NanoAOD-like JER smearing applied (21Sep20v7 skim)
    '''
    inpath_noSmear = bucoffea_path('./submission/merged_2020-09-17_vbfhinv_noJER_nanoAODv7_deepTau')
    inpath_09Jun20v7 = bucoffea_path('./submission/merged_2020-09-17_vbfhinv_withJER_nanoAODv7_deepTau')
    inpath_21Sep20v7 = bucoffea_path('./submission/merged_2020-09-25_vbfhinv_21Sep20v7')

    acc_dict = {
        'noSmear' : dir_archive(inpath_noSmear),
        '09Jun20v7' : dir_archive(inpath_09Jun20v7),
        '21Sep20v7' : dir_archive(inpath_21Sep20v7)
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    variables = ['ak4_eta0', 'ak4_eta1']
    regions = pretty_titles_for_region.keys()

    for region in regions:
        for variable in variables:
            compare_data_mc(acc_dict, variable=variable, year=2017, region=region)

if __name__ == '__main__':
    main()

