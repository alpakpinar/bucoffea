#!/usr/bin/env python

import os
import sys
import re
import warnings
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

warnings.filterwarnings('ignore')

def preprocess(h, acc, region):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)
    return h

def get_regex(region, year):
    data = {
        'sr_vbf' : f'MET_{year}',
        'cr_1m_vbf' : f'MET_{year}',
        'cr_2m_vbf' : f'MET_{year}',
        'cr_1e_vbf' : f'EGamma_{year}',
        'cr_2e_vbf' : f'EGamma_{year}',
        'cr_g_vbf' : f'EGamma_{year}',
    }
    # MC samples for each region
    mc = {
        'sr_vbf' : re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_g_vbf' : re.compile(f'(GJets_(DR-0p4|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*{year}'),
    }

    return {'data' : data[region], 'mc' : mc[region]}

def plot_eta_comparison(acc_dict, outtag, region='sr_vbf', year=2017):
    '''Plot data/MC for given region, as a function of trailing jet eta, with and without smearing.'''
    h_dict = {}
    for tag, acc in acc_dict.items():
        acc.load('ak4_eta1')
        h_dict[tag] = preprocess(acc['ak4_eta1'], acc, region)

    # Plot data/MC for with and without JER
    regex = get_regex(region, year)
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    ratios = {}

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
    }

    for tag, h in h_dict.items():
        h_data = h.integrate('dataset', re.compile(regex['data']))
        h_mc = h.integrate('dataset', re.compile(regex['mc']))

        hist.plotratio(h_data, h_mc, unc='num', ax=ax, label=tag, error_opts=data_err_opts, clear=False)

        ratios[tag] = h_data.values()[()] / h_mc.values()[()]

    ax.set_xlabel(r'Trailing jet $\eta$')
    ax.set_ylabel('Data / MC')
    ax.set_ylim(0,2)
    ax.grid(True)
    ax.legend()

    # Plot ratio of the data/MC ratios
    dratio = ratios['With JER'] / ratios['No JER']
    centers = h.integrate('dataset').axes()[0].centers()
    rax.plot(centers, dratio, marker='o', ls='', color='k')

    rax.set_ylim(0.5,1.5)
    rax.set_ylabel('With JER / No JER')
    rax.grid(True)

    ylim = rax.get_ylim()
    rax.plot([-2.8, -2.8], ylim, color='red')
    rax.plot([-2.4, -2.4], ylim, color='red')
    rax.plot([2.8, 2.8], ylim, color='red')
    rax.plot([2.4, 2.4], ylim, color='red')
    rax.set_ylim(ylim)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{region}_eta_comp.pdf')
    fig.savefig(outpath)
    print(f'MSG% File saved: {outpath}')

def main():
    # Path to the merged coffea files with and without smearing applied
    inpath_ws = bucoffea_path('./submission/merged_2020-09-17_vbfhinv_withJER_nanoAODv7_deepTau')
    inpath_ns = bucoffea_path('./submission/merged_2020-09-17_vbfhinv_noJER_nanoAODv7_deepTau')
    outtag = '17Sep20'

    acc_dict = {
        'With JER' : dir_archive(inpath_ws),
        'No JER' : dir_archive(inpath_ns)
    }
    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    for region in ['sr_vbf', 'cr_2m_vbf', 'cr_1m_vbf', 'cr_g_vbf']:
        plot_eta_comparison(acc_dict, outtag, region=region)

if __name__ == '__main__':
    main()
