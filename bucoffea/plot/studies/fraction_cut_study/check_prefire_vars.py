#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import warnings
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint 

pjoin = os.path.join

warnings.filterwarnings('ignore')

def preprocess(h, acc, variable, year):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h_data = h.integrate('dataset', f'SingleMuon_{year}')
    h_mc = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))

    return h_data, h_mc

def compare_with_no_prefire(acc, outtag, variable='ak4_eta0', year=2017):
    '''Compare the efficiency of EM fraction cut in data/MC with prefire weight being applied or not.'''
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, variable, year)

    h_dataNoCut = h_data.integrate('region', 'cr_2m_noEmEF')
    h_dataWithCut = h_data.integrate('region', 'cr_2m_withEmEF')

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plotratio(h_dataWithCut, h_dataNoCut, unc='clopper-pearson', label='Data', ax=ax, error_opts=data_err_opts)
    eff_data = h_dataWithCut.values()[()] / h_dataNoCut.values()[()]

    xlim = ax.get_xlim()
    ax.plot(xlim, [1., 1.], color='black')
    ax.set_xlim(xlim)

    # Calculate efficiencies in MC for two cases:
    # 1. With regular (central) prefire weight applied
    # 2. With regular (central) prefire weight not applied

    h_mcNoCut = {
        'MC Prefire Applied' : h_mc.integrate('region', 'cr_2m_noEmEF'),
        'MC Prefire Not Applied' : h_mc.integrate('region', 'cr_2m_noEmEF_no_prefire')
    }

    h_mcWithCut = {
        'MC Prefire Applied' : h_mc.integrate('region', 'cr_2m_withEmEF'),
        'MC Prefire Not Applied' : h_mc.integrate('region', 'cr_2m_withEmEF_no_prefire')
    }

    eff_mc = {}
    for key in h_mcNoCut.keys():
        hist.plotratio(h_mcWithCut[key], h_mcNoCut[key], unc='clopper-pearson', clear=False, label=key, ax=ax, error_opts=data_err_opts)
        eff_mc[key] = h_mcWithCut[key].values()[()] / h_mcNoCut[key].values()[()]

    ax.legend()
    ax.set_ylim(0.7,1.05)
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('')

    # Plot ratios to efficiencies in data
    centers = h_dataNoCut.axes()[0].centers()
    for key in eff_mc.keys():
        r = eff_data / eff_mc[key]
        rax.plot(centers, r, label=key.split(' ')[-1], ls='', marker='o')
    
    rax.grid(True)
    rax.set_ylim(0.8,1.2)
    rax.set_ylabel('Data / MC eff.')
    rax.set_xlabel(r'Jet $\eta$')
    rax.legend(ncol=2)

    # Save figure
    outdir = f'./output/{outtag}/prefire_vars'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{year}_pref_no_pref_comp.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def compare_prefire_vars(acc, outtag, variable='ak4_eta0', year=2017):
    '''Compare the efficiency of EM fraction cut in data/MC with prefire variations.'''
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, variable, year)

    h_dataNoCut = h_data.integrate('region', 'cr_2m_noEmEF')
    h_dataWithCut = h_data.integrate('region', 'cr_2m_withEmEF')

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.
    }

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plotratio(h_dataWithCut, h_dataNoCut, unc='clopper-pearson', label='Data', ax=ax, error_opts=data_err_opts)
    eff_data = h_dataWithCut.values()[()] / h_dataNoCut.values()[()]

    xlim = ax.get_xlim()
    ax.plot(xlim, [1., 1.], color='black')
    ax.set_xlim(xlim)

    # Calculate efficiencies for three cases: 
    # 1. Regular prefire weight applied to MC
    # 2. Prefire up weight applied to MC
    # 3. Prefire down weight applied to MC

    h_mcNoCut = {
        'MC Prefire Central' : h_mc.integrate('region', 'cr_2m_noEmEF'),
        'MC Prefire Up' : h_mc.integrate('region', 'cr_2m_noEmEF_prefireUp'),
        'MC Prefire Down' : h_mc.integrate('region', 'cr_2m_noEmEF_prefireDown')
    }

    h_mcWithCut = {
        'MC Prefire Central' : h_mc.integrate('region', 'cr_2m_withEmEF'),
        'MC Prefire Up' : h_mc.integrate('region', 'cr_2m_withEmEF_prefireUp'),
        'MC Prefire Down' : h_mc.integrate('region', 'cr_2m_withEmEF_prefireDown')
    }

    eff_mc = {}
    for key in h_mcNoCut.keys():
        hist.plotratio(h_mcWithCut[key], h_mcNoCut[key], unc='clopper-pearson', clear=False, label=key, ax=ax, error_opts=data_err_opts)
        eff_mc[key] = h_mcWithCut[key].values()[()] / h_mcNoCut[key].values()[()]

    ax.legend()
    ax.set_ylim(0.7,1.05)
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('')

    # Plot ratios to efficiencies in data
    centers = h_dataNoCut.axes()[0].centers()
    for key in eff_mc.keys():
        r = eff_data / eff_mc[key]
        rax.plot(centers, r, label=key.split(' ')[-1], ls='', marker='o')
    
    rax.grid(True)
    rax.set_ylim(0.8,1.2)
    rax.set_ylabel('Data / MC eff.')
    rax.set_xlabel(r'Jet $\eta$')
    rax.legend(ncol=3)

    # Save figure
    outdir = f'./output/{outtag}/prefire_vars'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{year}_prefire_comp.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    for year in [2017, 2018]:
        compare_prefire_vars(acc, outtag, year=year)
        compare_with_no_prefire(acc, outtag, year=year)

if __name__ == '__main__':
    main()