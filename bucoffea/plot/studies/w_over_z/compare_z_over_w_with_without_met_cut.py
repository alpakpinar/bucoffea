#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
}

def preprocess(h, acc, distribution):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        h = h.rebin('mjj', REBIN['mjj'])

    return h

def compare_z_over_w_ratio(acc, outtag, distribution='mjj', proc='qcd', outputrootfile=None):
    '''Compare Z / W ratio for two cases.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        dataset_regex_for_ratio = {
            'qcd' : {
                'zjets' : re.compile(f'DYJetsToLL.*{year}'),
                'wjets' : re.compile(f'WJetsToLNu.*{year}'),
            },
            'ewk' : {
                'zjets' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}'),
                'wjets' : re.compile(f'EWKW2Jets.*{year}'),
            },
        }
        # Numerator and denominator histograms
        fig, ax = plt.subplots()

        h_num = h.integrate('region', 'cr_2e_vbf').integrate('dataset', dataset_regex_for_ratio[proc]['zjets'])
        # W region with MET > 80 cut
        h_den = h.integrate('region', 'cr_1e_vbf').integrate('dataset', dataset_regex_for_ratio[proc]['wjets'])
        # W region without MET > 80 cut
        h_den_nometcut = h.integrate('region', 'cr_1e_vbf_nometcut').integrate('dataset', dataset_regex_for_ratio[proc]['wjets'])

        # Plot the two ratios
        hist.plotratio(h_num, h_den, 
            ax=ax, 
            unc='num',
            error_opts=data_err_opts,
            label=r'W: $MET > 80 \ GeV$',
            )

        hist.plotratio(h_num, h_den_nometcut, 
            ax=ax, 
            unc='num',
            error_opts=data_err_opts,
            label=r'W: $MET$ inclusive',
            clear=False
            )

        ax.set_xlim(200,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel('Ratio')
        ax.legend()

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        ax.yaxis.set_ticks_position('both')

        ax.text(0., 1., r'{} $Z(ee) \ / \ W(e\nu)$'.format(proc.upper()),
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year,
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'{proc}_z_over_w_with_without_metcut_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

        # Save ratios to the output root file if one is specified while calling the function
        if outputrootfile is not None:
            ratio_normal = h_num.values()[()] / h_den.values()[()]
            ratio_without_met_cut = h_num.values()[()] / h_den_nometcut.values()[()]
            edges = h_num.axes()[0].edges()
            outputrootfile[f'{proc}_z_over_w_electrons_with_met_cut_{year}'] = (ratio_normal, edges) 
            outputrootfile[f'{proc}_z_over_w_electrons_without_met_cut_{year}'] = (ratio_without_met_cut, edges) 

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # Save ratios to an output root file
    outdir = f'./output/{outtag}/root'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outputrootpath = pjoin(outdir, f'z_over_w_with_without_met_cut.root')
    outputrootfile = uproot.recreate(outputrootpath)

    for proc in ['qcd', 'ewk']:
        compare_z_over_w_ratio(acc, outtag, proc=proc, outputrootfile=outputrootfile)

if __name__ == '__main__':
    main()
