#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def calculate_integral(h):
    '''For a given histogram, calculate the integral of the distribution.'''
    edges = h.axes()[0].edges()
    binwidths = np.diff(edges)
    binvals = h.values()[()]

    integral = np.sum(binwidths * binvals)
    return integral

def plot_vh_in_sr(acc, outtag, variable='mjj'):
    '''Plot WH or ZH before and after the VBF signal region selection.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    
    # Rebin for mjj
    if variable == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)

    # Output directory to save figures
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        _h = h.integrate('region', 'sr_vbf_no_veto_all')[re.compile(f'(VBF_HToInvisible_M125_pow|WH|ZH|GluGlu).*{year}')]

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='dataset')

        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e7)

        handles, labels = ax.get_legend_handles_labels()
        newlabels = {
            'VBF.*M125_pow.*' : r'VBF $H(inv)$',
            'WH.*' : r'$WH(inv)$',
            'ZH.*' : r'$ZH(inv)$',
            'GluGlu.*' : r'$ggH(inv)$',
        }

        for handle, label in zip(handles, labels):
            newlabel = None
            for regex, _newlabel in newlabels.items():
                if re.match(regex, label):
                    newlabel = _newlabel
                    break
            if not newlabel:
                raise RuntimeError(f'Could not handle the legend label: {label}')
            
            handle.set_label(newlabel)

        ax.legend(title='Dataset', handles=handles)

        title = f'Signal Region: {year}'
        ax.set_title(title, fontsize=14)

        # Calculate the ratio of VH integral / VBF integral to get a sense of the relative contribution
        integral_vbf = calculate_integral(_h.integrate('dataset', re.compile(f'VBF.*{year}')))
        integral_wh = calculate_integral(_h.integrate('dataset', re.compile(f'WH.*{year}')))
        integral_zh = calculate_integral(_h.integrate('dataset', re.compile(f'ZH.*{year}')))

        integral_ratio_wh = (integral_wh / integral_vbf) * 100
        integral_ratio_zh = (integral_zh / integral_vbf) * 100

        ax.text(0.97, 0.05, f'WH/VBF: {integral_ratio_wh:.2f}%',
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )
        ax.text(0.97, 0.15, f'ZH/VBF: {integral_ratio_zh:.2f}%',
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'signal_region_comparison_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_vh_in_sr(acc, outtag, variable='mjj')

if __name__ == '__main__':
    main()