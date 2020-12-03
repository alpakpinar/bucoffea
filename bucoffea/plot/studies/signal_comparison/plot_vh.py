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

def plot_vh_in_sr(acc, outtag, dataset, variable='mjj'):
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
        datasetregex = re.compile(dataset.format(year))
        _h = h.integrate('dataset', datasetregex)[ re.compile('inclusive|sr_vbf_no_veto_all') ]

        # Also get the VBF signal for reference
        vbfregex = re.compile(f'VBF_HToInvisible_M125_pow_pythia8_{year}')
        hsignal = h.integrate('dataset', vbfregex).integrate('region', 'sr_vbf_no_veto_all')

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='region')
        hist.plot1d(hsignal, ax=ax, clear=False)

        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e5)

        dataset_tag = dataset.split("_")[0]

        handles, labels = ax.get_legend_handles_labels()
        newlabels = {
            'inclusive' : f'{dataset_tag} Inclusive',
            'sr_vbf_no_veto_all' : f'{dataset_tag} Signal Region',
            'None' : 'VBF Signal Region'
        }

        for handle, label in zip(handles, labels):
            handle.set_label(
                newlabels[label]
            )

        ax.legend(title='Region', handles=handles)

        title = f'{dataset_tag}(inv) {year}'
        ax.set_title(title)

        # Calculate the ratio of VH integral / VBF integral to get a sense of the relative contribution
        integral_vbf = calculate_integral(hsignal)
        integral_vh = calculate_integral(_h.integrate('region', 'sr_vbf_no_veto_all'))

        integral_ratio = (integral_vh / integral_vbf) * 100

        ax.text(1., 1., f'VH/VBF: {integral_ratio:.3f}%',
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        outpath = pjoin(outdir, f'{dataset_tag}_comparison_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    datasets = [
        'WH_WToQQ_Hinv.*{}',
        'ZH_ZToQQ_HToInv.*{}'
    ]

    for dataset in datasets:
        plot_vh_in_sr(acc, outtag, dataset=dataset, variable='mjj')

if __name__ == '__main__':
    main()