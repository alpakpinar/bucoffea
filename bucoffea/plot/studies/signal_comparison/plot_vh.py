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

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='region')

        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e5)

        handles, labels = ax.get_legend_handles_labels()
        newlabels = {
            'inclusive' : 'Inclusive',
            'sr_vbf_no_veto_all' : 'Signal Region'
        }

        for handle, label in zip(handles, labels):
            handle.set_label(
                newlabels[label]
            )

        ax.legend(title='Region', handles=handles)

        outpath = pjoin(outdir, f'{dataset.split("_")[0]}_comparison_{year}.pdf')
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