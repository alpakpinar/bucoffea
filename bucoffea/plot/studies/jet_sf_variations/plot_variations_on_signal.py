#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

np.seterr(divide='ignore')

legend_labels = {
    'sr_vbf_no_veto_all' : 'Nominal',
    'sr_vbf_ak4sfUp_no_veto_all' : 'Jet SF Up',
    'sr_vbf_ak4sfDown_no_veto_all' : 'Jet SF Down'
}

xlabels = {
    'mjj' : r'$M_{jj} \ (GeV)$',
    'ak4_eta0' : r'Leading jet $\eta$',
    'ak4_eta1' : r'Trailiing jet $\eta$',
}

def plot_variations(acc, outtag, variable='mjj'):
    '''Plot jet SF variations on VBF signal.'''
    acc.load(variable)
    h = acc[variable]
    
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if variable == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)

    # Pick the signal dataset
    for year in [2017]:
        h = h.integrate('dataset', re.compile(f'VBF.*{year}'))[re.compile('sr_.*_no_veto_all')]

        fig, ax, rax = fig_ratio()
        hist.plot1d(h, ax=ax, overlay='region')

        ax.set_xlabel('')
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            newlabel = legend_labels[label]
            handle.set_label(newlabel)

        ax.legend(title='Jet SF Variations', handles=handles)

        ax.set_title(f'VBF MC: {year}')

        # Plot ratios
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        hnom = h.integrate('region', 'sr_vbf_no_veto_all')
        centers = hnom.axes()[0].centers()

        hnom = hnom.values()[()]
        hup = h.integrate('region', re.compile(f'sr_.*ak4sfUp.*')).values()[()]
        hdown = h.integrate('region', re.compile(f'sr_.*ak4sfDown.*')).values()[()]

        rax.plot(centers, hdown / hnom, **data_err_opts)
        rax.plot(centers, hup / hnom, **data_err_opts)

        rax.set_ylabel('Ratio to Nominal')
        rax.set_ylim(0.97,1.03)
        rax.axhline(1, xmin=0, xmax=1, color='k')
        rax.grid(True)

        loc = MultipleLocator(0.01)
        rax.yaxis.set_major_locator(loc)

        rax.set_xlabel(xlabels[variable])

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f'jet_sf_uncs_vbf_{variable}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for variable in ['mjj', 'ak4_eta0', 'ak4_eta1']:
        plot_variations(acc, outtag, variable=variable)

if __name__ == '__main__':
    main()