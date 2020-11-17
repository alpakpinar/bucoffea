#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

legend_labels = {
    'DY.*' : "Z$\\rightarrow\\ell\\ell$",
    'Top.*' : "Top quark",
    'WJets.*LNu.*' : "W$\\rightarrow\\ell\\nu$",
    'QCD.*' : "QCD",
    'Diboson.*' : "WW/WZ/ZZ",
    'ZJetsToNuNu.*' : "Z$\\rightarrow\\nu\\nu$",
    'EWKZ.*ZToLL.*' : r"EWK $Z\rightarrow \ell\ell$",
    'EWKZ.*ZToNuNu.*' : r"EWK $Z\rightarrow \nu\nu$",
    'EWKW.*' : r"EWK $W\rightarrow \ell\nu$",
    'VBF_HToInv.*M125.*' : "VBF H(inv)",
    'MET|Single(Electron|Photon|Muon)|EGamma.*' : "Data"

}

def get_title(distribution, year):
    if distribution == 'ak4_pt0_eta0':
        title = r'Leading Jet, $2.5 < |\eta| < 3.0$ {}'
    elif distribution == 'ak4_pt1_eta1':
        title = r'Trailing Jet, $2.5 < |\eta| < 3.0$ {}'

    return title.format(year)

def plot_jet_pt(acc, outtag, distribution):
    '''Plot the jet pt distribution for leading or trailing jet (in endcap).'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    endcap_slice = slice(2.5, 3.0)
    h = h.integrate('jeteta', endcap_slice).integrate('region', 'sr_vbf')

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    signal_line_opts = {
        'color' : 'crimson'
    }

    for year in [2017, 2018]:
        h_data = h[f'MET_{year}']
        h_mc = h[re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}')]
        h_signal = h[re.compile(f'VBF_HToInv.*M125.*{year}')]

        fig, ax = plt.subplots()
        hist.plot1d(h_data, ax=ax, overlay='dataset', error_opts=data_err_opts)
        hist.plot1d(h_mc, ax=ax, overlay='dataset', stack=True, clear=False)
        hist.plot1d(h_signal, ax=ax, overlay='dataset', clear=False, line_opts=signal_line_opts)

        ax.set_title(get_title(distribution, year))

        # Fix legend labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for handle, label in zip(handles, labels):
            if not ('MET' in label or 'VBF' in label):
                handle.set_linestyle('-')
                handle.set_edgecolor('k')
            
            for k, v in legend_labels.items():
                if re.match(k, label):
                    new_label = v
            
            new_labels.append(new_label)

        ax.legend(title='VBF Signal Region', handles=handles, labels=new_labels, ncol=2)

        loc = MultipleLocator(100)
        ax.xaxis.set_major_locator(loc)

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        disttag = re.sub(r'_eta(\d)', '', distribution)

        outpath = pjoin(outdir, f"{disttag}_{year}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for distribution in ['ak4_pt0_eta0', 'ak4_pt1_eta1']:
        plot_jet_pt(acc, outtag, distribution=distribution)

if __name__ == '__main__':
    main()
