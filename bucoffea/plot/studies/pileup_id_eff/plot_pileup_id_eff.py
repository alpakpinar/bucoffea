#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.helpers.paths import bucoffea_path
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500.,2000., 2750., 3500., 5000.])
}
            
def get_dataset_for_proc(proc, year):
    proc_to_dataset = {
        'znunu' : f'ZJetsToNuNu.*{year}',
        'data' : f'MET_{year}'
    }

    return proc_to_dataset[proc]

def preprocess(h, acc, region, dist):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if dist in REBIN.keys():
        new_ax = REBIN[dist]
        h = h.rebin(new_ax.name, new_ax)

    h = h.integrate('region', region)

    return h

def calculate_pileupid_eff(acc_dict, outtag, proc='znunu', region='sr_vbf', dist='mjj'):
    '''For the given process in given region, calculate the efficiency of pileup ID.'''
    h_dict = {}

    for key, acc in acc_dict.items():
        acc.load(dist)
        h = acc[dist]
        h_dict[key] = preprocess(h, acc, region, dist)

    for year in [2017, 2018]:
        for key, h in h_dict.items():
            # Integrate over the relevant dataset for each year
            datasetregex = get_dataset_for_proc(proc, year)
            _h = h.integrate('dataset', re.compile(datasetregex))
            h_dict[key] = _h

        # Histograms are good to go! Take the noPU/PU ratio to get efficiencies as a function of desired variable
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
            'color':'k',
        }

        fig, ax = plt.subplots()
        hist.plotratio(
            h_dict['withPileup'],
            h_dict['noPileup'],
            ax=ax,
            error_opts=data_err_opts
        )
        
        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'puid_eff_{proc}_{dist}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    # Two accumulators: One of them has PU ID applied, the other one doesn't have PU ID
    acc_dict = {
        'withPileup' : dir_archive(bucoffea_path('submission/merged_2020-12-01_vbfhinv_03Sep20v7')),
        'noPileup' : dir_archive(bucoffea_path('submission/merged_2020-12-01_vbfhinv_03Sep20v7_nopuid')),
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')

    outtag = 'comp_01Dec20'

    for proc in ['znunu']:
        for dist in ['mjj', 'npvgood']:
            calculate_pileupid_eff(acc_dict, outtag,
                    proc=proc,
                    region='sr_vbf',
                    dist=dist)
    
if __name__ == '__main__':
    main()