#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

def plot_calopf_met(acc, outtag, proc, region):
    '''Plot calo/PF MET distribution for some processes in the given region.'''
    distribution = 'dpfcalo_cr'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)


    for year in [2017, 2018]:
        # For now, plot the leading backgrounds in each CR
        dataset_by_region = {
            'cr_(1m|1e).*no_calo.*' : f'WJetsToLNu.*{year}',
            'cr_g.*no_calo.*' : f'GJets_DR-0p4.*{year}',
        }
        _datasetregex = None
        for regionregex, datasetregex in dataset_by_region.items():
            if re.match(regionregex, region):
                _datasetregex = datasetregex

        if not _datasetregex:
            raise RuntimeError(f'Could not find a proper dataset regex for: {region}')

        _h = h.integrate('dataset', _datasetregex)

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax)

        ax.axvline(0.5, ymin=0, ymax=1, color='red')
        ax.axvline(-0.5, ymin=0, ymax=1, color='red')

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'dpfcalo_{region}_{proc}_{year}.pdf')
        fig.savefig(outpath)

        print(f'File saved: {outpath}')
        
def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    regions = [
        'cr_1m_vbf_no_calo_cut',
        'cr_1e_vbf_no_calo_cut',
        'cr_g_vbf_no_calo_cut',
    ]
    
    for region in regions:
        plot_calopf_met(acc, outtag, proc=None, region=region)

if __name__ == '__main__':
    main()
