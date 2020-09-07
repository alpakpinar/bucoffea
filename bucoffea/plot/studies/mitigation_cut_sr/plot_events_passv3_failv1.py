#!/usr/bin/env python

import os
import sys
import re
import warnings
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

XLABELS = {
    'ak4_eta0' : r'Leading jet $\eta$',
    'ak4_eta1' : r'Trailing jet $\eta$',
    'ak4_phi0' : r'Leading jet $\phi$',
    'ak4_phi1' : r'Trailing jet $\phi$',
    'ak4_nef0' : r'Leading jet neutral EM fraction',
    'ak4_nef1' : r'Trailing jet neutral EM fraction',
    'ak4_nhf0' : r'Leading jet neutral hadron fraction',
    'ak4_nhf1' : r'Trailing jet neutral hadron fraction'
}

def plot_events(acc, outtag, variable):
    '''Plot events which pass the v3 selection but fail the v1 selection.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Get the relevant region
    h = h.integrate('region', 'sr_vbf_passv3_failv1').integrate('dataset', 'MET_2017')

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax)
    ax.get_legend().remove()
    ax.set_title('Events passing EEv3, failing EEv1')
    ax.set_xlabel(XLABELS[variable])

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['ak4_eta0', 'ak4_eta1', 'ak4_phi0', 'ak4_phi1', 'ak4_nef0', 'ak4_nef1', 'ak4_nhf0', 'ak4_nhf1']
    for variable in variables:
        plot_events(acc, outtag, variable=variable)

if __name__ == '__main__':
    main()
