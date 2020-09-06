#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
}

def compare_ee_signal(acc, outtag, variable='mjj', year=2017):
    '''
    Compare the signal yields for two cases:
    1. When the neutral EM fraction cut is applied to all jets
    2. When it is applied only to the jets in endcap
    '''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if variable == 'mjj':
        h = h.rebin('mjj', REBIN['mjj'])

    h = h.integrate('dataset', re.compile(f'VBF.*{year}'))[re.compile('^sr_vbf((?!_no_veto).)*$')]

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region', overflow='over')

    ax.set_xlabel('')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e6)
    ax.set_title(f'VBF {year}')

    # Plot the ratio
    h_nom = h.integrate('region', 'sr_vbf')
    h_ee = h.integrate('region', 'sr_vbf_eeOnly')

    # Percent of events we lose with the nominal cut, w.r.t. EE cut
    r = h_ee.values(overflow='over')[()] / h_nom.values(overflow='over')[()]
    centers = h_nom.axes()[0].centers(overflow='over')
    rax.plot(centers, r, ls='', marker='o', color='black')

    rax.grid(True)
    rax.set_ylim(0.8,1.2)
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('EE-only cut / Nominal cut')

    xlim = rax.get_xlim()
    rax.plot(xlim, [1.0, 1.0], 'r--')
    rax.plot(xlim, [1.05, 1.05], 'r--')
    rax.set_xlim(xlim)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'{variable}_signal_comp_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        compare_ee_signal(acc, outtag, year=year)

if __name__ == '__main__':
    main()
