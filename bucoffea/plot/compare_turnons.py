#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import uproot
import warnings

from pprint import pprint
from coffea import hist

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate

from bucoffea.plot.style import markers, matplotlib_rc
from bucoffea.plot.util import (acc_from_dir, fig_ratio, lumi, merge_datasets,
                                merge_extensions, scale_xs_lumi)
from bucoffea.helpers.paths import bucoffea_path

from klepto.archives import dir_archive

matplotlib_rc()

pjoin = os.path.join

# Ignore warnings
warnings.filterwarnings('ignore')

def compare_turnons(acc, outtag, low_recoil=200):
    '''Compare turn-ons for SR and QCD CR'''
    variable = 'recoil'
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebinning
    newbin = hist.Bin('recoil',"Recoil (GeV)",np.array(list(range(0,500,20)) + list(range(500,1100,100))))
    h = h.rebin('recoil', newbin)

    # Output dir to save figures
    outdir = f'./output/trig_check_for_qcd/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        # Pick the data
        _h = h.integrate('dataset', f'MET_{year}') 

        # Num and denom for SR
        hnum_sr = _h.integrate('region', 'tr_sr_num')
        hden_sr = _h.integrate('region', 'tr_sr_den')

        # Num and denom for QCD CR
        hnum_cr = _h.integrate('region', 'tr_cr_qcd_num')
        hden_cr = _h.integrate('region', 'tr_cr_qcd_den')

        # Plot the efficiencies for the two cases
        fig, ax, rax = fig_ratio()

        error_opts = markers('data')
        error_opts['color'] = 'C0'

        hist.plotratio(hnum_sr, hden_sr,
                ax=ax,
                guide_opts={},
                unc='clopper-pearson',
                error_opts=error_opts,
                label='Signal Region'
                )

        error_opts['color'] = 'C1'

        hist.plotratio(hnum_cr, hden_cr,
                ax=ax,
                guide_opts={},
                unc='clopper-pearson',
                error_opts=error_opts,
                label='QCD Control Region',
                clear=False
                )

        xlim = ax.get_xlim()
        ax.set_xlim(200,xlim[-1])
        ax.set_xlabel('')
        ax.set_ylabel('Efficiency')
        ax.legend(title='Region')

        ax.set_title(f'MET Dataset: {year}')
        ax.set_ylim(0.8,1.1)

        # Plot the ratio of efficiencies
        eff_num = hnum_sr.values()[()] / hden_sr.values()[()]
        eff_den = hnum_cr.values()[()] / hden_cr.values()[()]

        centers = hnum_sr.axes()[0].centers()
        r = eff_num / eff_den
        rax.plot(centers, r, marker='o', ls='', color='k')

        rax.grid(True)
        rax.set_ylim(0.9,1.1)
        rax.set_ylabel('Eff in SR / Eff in CR')
        rax.set_xlabel('Recoil (GeV)')

        loc1 = MultipleLocator(0.05)
        loc2 = MultipleLocator(0.01)
        rax.yaxis.set_major_locator(loc1)
        rax.yaxis.set_minor_locator(loc2)

        # Save figure
        outpath = pjoin(outdir, f'eff_comp_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    acc.load('sumw')
    acc.load('sumw2')

    compare_turnons(acc, outtag)

if __name__ == '__main__':
    main()