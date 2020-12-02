#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def compare_vbf_signals(acc, outtag, variable='mjj'):
    '''Compare VBF signals for 2018: with and without dipole recoil.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    # h = merge_datasets(h)

    h = h.integrate('region', 'sr_vbf_no_veto_all')

    # Will only look at 2018 for this comparison
    year = 2018

    h = h[ re.compile(f'VBF_HToInv.*{year}') ]

    fig, ax, rax = fig_ratio()
    hist.plot1d(h, ax=ax, overlay='dataset')

    ax.set_xlabel('')
    ax.set_title(r'VBF $H(inv)$ 2018: Dipole Recoil Comparison')

    # Simplify legend labels
    handles, labels = ax.get_legend_handles_labels()

    for handle, label in zip(handles, labels):
        if 'DipoleRecoil' in label:
            newlabel = 'With dipole recoil'
        else:
            newlabel = 'Without dipole recoil'

        handle.set_label(newlabel)
    
    ax.legend(handles)

    # Plot ratio
    hnum = h.integrate('dataset', f'VBF_HToInvisible_M125_PSweights_withDipoleRecoil_pow_pythia8_{year}') # With DR
    hden = h.integrate('dataset', f'VBF_HToInvisible_M125_PSweights_pow_pythia8_{year}') # Without DR

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    hist.plotratio(hnum, hden,
        ax=rax,
        unc='num',
        error_opts=data_err_opts
    )

    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('With DR / Without')
    rax.set_ylim(0.8,1.2)
    rax.grid(True)

    loc = MultipleLocator(0.05)
    rax.yaxis.set_major_locator(loc)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'vbf_signal_comparison.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    compare_vbf_signals(acc, outtag)

if __name__ == '__main__':
    main()