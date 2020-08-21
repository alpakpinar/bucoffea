#!/usr/bin/env python

import os
import sys
import re
import matplotlib.ticker
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

# ========================
# Script to check the efficiency of GEN-level jet 
# matching requirement on H(inv) sample
# ========================

REBIN = {
    'ak4_pt0' : hist.Bin('jetpt', r'Jet $p_T$ (GeV)', 25, 0, 1000),
    'met' : hist.Bin('met', r'$p_T^{miss}$ (GeV)', list(range(0,500,50)))
}

XLABELS = {
    'ak4_pt0' : r'Jet $p_T$ (GeV)',
    'ak4_eta0' : r'Jet $\eta$',
    'ak4_nef0' : 'Jet Neutral EM Fraction',
    'met' : r'$p_T^{miss}$ (GeV)'
}

def check_gen_efficiency(acc, outtag, variable='ak4_eta0'):
    '''Calculate and plot the efficiency of the gen-level jet requirement on the leading jet.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin, if neccessary
    if variable in REBIN.keys():
        if variable == 'ak4_pt0':
            h = h.rebin('jetpt', REBIN[variable])
        elif variable == 'met':
            h = h.rebin('met', REBIN[variable])

    # Only signal MC is present, integrate out the dataset
    h = h.integrate('dataset')[re.compile('sr_.*gen.*')]

    # Plot the comparison of the distributions with and without the GEN-jet requirement
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h, ax=ax, overlay='region')

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 1e5)
    ax.set_title(r'$H(inv) \ 2017$')

    # Calculate and plot the ratio
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k'
    }

    h_num = h.integrate('region', 'sr_with_gen_requirement')
    h_denom = h.integrate('region', 'sr_no_gen_requirement')
    hist.plotratio(h_num, h_denom, unc='num', ax=rax, error_opts=data_err_opts)

    rax.set_xlabel(XLABELS[variable])
    rax.set_ylim(0.97,1.03)
    rax.set_ylabel('With GEN matching / without')

    loc = matplotlib.ticker.MultipleLocator(base=0.01)
    rax.yaxis.set_major_locator(loc)
    rax.grid(True)

    # Save figure
    outdir = f'./output/{outtag}/gen_check'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'gen_requirement_eff_check_{variable}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
        inpath,
        serialized=True,
        memsize=1e3,
        compression=0
    )
    
    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    variables = ['ak4_eta0', 'ak4_pt0']
    for variable in variables:
        check_gen_efficiency(acc, outtag, variable=variable)

if __name__ == '__main__':
    main()
