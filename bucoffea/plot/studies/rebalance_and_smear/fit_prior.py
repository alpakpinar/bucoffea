#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

pjoin = os.path.join

def func(x,a,b,c):
    return c * x * np.exp(-a*(x-b))

def main():
    inputrootpath = './output/merged_2021-01-23_qcd_prior_study_qcd_18Jan21v7_with_htcut/rbs_prior_dists.root'
    inputrootfile = uproot.open(inputrootpath)

    outtag = 'merged_2021-01-23_qcd_prior_study_qcd_18Jan21v7_with_htcut'
    outdir = f'./output/{outtag}/prior_fits'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ht_bins = [
        'ht_100_to_300',
        'ht_300_to_500',
        'ht_500_to_700',
        'ht_700_to_900',
        'ht_900_to_1300',
        'ht_1300_to_2000',
        'ht_2000_to_5000',
    ]

    for ht_bin in ht_bins:
        h = inputrootfile[f'gen_htmiss_{ht_bin}_2017']

        fig, ax = plt.subplots()
        # Original prior histogram
        hep.histplot(h.values, h.edges, ax=ax, label='Binned prior')
    
        xcenters = 0.5 * np.sum(h.bins, axis=1)
    
        # Fit the prior distribution
        popt, _ = curve_fit(func, xcenters, h.values)
    
        x = np.linspace(0,500,500)
        y = func(x, *popt)
        ax.plot(x, y, label='Fitted prior, $f(x)$')
        
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim(1e-9,1e1)
        ax.set_xlabel(r'$H_T^{miss} \ (GeV)$')
        ax.set_ylabel('Normalized Counts')
    
        ht_interval = re.findall('ht_\d+_to_\d+', ht_bin)[0]
        ht_int_split = ht_interval.split('_')
        lo, hi = int(ht_int_split[1]), int(ht_int_split[-1])
    
        ax.set_title(f'${lo} < H_T < {hi} \ GeV$', fontsize=14)
    
        ax.text(0., 1., '2017',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )
    
        a,b,c = popt
    
        sgn = '+' if b < 0 else '-'
    
        ax.text(0.98, 0.75, r'$f(x) = {:.2f}*x*e^{{-{:.2f}(x{}{:.2f})}}$'.format(c,a,sgn,np.abs(b)), 
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
            )
    
        ax.text(0.98, 0.65, r'$p_{{max}}$ @ $H_T^{{miss}} = {:.2f} \ GeV$'.format(1/a), 
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
            )
    
        # Save figure
        outpath = pjoin(outdir, f'fit_{ht_bin}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

if __name__ == '__main__':
    main()