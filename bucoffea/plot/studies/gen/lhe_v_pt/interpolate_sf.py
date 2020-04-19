#!/usr/bin/env python
from textwrap import dedent
import uproot
import argparse
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from bucoffea.helpers import exponential
import numpy as np

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Input file containing k-factors.', default='2017_gen_v_pt_qcd_sf.root')
    parser.add_argument('--photons_only', help='Only do the fitting for photon k-factors.', action='store_true')
    parser.add_argument('--outtag', help='Tag for output naming.')
    args = parser.parse_args()
    return args

args = parse_cli()

f = uproot.open(args.inpath)

pretty_name = {
    'wjet' : 'W',
    'dy' : 'DY',
    'gjets' : r'$\gamma$+jets'
}

process_list = ['gjets'] if args.photons_only else ['wjet','dy','gjets']

for process in process_list:
    x = {}
    y = {}

    for selection in ['inclusive','monojet','vbf']:
        if process in ['wjet','dy']:
            h = f[f'{process}_combined_{selection}']
        elif process in ['gjets']:
            h = f[f'{process}_stat1_{selection}']

        x[selection] = 0.5*(h.bins[:,0]+h.bins[:,1])
        y[selection] = h.values
    
    for sel in x.keys():
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": (2, 1)}, sharex=True)

        # if not sel=='monojet':
        #     continue
        ax.plot(x[sel],y[sel],'o',label=f'{pretty_name[process]}, {sel} selection')
        popt, _ = curve_fit(exponential, x[sel], y[sel], bounds = (0, [3, 5e-3, 1]))

        ix = np.linspace(min(x[sel]), max(x[sel]),1e3)
        ax.plot(ix, exponential(ix, *popt),'-', label='Exponential fit')
        rax.plot(x[sel], y[sel] / exponential(x[sel], *popt) ,'-o')

        rax.set_ylim(0.9,1.1)
        ax.legend(fontsize=14)
        rax.set_xlabel('$p_{T}$ (V) (GeV)',fontsize=14)
        ax.set_ylabel('NLO SF',fontsize=14)
        rax.set_ylabel('Points / fit',fontsize=14)
        rax.plot([min(x[sel]),max(x[sel])],[1,1],'--',color='gray')


        text = dedent(
        f'''
        $a \\times exp(-b\\times p_{{T}}) + c$
        a = {popt[0]:.3f}
        b = {1e3*popt[1]:.3f} / TeV
        c = {popt[2]:.3f}
        '''
        )

        ax.text(0.6, 0.3, text,
                fontsize=10,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes
               )
        
        # Fix y-limit for photon k-factor plots
        if args.photons_only:
            ax.set_ylim(1.3, 2.0)

        # Save figure
        outdir = f'output/interpolation/{args.outtag}' if args.outtag else 'output/interpolation'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'interpolation_{sel}_{process}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
