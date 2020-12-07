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

def plot_jet_id_unc_on_ratio(acc, outtag):
    '''Plot the jet ID uncertainties on Z/W ratio.'''
    # Output directory to save plots
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    acc.load('mjj')
    h = acc['mjj']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)

    for year in [2017, 2018]:
        _h_z = h.integrate('dataset', re.compile(f'ZJetsToNuNu.*{year}'))
        _h_w = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))

        h_z = {
            'jetsfUp' : _h_z.integrate('region', 'sr_vbf_no_veto_all_jetsfUp'),
            'jetsfDown' : _h_z.integrate('region', 'sr_vbf_no_veto_all_jetsfDown'),
            'nom' : _h_z.integrate('region', 'sr_vbf_no_veto_all'),
        }

        h_w = {
            'jetsfUp' : _h_w.integrate('region', 'sr_vbf_no_veto_all_jetsfUp'),
            'jetsfDown' : _h_w.integrate('region', 'sr_vbf_no_veto_all_jetsfDown'),
            'nom' : _h_w.integrate('region', 'sr_vbf_no_veto_all'),
        }
        
        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        ratios = {}

        fig, ax, rax = fig_ratio()
        for key in h_z.keys():
            hist.plotratio(h_z[key], h_w[key], 
                    ax=ax, 
                    unc='num', 
                    label=key, 
                    error_opts=data_err_opts,
                    clear=False
                    )

            # Store the ratios
            ratios[key] = h_z[key].values()[()] / h_w[key].values()[()]

        labels = [
            'Jet ID SF up',
            'Jet ID SF down',
            'Nominal',
        ]

        ax.legend(labels=labels)
        ax.set_ylabel(r'$Z(\nu\nu) \ / \ W(\ell\nu)$', fontsize=14)
        ax.set_xlabel('')

        ax.text(1., 1., year,
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        r_nom = ratios['nom']
        centers = h_z['nom'].axis('mjj').centers()

        for key, r in ratios.items():
            if key == 'nom':
                continue
            rr = r / r_nom
            rax.plot(centers, rr, **data_err_opts)

        rax.grid(True)
        rax.set_ylim(0.95, 1.05)
        rax.set_ylabel('Ratio to Nom.', fontsize=14)
        rax.set_xlabel(r'$M_{jj} \ (GeV)$', fontsize=14)

        rax.axhline(1, xmin=0, xmax=1, color='k')

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_variations_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_jet_id_unc_on_ratio(acc, outtag)

if __name__ == '__main__':
    main()