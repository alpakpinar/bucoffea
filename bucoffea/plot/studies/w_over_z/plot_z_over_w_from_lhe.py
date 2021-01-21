#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from coffea import hist
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive

pjoin = os.path.join

def get_region(dataset, channel):
    mapping = {
        'DY' : {'electrons' : {
                    'lhe'  : 'cr_2e_lhe_vbf',
                    'reco' : 'cr_2e_vbf',
                }, 
                'muons' : {
                    'lhe'  : 'cr_2m_lhe_vbf',
                    'reco' : 'cr_2m_vbf',
                }
            },
        'W' : {'electrons' : {
                    'lhe'  : 'cr_1e_lhe_vbf',
                    'reco' : 'cr_1e_vbf',
                }, 
                'muons' : {
                    'lhe'  : 'cr_1m_lhe_vbf',
                    'reco' : 'cr_1m_vbf',
                }
            },
    }
    return mapping[dataset][channel]

def get_plot_tag(channel, year):
    mapping = {
        'electrons' : f'$Z(ee)$ / $W(e\\nu)$ {year}',
        'muons' : f'$Z(\\mu\\mu)$ / $W(\\mu\\nu)$ {year}',
    }
    return mapping[channel]

def plot_z_over_w(acc, outtag, channel='electrons', compare_with_root=None):
    '''Plot Z/W ratio, where events are selected using regular jet cuts + LHE-level lepton cuts.'''
    distribution = 'mjj'
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Rebin mjj
    mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_ax)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
    }
    
    for year in [2017, 2018]:
        h_z_lhe = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}')).integrate('region', get_region('DY', channel)['lhe'])
        h_w_lhe = h.integrate('dataset', re.compile(f'WJetsToLNu.*{year}')).integrate('region', get_region('W', channel)['lhe'])

        # Plot the ratio
        fig, ax = plt.subplots()
        hist.plotratio(h_z_lhe, h_w_lhe, 
            ax=ax,
            unc='num',
            label='LHE',
            error_opts=data_err_opts
            )


        # Can compare with the ratio obtained from RECO-level lepton selection
        # Also without the MET cut so that we're consistent
        if compare_with_root is not None:
            r = compare_with_root[f'qcd_z_over_w_electrons_without_met_cut_{year}']
            centers = 0.5 * np.sum(r.bins, axis=1)

            ax.plot(centers, r.values, ls='', label='RECO', marker='o')

        ax.legend(title='Lepton requirement')

        ax.text(0., 1., get_plot_tag(channel, year),
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )

        ax.set_xlim(200,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel('Ratio')

        if channel == 'electrons':
            ax.text(1., 1., 'No MET cut applied',
                fontsize=14,
                ha='right',
                va='bottom',
                transform=ax.transAxes
                )

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_{channel}_from_lhe_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved:{outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    # Compare the Z/W obtained by LHE requirements to RECO ones (without MET cut)
    rootpath = './output/merged_2021-01-08_vbfhinv_1e_cr_without_metcut/root/z_over_w_with_without_met_cut.root'
    rootfile = uproot.open(rootpath)

    for channel in ['electrons', 'muons']:
        if channel == 'electrons':
            compare_with_root = rootfile
        else:
            compare_with_root = None
        plot_z_over_w(acc, outtag, channel=channel, compare_with_root=compare_with_root)

if __name__ == '__main__':
    main()