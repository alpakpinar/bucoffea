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

def get_ylabel(ratio_tag):
    '''For the given ratio rag, get the y-label to be used in the plot.'''
    mapping = {
        'zmumu_over_wmunu' : r'$Z(\mu\mu)$ / $W(\mu\nu)$',
        'zee_over_wenu' : r'$Z(ee)$ / $W(e\nu)$',
        'zmumu_over_znunu' : r'$Z(\mu\mu)$ / $Z(\nu\nu)$',
        'zee_over_znunu' : r'$Z(ee)$ / $Z(\nu\nu)$',
        'wmunu_over_znunu' : r'$W(\mu\nu)$ / $Z(\nu\nu)$',
        'wenu_over_znunu'  : r'$W(e\nu)$ / $Z(\nu\nu)$',
    }

    return mapping[ratio_tag]

def plot_cr_ratios_with_without_bweight(acc, outtag, year, distribution='mjj'):
    '''Plot ratio of CR total backgrounds (prefit MC) with and without b-jet weights.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)

    # Output directory to save plots
    outdir = f'./output/{outtag}/mc_vs_mc/cr_ratios'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ratios = [
        {'num' : {
            'dataset' : re.compile(f'DYJetsToLL.*{year}'),
            'region'  : re.compile(f'cr_2m_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'WJetsToLNu.*{year}'),
            'region'  : re.compile(f'cr_1m_vbf.*')
            },
        'ratio_tag' : 'zmumu_over_wmunu'
        },
        {'num' : {
            'dataset' : re.compile(f'DYJetsToLL.*{year}'),
            'region'  : re.compile(f'cr_2e_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'WJetsToLNu.*{year}'),
            'region'  : re.compile(f'cr_1e_vbf.*')
            },
        'ratio_tag' : 'zee_over_wenu'
        },
        {'num' : {
            'dataset' : re.compile(f'DYJetsToLL.*{year}'),
            'region'  : re.compile(f'cr_2m_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'ZJetsToNuNu.*{year}'),
            'region'  : re.compile(f'sr_vbf(?!(_no_veto_all)).*')
            },
        'ratio_tag' : 'zmumu_over_znunu'
        },
        {'num' : {
            'dataset' : re.compile(f'DYJetsToLL.*{year}'),
            'region'  : re.compile(f'cr_2e_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'ZJetsToNuNu.*{year}'),
            'region'  : re.compile(f'sr_vbf(?!(_no_veto_all)).*')
            },
        'ratio_tag' : 'zee_over_znunu'
        },
        {'num' : {
            'dataset' : re.compile(f'WJetsToLNu.*{year}'),
            'region'  : re.compile(f'cr_1m_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'ZJetsToNuNu.*{year}'),
            'region'  : re.compile(f'sr_vbf(?!(_no_veto_all)).*')
            },
        'ratio_tag' : 'wmunu_over_znunu'
        },
        {'num' : {
            'dataset' : re.compile(f'WJetsToLNu.*{year}'),
            'region'  : re.compile(f'cr_1e_vbf.*')
            },
        'den' : {
            'dataset' : re.compile(f'ZJetsToNuNu.*{year}'),
            'region'  : re.compile(f'sr_vbf(?!(_no_veto_all)).*')
            },
        'ratio_tag' : 'wenu_over_znunu'
        },
    ]
    for ratio in ratios:
        h_num = h.integrate('dataset', ratio['num']['dataset'])[ ratio['num']['region'] ]
        h_den = h.integrate('dataset', ratio['den']['dataset'])[ ratio['den']['region'] ]

        # In each histograms, two regions should be stored by this point:
        # 1. The histogram with the b-weights applied (the first one)
        # 2. The histogram with hard b-veto applied (the second one) 
        # Here, we will plot the ratio for the two cases

        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        fig, ax, rax = fig_ratio()
        h_num_with_bweight = h_num.integrate('region', h_num.identifiers('region')[0])
        h_den_with_bweight = h_den.integrate('region', h_den.identifiers('region')[0])
        hist.plotratio(
            h_num_with_bweight,
            h_den_with_bweight,
            ax=ax,
            unc='num',
            label='With b-weights',
            error_opts=data_err_opts
        )

        h_num_with_bveto = h_num.integrate('region', h_num.identifiers('region')[1])
        h_den_with_bveto = h_den.integrate('region', h_den.identifiers('region')[1])
        hist.plotratio(
            h_num_with_bveto,
            h_den_with_bveto,
            ax=ax,
            unc='num',
            label='With hard b-veto',
            error_opts=data_err_opts,
            clear=False
        )

        ax.set_ylabel( get_ylabel(ratio['ratio_tag']), fontsize=14)
        ax.set_xlabel('')
        ax.set_xlim(200,5000)
        # ax.set_ylim(0,0.25)
        ax.legend()

        ax.text(1., 1., year, 
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Plot the ratio of ratios
        sumw_num_with_bweight, sumw2_num_with_bweight = h_num_with_bweight.values(sumw2=True)[()]
        sumw_den_with_bweight = h_den_with_bweight.values()[()]

        ratio_with_bweight = sumw_num_with_bweight / sumw_den_with_bweight
        ratio_with_bveto = h_num_with_bveto.values()[()] / h_den_with_bveto.values()[()]

        dratio = ratio_with_bweight / ratio_with_bveto

        # Stat error on the ratio with b-weights
        ratio_with_bweight_err = np.sqrt(sumw2_num_with_bweight) / sumw_den_with_bweight

        percent_err_on_ratio = ratio_with_bweight_err / ratio_with_bveto

        err_interval = np.vstack((
            1+percent_err_on_ratio,
            1-percent_err_on_ratio,
        ))

        edges = h_num_with_bveto.axes()[0].edges()
        centers = h_num_with_bveto.axes()[0].centers()

        rax.plot(centers, dratio, color='k', **data_err_opts)

        fill_opts = {
            'step' : 'post',
            'color' : 'gray',
            'edgecolor' : 'none',
            'linewidth' : 0,
            'alpha' : 0.5
        }

        rax.fill_between(edges, 
            np.r_[err_interval[0,:], err_interval[0,-1]], 
            np.r_[err_interval[1,:], err_interval[1,-1]], 
            **fill_opts
        )

        rax.grid(True)
        rax.set_ylim(0.9,1.1)
        rax.set_ylabel('With b-weight / veto')
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    
        loc1 = MultipleLocator(0.05)
        rax.yaxis.set_major_locator(loc1)

        # Save figure
        outpath = pjoin(outdir, f'{ratio["ratio_tag"]}_bveto_comp_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        plot_cr_ratios_with_without_bweight(acc, outtag, year)

if __name__ == '__main__':
    main()