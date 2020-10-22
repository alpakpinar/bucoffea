#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def labels_for_variations(variation):
    mapping = {
        'jer' : 'JER',
        'unclustEn' : 'Unclust. Energy',
        'jesTotal' : 'JES Total'
    }

    return mapping[variation]

def get_varied_ratios(h_data, h_mc, variation, region):
    '''Given data and MC histograms, get the varied data/MC ratio for the specified variation.'''
    data_vals = h_data.integrate('dataset').values()[()]

    # Up and down variations for MC
    mc_up = h_mc.integrate('region', f'cr_2e_j_{region}_{variation}Up').values()[()]
    mc_down = h_mc.integrate('region', f'cr_2e_j_{region}_{variation}Down').values()[()]

    # Up and down data / MC ratios
    data_over_mc_up = data_vals / mc_up
    data_over_mc_down = data_vals / mc_down

    return data_over_mc_up, data_over_mc_down

def data_mc_comparison_plot(acc, distribution='met', year=2017, smear=False, region='norecoil'):
    '''For the given distribution and year, construct data/MC comparison plot with all the JME variations'''
    # The list of variations, depending on we use smearing or not
    if smear:
        variations = ['jer', 'jesTotal']
    else:
        variations = ['jesTotal', 'unclustEn']

    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # TODO: Rebinning?

    # Start plotting 
    fig, ax, rax = fig_ratio()

    # Get data first
    h_data = h.integrate('region', f'cr_2e_j_{region}')[f'EGamma_{year}']

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    hist.plot1d(h_data, ax=ax, overlay='dataset', error_opts=data_err_opts)

    # Next, get MC (the nominal one)
    # TODO: Check the regex here!
    h_mc_nom = h.integrate('region', f'cr_2e_j_{region}')[ re.compile(f'(?!(EGamma)).*{year}') ]
    hist.plot1d(h_mc_nom, ax=ax, overlay='dataset', clear=False)

    ax.set_xlabel('')

    # To test the above regex
    pprint(h_mc_nom.values())

    # Plot the ratio of nominal data/MC values
    hist.plotratio(h_data.integrate('dataset'), h_mc_nom.integrate('dataset'),
            ax=rax,
            denom_fill_opts={},
            guide_opts={},
            unc='num',
            error_opts=data_err_opts
            )

    rax.grid(True)
    rax.set_ylim(0,2)
    rax.set_ylabel('Data / MC')
    rax.set_xlabel(f'{distribution.capitalize()} (GeV)')

    loc1 = matplotlib.ticker.MultipleLocator(base=0.5)
    loc2 = matplotlib.ticker.MultipleLocator(base=0.1)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)
    
    # Now, for each variation, get the varied ratios and plot them in the ratio pad
    data_mc_nom = h_data.integrate('dataset').values()[()] / h_mc_nom.integrate('dataset').values()[()]
    edges = h_data.integrate('dataset').axes()[0].edges()

    for variation in variations:
        h_mc_var = h.integrate('dataset', re.compile(f'(?!(EGamma)).*{year}'))
        data_mc_up, data_mc_down = get_varied_ratios(h_data, h_mc_var, variation, region)

        # Fractional change in data/MC compared to nominal
        frac_change_up = data_mc_up / data_mc_nom
        frac_change_down = data_mc_down / data_mc_nom

        opts = {'step': 'post', 'linewidth': 0, 'label' : labels_for_variations(variation) }

        # TODO: To be tested
        rax.fill_between(edges, frac_change_up, frac_change_down, **opts)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    data_mc_comparison_plot(acc,
            distribution='met',
            year=2017,
            smear=False,
            region='norecoil'
            )

if __name__ == '__main__':
    main()