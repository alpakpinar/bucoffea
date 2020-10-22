#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
import warnings
import argparse

from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

REBIN = {
    'met' : hist.Bin('met', r'MET (GeV)', 20, 0, 500),
    'vpt' : hist.Bin('vpt', r'$p_T(Z) \ (GeV)$', 25, 0, 1000)
}

colors = {
    '.*DY.*' : '#ffffcc',
    '.*Diboson.*' : '#4292c6',
    'Top.*' : '#6a51a3',
}

xlabels = {
    'vpt' : r'$p_T(Z) \ (GeV)$',
    'met' : r'MET (GeV)',
    'ak4_pt0' : r'Leading jet $p_T$ (GeV)',
    'ak4_eta0' : r'Leading jet $\eta$'
}

region_labels = {
    'norecoil' : r'Leading Jet $p_T > 100$',
    'norecoil_jptv2' : r'Leading Jet $p_T > 50$',
    'norecoil_nojpt' : r'Jet Inclusive'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='The path containing merged coffea files.')
    parser.add_argument('--distribution', help='Regex matching the distributions to be plotted.', default='.*')
    parser.add_argument('--smeared', help='Flag showing that the input has smearing applied.', action='store_true')
    args = parser.parse_args()
    
    return args

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

def data_mc_comparison_plot(acc, outtag, distribution='met', year=2017, smear=False, region='norecoil'):
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

    h.axis('dataset').sorting = 'integral'

    if distribution in REBIN.keys():
        h = h.rebin(distribution, REBIN[distribution])

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
    h_mc_nom = h.integrate('region', f'cr_2e_j_{region}')[ re.compile(f'(Top_FXFX|DYJetsToLL|Diboson).*{year}') ]
    hist.plot1d(h_mc_nom, ax=ax, overlay='dataset', stack=True, clear=False)

    ax.set_xlabel('')
    ax.set_yscale('log')
    if region != 'norecoil_nojpt':
        ax.set_ylim(1e-2,1e6)
    else:
        ax.set_ylim(1e-1,1e8)

    # Apply correct colors to BG histograms
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        col = None
        for k, v in colors.items():
            if re.match(k, label):
                col = v
                break
        if col:
            handle.set_color(col)
            handle.set_linestyle('-')
            handle.set_edgecolor('k')

    # Update legend
    ax.legend(ncol=1)

    ax.set_title(region_labels[region])

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
    rax.set_xlabel(xlabels[distribution])

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
        # rax.fill_between(edges, frac_change_up, frac_change_down, **opts)

    # Save figure
    if smear:
        outdir = f'./output/{outtag}/data_mc/smeared'
    else:
        outdir = f'./output/{outtag}/data_mc/not_smeared'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'data_mc_comp_{distribution}_{region}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    args = parse_cli()
    inpath = args.inpath

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    regions = [
        'norecoil',
        'norecoil_nojpt',
        'norecoil_jptv2'
    ]

    for year in [2017, 2018]:
        for region in regions:
            for distribution in ['met', 'vpt', 'ak4_pt0', 'ak4_eta0']:
                if not re.match(distribution, args.distribution):
                    continue
                data_mc_comparison_plot(acc, outtag,
                        distribution=distribution,
                        year=year,
                        smear=args.smeared,
                        region=region
                        )

if __name__ == '__main__':
    main()