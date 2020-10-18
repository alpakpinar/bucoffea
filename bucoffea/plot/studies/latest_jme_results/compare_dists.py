#!/usr/bin/env python

import os
import sys
import re
import argparse
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
import matplotlib.ticker
import warnings

warnings.filterwarnings('ignore')

pjoin = os.path.join

REBIN = {
    'met' : hist.Bin('met', r'$p_T^{miss} \ (GeV)$', 12, 0, 300),
    'vpt' : hist.Bin('vpt', r'$p_T(Z) \ (GeV)$', 30, 0, 1200),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(0,1000,20)) )
}

region_to_title = {
    'norecoil_nojpt' : r'$Z(ee) \ {}$: Jet Inclusive',
    'norecoil_jptv2' : r'$Z(ee) \ {}$: Jet $p_T > 50 \ GeV$',
    'norecoil'       : r'$Z(ee) \ {}$: Jet $p_T > 100 \ GeV$'
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path to merged coffea files.')
    parser.add_argument('--variables', help='Variables to plot.', nargs='*', default=['met', 'vpt', 'ak4_pt0'])
    args = parser.parse_args()
    return args

def compare_dists(acc, outtag, variable='met', variation='jesTotal', year=2017, region='norecoil'):
    '''Compare distributions of the given variable with the given variations.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if variable in REBIN.keys():
        if variable == 'ak4_pt0':
            h = h.rebin('jetpt', REBIN[variable])
        else:
            h = h.rebin(variable, REBIN[variable])

    # Integrate over the DY dataset for the given year, get the correct region
    h = h.integrate('dataset', re.compile(f'DYJets.*{year}'))
    
    # Get the nominal (non-varied) histogram
    h_nom = h.integrate('region', f'cr_2e_j_{region}')
    
    # Up and down variations for the specified variation
    h_up = h.integrate('region', f'cr_2e_j_{region}_{variation}Up')
    h_down = h.integrate('region', f'cr_2e_j_{region}_{variation}Down')

    # Plot distributions
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hist.plot1d(h_nom, ax=ax)
    hist.plot1d(h_up, ax=ax, clear=False)
    hist.plot1d(h_down, ax=ax, clear=False)

    new_labels = [
        'Nominal',
        f'{variation}Up',
        f'{variation}Down'
    ]
    # Fix the labels
    ax.legend(labels=new_labels)

    ax.set_title(region_to_title[region].format(year))
    ax.set_yscale('log')
    if region == 'norecoil':
        ax.set_ylim(1e-1,1e6)
    elif region == 'norecoil_jptv2':
        ax.set_ylim(1e-1,1e7)
    elif region == 'norecoil_nojpt':
        ax.set_ylim(1e-1,1e8)
    ax.set_xlabel('')

    # Plot ratios to nominal in ratio pad
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'elinewidth': 1
    }

    data_err_opts['color'] = 'C1'
    hist.plotratio(h_up, h_nom, unc='num', ax=rax, clear=False, error_opts=data_err_opts)
    data_err_opts['color'] = 'C2'
    hist.plotratio(h_down, h_nom, unc='num', ax=rax, clear=False, error_opts=data_err_opts)

    rax.grid(True)
    if variable == 'met':
        rax.set_ylim(0.7,1.3)
        loc = matplotlib.ticker.MultipleLocator(base=0.1)
    elif variable == 'ak4_pt0':
        if variation == 'jesTotal':
            rax.set_ylim(0.7,1.3)
            loc = matplotlib.ticker.MultipleLocator(base=0.1)
        else:
            rax.set_ylim(0.9,1.1)
            loc = matplotlib.ticker.MultipleLocator(base=0.05)
    elif variable == 'vpt':
        rax.set_ylim(0.9,1.1)
        loc = matplotlib.ticker.MultipleLocator(base=0.05)
    
    rax.yaxis.set_major_locator(loc)
    
    rax.set_ylabel('Ratio to Nominal')

    # Save figure
    outdir = f'./output/{outtag}/{region}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variable}_{variation}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    args = parse_cli()
    inpath = args.inpath
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        for region in ['norecoil', 'norecoil_nojpt', 'norecoil_jptv2']:
            for variable in ['vpt', 'met', 'ak4_pt0']:
                if variable not in args.variables:
                    continue 
                for variation in ['jesTotal', 'unclustEn']:
                    compare_dists(acc, outtag, variable=variable, variation=variation, year=year, region=region)

if __name__ == '__main__':
    main()
