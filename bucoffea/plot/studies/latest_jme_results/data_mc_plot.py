#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
import warnings
import argparse
import copy

from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from bucoffea.plot.style import matplotlib_rc
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

matplotlib_rc()

warnings.filterwarnings('ignore')

REBIN = {
    'met' : hist.Bin('met', r'MET (GeV)', 40, 0, 200),
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

def get_x_label(distribution, smeared=False):
    '''For the given distribution, get the x-label.'''
    xlabels = {
        'vpt' : r'$p_T(Z) \ (GeV)$',
        'ak4_pt0' : r'Leading jet $p_T$ (GeV)',
        'ak4_eta0' : r'Leading jet $\eta$'
    }

    if distribution != 'met':
        return xlabels[distribution]
    else:
        if smeared:
            return 'T1Smear MET (GeV)'
        else:
            return 'T1 MET (GeV)'

    raise RuntimeError(f'Could not find a label for distribution: {distribution}')

def get_variation_label(variations):
    '''Get legend label for the combined variations.'''
    if len(variations) == 1:
        label = variations[0] 
    else:
        label = ' + '.join(variations)
    return label

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='The path containing merged coffea files.')
    parser.add_argument('--distribution', help='Regex matching the distributions to be plotted.', default='.*')
    parser.add_argument('--smeared', help='Flag showing that the input has smearing applied.', action='store_true')
    parser.add_argument('--variations', help='List of variations to be plotted in the ratio pad.', nargs='*', default=None)
    parser.add_argument('--plot_with_sf', help='If specified, only plot data/MC with the SF in place.', action='store_true')
    args = parser.parse_args()
    
    return args

def labels_for_variations(variation):
    mapping = {
        'jer' : 'JER',
        'unclustEn' : 'Unclust. Energy',
        'jesTotal' : 'JES Total'
    }

    return mapping[variation]

def compute_integral(h):
    '''
    Compute the integral of the given histogram.
    NOTE: Give this guy 1D histograms only!
    '''
    bin_edges = h.axes()[0].edges()
    bin_widths = np.diff(bin_edges)

    vals = h.values()[()]
    integral = np.sum(vals * bin_widths)
    
    return integral

def get_data_mc_sf_from_integral(h_data, h_mc):
    '''
    Calculate the integral of data and MC distributions.
    Make a data/MC SF out of the ratio between the two integrals.
    '''
    integral_data = compute_integral(h_data)
    integral_mc = compute_integral(h_mc)

    # Calculate the data/MC ratio
    sf = integral_data / integral_mc
    return sf

def get_combined_variations(h_data, h_mc, variations, region, smear=False):
    '''
    Given data and MC histograms, get the varied data/MC ratio for the specified variation.
    Combine the given variations (in quadrature) to get the combined unc.
    '''
    data_vals = h_data.integrate('dataset').values()[()]

    # First, get the nominal data/MC value
    mc_nom = h_mc.integrate('region', f'cr_2e_j_{region}').values()[()]

    data_over_mc_nom = data_vals / mc_nom

    # Store the up and down variations in dicts
    data_over_mc_up = {}
    data_over_mc_down = {}

    # Up and down variations for MC
    for variation in variations:
        # Handle JER variations for T1 MET --> Symmetrize
        if not smear and variation == 'jer':
            mc_up = h_mc.integrate('region', f'cr_2e_j_{region}_{variation}Up').values()[()]
            
            # Symmetric up and down values for JER
            data_over_mc_up[variation] = data_vals / mc_up
            data_over_mc_down[variation] = data_over_mc_nom - (data_over_mc_up[variation] - data_over_mc_nom)
        else:
            mc_up = h_mc.integrate('region', f'cr_2e_j_{region}_{variation}Up').values()[()]
            mc_down = h_mc.integrate('region', f'cr_2e_j_{region}_{variation}Down').values()[()]
    
            # Up and down data / MC ratios
            data_over_mc_up[variation] = data_vals / mc_up
            data_over_mc_down[variation] = data_vals / mc_down

    # Now, combine the variations
    total_up_v = 0
    total_down_v = 0
    for variation in variations:
        up_v = data_over_mc_up[variation]
        down_v = data_over_mc_down[variation]

        # Percent variation w.r.t. nominal ratio
        percent_up_v = up_v / data_over_mc_nom - 1
        percent_down_v = down_v / data_over_mc_nom - 1

        total_up_v += np.sqrt(percent_up_v**2)
        total_down_v += np.sqrt(percent_down_v**2)

    return total_up_v, total_down_v

def data_mc_comparison_plot(acc, outtag, 
            distribution='met', 
            year=2017, 
            smear=False, 
            region='norecoil', 
            ratio_with_sf=False, 
            only_plot_with_sf=False,
            variations=None):
    '''For the given distribution and year, construct data/MC comparison plot with all the JME variations'''
    # Do quick internal check
    if only_plot_with_sf and not ratio_with_sf:
        raise RuntimeError('only_plot_with_sf option only works if ratio_with_sf is set to True.')

    # By default, the list of variations contains all three
    if variations is None:
        variations = ['jesTotal', 'jer', 'unclustEn']

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
    
    # Calculate data / MC SF
    if ratio_with_sf:
        sf = get_data_mc_sf_from_integral(
            h_data.integrate('dataset'), 
            h_mc_nom.integrate('dataset')
            )
        # In-place scaling
        h_mc_scaled = copy.deepcopy(h_mc_nom)
        h_mc_scaled.scale(sf)

    # Use the scaled histogram or the other one
    h_mc = h_mc_scaled if only_plot_with_sf else h_mc_nom

    hist.plot1d(h_mc, ax=ax, overlay='dataset', stack=True, clear=False)

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
    ax.legend(ncol=1, prop={'size':12.})

    fig.text(0., 1., region_labels[region],
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
            )

    # Plot the ratio of nominal data/MC values
    hist.plotratio(h_data.integrate('dataset'), h_mc.integrate('dataset'),
            ax=rax,
            guide_opts={},
            unc='num',
            error_opts=data_err_opts
            )

    rax.grid(True)
    rax.set_ylim(0,2)

    loc1 = matplotlib.ticker.MultipleLocator(base=0.5)
    loc2 = matplotlib.ticker.MultipleLocator(base=0.1)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)
    
    # Now, for each variation, get the varied ratios and plot them in the ratio pad
    data_mc_nom = h_data.integrate('dataset').values()[()] / h_mc_nom.integrate('dataset').values()[()]
    edges = h_data.integrate('dataset').axes()[0].edges()
    centers = h_data.integrate('dataset').axes()[0].centers()

    # Plot several combinations of the three variations we have
    variations_to_plot = [
        variations,
        variations[:2],
        variations[:1]
    ]

    for variations in variations_to_plot:
        h_mc_var = h.integrate('dataset', re.compile(f'(?!(EGamma)).*{year}'))
        combined_var_up, combined_var_down = get_combined_variations(h_data, h_mc_var, variations, region, smear)

        up_var = np.r_[combined_var_up, combined_var_up[-1]]
        down_var = np.r_[combined_var_down, combined_var_down[-1]]

        rax.fill_between(edges, 
                1+up_var, 
                1-down_var,
                step='post',
                label=get_variation_label(variations),
                alpha=0.5
                )

    rax.legend(prop={'size':10.}, ncol=3)

    if ratio_with_sf and not only_plot_with_sf:
        data_err_opts['color'] = 'red'

        h_mc = h_mc_nom.integrate('dataset')
        h_mc.scale(sf)

        hist.plotratio(h_data.integrate('dataset'), h_mc,
                ax=rax,
                guide_opts={},
                unc='num',
                error_opts=data_err_opts,
                clear=False
                )

        plt.text(1., 1., f'Data/MC SF: {sf:.3f}',
                fontsize=14,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
               )

    rax.set_xlabel(get_x_label(distribution, smeared=smear))
    rax.set_ylabel('Data / MC')

    # Save figure
    if smear:
        outdir = f'./output/{outtag}/data_mc/smeared/{region}'
    else:
        outdir = f'./output/{outtag}/data_mc/not_smeared/{region}'
    
    if only_plot_with_sf:
        outdir = pjoin(outdir, 'scaled')
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'data_mc_comp_{distribution}_{year}.pdf')
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
                if not re.match(args.distribution, distribution):
                    continue
                data_mc_comparison_plot(acc, outtag,
                        distribution=distribution,
                        year=year,
                        smear=args.smeared,
                        region=region,
                        ratio_with_sf=True,
                        only_plot_with_sf=args.plot_with_sf,
                        variations=args.variations
                        )

if __name__ == '__main__':
    main()