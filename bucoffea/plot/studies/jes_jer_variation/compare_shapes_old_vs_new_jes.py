#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import mplhep as hep

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from pprint import pprint

pjoin = os.path.join

np.seterr(divide='ignore', invalid='ignore')

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.
}

def preprocess(h,acc):
    '''Pre-process a given histogram.'''
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    mjj_bin = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
    h = h.rebin('mjj', mjj_bin)

    return h

def calculate_poisson_err(rsumw, sumw2_var, sumw_nom):
    '''
    Calculate the Poisson error on varied / nominal ratio.
    Returns an interval containing the up / down variations on errors.
    '''
    err = np.abs(
        hist.poisson_interval(rsumw, sumw2_var / sumw_nom**2) - rsumw
    )
    return err

def integrate_over_dataset(h, year, dataset_tag):
    '''For a given histogram, integrate over the relevant dataset and region and return the new histogram.'''
    tag_to_dataset_regex = {
        f'ggh_{year}' : f'GluGlu.*HiggspTgt190.*{year}', # stat-enriched ggH
        f'vbf_{year}' : f'VBF.*{year}',
        f'qcd_zjets_{year}' : f'ZJetsToNuNu.*{year}',
        f'ewk_zjets_{year}' : f'EWKZ.*{year}',
    }

    dataset_to_integrate = tag_to_dataset_regex[dataset_tag]

    h = h.integrate('dataset', re.compile(dataset_to_integrate) )

    return h

def make_comparison_plots(acc_dict, dataset_tag, year, variation='jesRelativeBal'):
    '''For the given variation, make a comparison plot for the shapes of the variation as a function of mjj.'''
    histos = {}
    for v, acc in acc_dict.items():
        acc.load('mjj')
        htemp = preprocess(acc['mjj'], acc)
        # Integrate over the dataset
        histos[v] = integrate_over_dataset(htemp, year, dataset_tag)

    # Get mjj edges
    edges = histos['v1'].axis('mjj').edges()
    centers = histos['v1'].axis('mjj').centers()

    # Now, calculate the nominal yields and the variations for the two versions
    nom_sumw, nom_sumw2 = {}, {}
    var_sumw, var_sumw2 = {}, {}
    for v, h in histos.items():
        # Nominal yields: Just the regular SR
        nom_sumw[v], nom_sumw2[v] = h.integrate('region', 'sr_vbf').values(sumw2=True)[()]
        var_sumw[v], var_sumw2[v] = {}, {}
        for region in h.identifiers('region'):
            if variation not in region.name:
                continue
            # Compute and store the up/down variations for this specific variation
            var_direction = re.findall('Up|Down', region.name)[0]
            var_sumw[v][var_direction], var_sumw2[v][var_direction] = h.integrate('region', region).values(sumw2=True)[()]

    # Compute the varied / nominal ratios and plot them
    fig, ax, rax = fig_ratio()
    ratios_up, err_up = {}, {}
    ratios_down, err_down = {}, {}
    for v in histos.keys():
        sumw_nom  = nom_sumw[v]
        sumw_up   = var_sumw[v]['Up']
        sumw_down = var_sumw[v]['Down']
        
        ratios_up[v]   = sumw_up / sumw_nom
        ratios_down[v] = sumw_down / sumw_nom

        # Calculate the errors in the ratios
        err_up[v] = calculate_poisson_err(ratios_up[v], var_sumw2[v]['Up'], sumw_nom)
        err_down[v] = calculate_poisson_err(ratios_down[v], var_sumw2[v]['Down'], sumw_nom)

        hep.histplot(ratios_up[v], edges, yerr=err_up[v], ax=ax, label=f'Up {v}', histtype='errorbar')
        hep.histplot(ratios_down[v], edges, yerr=err_down[v], ax=ax, label=f'Down {v}', histtype='errorbar')

    ax.set_ylabel('Variation / Nominal')
    ax.set_ylim(0.6,1.4)
    ax.grid(True)
    ax.legend()

    ax.set_title(f'Variation: {variation}')

    # Plot ratio of variations between the two versions
    dratio_up = ratios_up['v2'] / ratios_up['v1']
    dratio_down = ratios_down['v2'] / ratios_down['v1']

    # The errors on the ratio, scaled by denominator
    ratio_up_err = err_up['v2'] / ratios_up['v1']
    ratio_down_err = err_down['v2'] / ratios_down['v1']

    rax.errorbar(centers, dratio_up, yerr=ratio_up_err, label='Up', **data_err_opts)
    rax.errorbar(centers, dratio_down, yerr=ratio_down_err, label='Down', **data_err_opts)

    rax.set_ylabel('v2 / v1')
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.grid(True)
    if not ('EC2' in variation or 'FlavorQCD' in variation):
        rax.set_ylim(0.4,1.6)
    else:
        rax.set_ylim(0.8,1.2)
    rax.legend(ncol=2)

    loc1 = MultipleLocator(0.2)
    loc2 = MultipleLocator(0.1)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)

    # Save figure
    outdir = './output/jes_comparison_v1_v2'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{dataset_tag}_{variation}_jes_comp.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    '''Script to compare JES/JER uncertainty shapes between old and new split JES source files.'''
    # Paths to merged coffea files with the regular (old) JES src file and the new one
    path_to_acc_v1 = bucoffea_path('submission/merged_2020-09-26_vbfhinv_splitJECuncs_25Aug20')
    path_to_acc_v2 = bucoffea_path('submission/merged_2020-10-30_vbfhinv_splitJECuncs_25Aug20_v2')
    acc_dict = {
        'v1' : dir_archive(path_to_acc_v1), 
        'v2' : dir_archive(path_to_acc_v2),
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')
    
    for year in [2017, 2018]:
        # List of all split JES variations (also check "Total" for testing)
        variations = [  'jesFlavorQCD', 
                        'jesRelativeBal',
                        'jesHF',
                        'jesBBEC1',
                        'jesEC2',
                        'jesAbsolute',
                        f'jesBBEC1_{year}',
                        f'jesEC2_{year}',
                        f'jesAbsolute_{year}',
                        f'jesHF_{year}',
                        f'jesRelativeSample_{year}',
                        'jesTotal'
                        ]
        
        for dataset_tag in [f'qcd_zjets_{year}']:
            for variation in variations:
                make_comparison_plots(acc_dict, 
                        dataset_tag=dataset_tag,
                        year=year,
                        variation=variation
                        )
            
if __name__ == '__main__':
    main()
