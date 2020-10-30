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
from pprint import pprint

pjoin = os.path.join

np.seterr(divide='ignore', invalid='ignore')

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'elinewidth': 1,
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

def integrate_over_dataset(h, year, dataset_tag):
    '''For a given histogram, integrate over the relevant dataset and region and return the new histogram.'''
    tag_to_dataset_regex = {
        f'ggh_{year}' : f'GluGlu.*HiggspTgt190.*{year}', # stat-enriched ggH
        f'vbf_{year}' : f'VBF.*{year}',
        f'qcd_zjets_{year}' : f'ZJetsToNuNu.*{year}',
        f'ewk_zjets_{year}' : f'EWKZ.*{year}',
    }

    dataset_to_integrate = tag_to_dataset_regex[dataset_tag]
    # Test
    pprint( h[re.compile(dataset_to_integrate)].values() )

    h = h.integrate('dataset', re.compile(dataset_to_integrate) )

    return h

def make_comparison_plot(acc_dict, dataset_tag, year, variation='jesRelativeBal'):
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
    nom_yields = {}
    var_yields = {}
    for v, h in histos.items():
        # Nominal yields: Just the regular SR
        nom_yields[v] = h.integrate('region', 'sr_vbf').values()[()]
        for region in h.identifiers('region'):
            if variation not in region:
                continue
            # Compute and store the up/down variations for this specific variation
            var_yields[v] = {}
            var_direction = re.findall('Up|Down', region)[0]
            print(var_direction)
            var_yields[v][var_direction] = h.integrate('region', region).values()[()]

    # Compute the varied / nominal ratios and plot them
    fig, ax, rax = fig_ratio()
    ratios_up = {}
    ratios_down = {}
    for v in histos.keys():
        sumw_nom  = nom_yields[v]
        sumw_up   = var_yields[v]['Up']
        sumw_down = var_yields[v]['Down']
        
        ratios_up[v]   = sumw_up / sumw_nom
        ratios_down[v] = sumw_down / sumw_nom

        hep.histplot(ratios_up[v], edges, ax=ax, label=f'Up_{v}')
        hep.histplot(ratios_down[v], edges, ax=ax, label=f'Down_{v}')

    ax.set_ylabel('Variation / Nominal')
    ax.legend()

    # Plot ratio of variations between the two versions
    dratio_up = ratios_up['v2'] / ratios_up['v1']
    dratio_down = ratios_down['v2'] / ratios_down['v1']

    rax.plot(centers, dratio_up, label='Up', **data_err_opts)
    rax.plot(centers, dratio_down, label='Down', **data_err_opts)

    rax.set_ylabel('v2 / v1')
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.grid(True)
    rax.set_ylim(0.8,1.2)
    rax.legend()

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
    
    make_comparison_plot(acc_dict, 
            dataset_tag='qcd_zjets',
            year=2017,
            variation='jesRelativeBal'
            )
            
if __name__ == '__main__':
    main()
