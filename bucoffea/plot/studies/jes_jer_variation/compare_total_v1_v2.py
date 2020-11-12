#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt
from bucoffea.plot.util import fig_ratio
from coffea import hist
from pprint import pprint

pjoin = os.path.join

def make_comparison_plot(computed_totals, read_totals, edges, outdir, year):
    '''Wrapper function to handle making of comparison plots.'''
    fig, ax, rax = fig_ratio()

    centers = 0.5*((edges + np.roll(edges, -1)) )[:-1]

    ax.plot(centers, read_totals['Up'], label='jesTotalUp', marker='*')
    ax.plot(centers, read_totals['Down'], label='jesTotalDown', marker='*')

    ax.plot(centers, computed_totals['v1']['Up'], label='v1 Up', marker='o')
    ax.plot(centers, computed_totals['v1']['Down'], label='v1 Down', marker='o')
    ax.plot(centers, computed_totals['v2']['Up'], label='v2 Up', marker='o')
    ax.plot(centers, computed_totals['v2']['Down'], label='v2 Down', marker='o')

    ax.legend()

    # TODO: Compute and plot ratios here!

    outpath = pjoin(outdir, f'total_jes_comp_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def compare_total(files, year):
    '''Given the v1 and v2 uncertainty files, compute and compare the total uncertainties with jesTotal.'''
    # Read the 11-split nuisances and the total nuisances
    split_nuisances = [s.decode('utf-8') for s in files['v1'].keys() if re.match(f'vbf_{year}_(?!(jer|jesTotal)).*', s.decode('utf-8'))]
    total_nuisances = [s.decode('utf-8') for s in files['v1'].keys() if re.match(f'vbf_{year}_jesTotal.*', s.decode('utf-8'))]

    mjj_edges = files['v1'][total_nuisances[0]].edges

    computed_totals = {
        'v1' : {'Up' : 0, 'Down' : 0},
        'v2' : {'Up' : 0, 'Down' : 0},
    }

    for nuis in split_nuisances:
        nuisv1 = files['v1'][nuis].values - 1
        nuisv2 = files['v2'][nuis].values - 1

        print(nuis)

        if 'Up' in nuis:
            computed_totals['v1']['Up'] += nuisv1 ** 2
            computed_totals['v2']['Up'] += nuisv2 ** 2
        if 'Down' in nuis:
            computed_totals['v1']['Down'] += nuisv1 ** 2
            computed_totals['v2']['Down'] += nuisv2 ** 2

    computed_totals['v1']['Up'] = 1 + np.sqrt(computed_totals['v1']['Up'])
    computed_totals['v1']['Down'] = 1 - np.sqrt(computed_totals['v1']['Down'])
    computed_totals['v2']['Up'] = 1 + np.sqrt(computed_totals['v2']['Up'])
    computed_totals['v2']['Down'] = 1 - np.sqrt(computed_totals['v2']['Down'])

    # Now, read the "jesTotal" nuisances, compare against manually computed total uncertainty
    read_totals = {}

    for nuis in total_nuisances:
        if 'Up' in nuis:
            percent_diff = files['v1'][nuis].values - 1
            read_totals['Up'] = 1 + np.sign(percent_diff) * np.abs(percent_diff)
        elif 'Down' in nuis:
            percent_diff = files['v1'][nuis].values - 1
            read_totals['Down'] = 1 + np.sign(percent_diff) * np.abs(percent_diff)

    # Make a comparison plot
    outdir = f'./output/jes_comparison_v1_v2/jes_total_comp'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    make_comparison_plot(computed_totals, read_totals, mjj_edges, outdir, year)

def main():
    # Path to files containing the JES uncertainties for v1 and v2 (non-smoothed ones)
    v1_shape_path = './output/merged_2020-09-26_vbfhinv_splitJECuncs_25Aug20/splitJEC/vbf/root/vbf_shape_jes_uncs_normal.root'
    v2_shape_path = './output/merged_2020-10-30_vbfhinv_splitJECuncs_25Aug20_v2/splitJEC/vbf/root/vbf_shape_jes_uncs_normal.root'

    files = {
        'v1': uproot.open(v1_shape_path),
        'v2': uproot.open(v2_shape_path)
    }

    for year in [2017]:
        compare_total(files, year=year)

if  __name__ == '__main__':
    main()