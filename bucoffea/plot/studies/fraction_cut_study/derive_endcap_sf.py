#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint
from compare_eff import preprocess, do_coarse_rebinning_for_2d, ratio_unc, calculate_efficiency

pjoin = os.path.join

warnings.filterwarnings('ignore')

def plot_sf_for_endcap(acc, outtag, region='cr_2m', year=2017):
    '''Plot 1D SF as a function of jet pt for endcap jets.'''
    variable = 'ak4_pt0_eta0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, region, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning_for_2d(h_data)
    h_mc = do_coarse_rebinning_for_2d(h_mc)

    h_data_pos_endcap = h_data.integrate('jeteta', slice(2.5,3.0))
    h_data_neg_endcap = h_data.integrate('jeteta', slice(-3.0,-2.5))
    h_mc_pos_endcap = h_mc.integrate('jeteta', slice(2.5,3.0))
    h_mc_neg_endcap = h_mc.integrate('jeteta', slice(-3.0,-2.5))

    # Calculate efficiencies for the two endcap regions
    data_eff_pos_endcap, mc_eff_pos_endcap, data_eff_unc_pos_endcap, mc_eff_unc_pos_endcap = calculate_efficiency(h_data_pos_endcap, h_mc_pos_endcap)
    data_eff_neg_endcap, mc_eff_neg_endcap, data_eff_unc_neg_endcap, mc_eff_unc_neg_endcap = calculate_efficiency(h_data_neg_endcap, h_mc_neg_endcap)

    # Calculate SF
    sf_pos_endcap = data_eff_pos_endcap / mc_eff_pos_endcap
    sf_neg_endcap = data_eff_neg_endcap / mc_eff_neg_endcap

    # Calculate the error on SF
    sf_err_pos_endcap = ratio_unc(
        data_eff_pos_endcap,
        mc_eff_pos_endcap,
        data_eff_unc_pos_endcap,
        mc_eff_unc_pos_endcap
    )

    sf_err_neg_endcap = ratio_unc(
        data_eff_neg_endcap,
        mc_eff_neg_endcap,
        data_eff_unc_neg_endcap,
        mc_eff_unc_neg_endcap
    )

    # Guard against NaN values
    sf_pos_endcap[np.isnan(sf_pos_endcap) | np.isinf(sf_pos_endcap)] = 1.
    sf_neg_endcap[np.isnan(sf_neg_endcap) | np.isinf(sf_neg_endcap)] = 1.

    pt_ax = h_data.axis('jetpt')
    xcenters = pt_ax.centers(overflow='over')

    # Plot the 1D SF for the two regions (for now, do not plot errorbars until we figure it out)
    fig, ax = plt.subplots()
    ax.errorbar(xcenters, sf_pos_endcap, yerr=sf_err_pos_endcap, marker='o', label=r'$2.5 < \eta < 3.0$')
    ax.errorbar(xcenters, sf_neg_endcap, yerr=sf_err_neg_endcap, marker='o', label=r'$-2.5 < \eta < -3.0$')

    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel('Data/MC SF')
    ax.set_ylim(0.7,1.3)
    ax.legend()

    ax.set_title(f'Jet SFs in Endcap: {year}')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'sf_1d_{year}.pdf')

    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    for year in [2017, 2018]:
        # Calculate 1D SF as a function of jet pt for the endcap jets only
        plot_sf_for_endcap(acc, outtag, region='cr_2m', year=year)

if __name__ == '__main__':
    main()
