#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import uproot

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from bucoffea.plot.trigger import get_xy, ratio_unc
from bucoffea.plot.util import fig_ratio
from bucoffea.plot.style import matplotlib_rc
from matplotlib.ticker import MultipleLocator

pjoin = os.path.join

matplotlib_rc()

def get_pretty_legend_label(tag):
    pretty_legend_label = {
        'two_central_jets' : 'Two Central Jets',
        'one_jet_forward_one_jet_central' : 'Mixed',
        'inclusive_nohfhf' : 'Inclusive (No HF-HF)'
    }
    return pretty_legend_label[tag]

def check_files(fnum, fden):
    if not os.path.exists(fnum):
        raise RuntimeError(f"File not found {fnum}")
    if not os.path.exists(fden):
        raise RuntimeError(f"File not found {fden}")

def do_fit(xsf, ysf, ysferr):
    '''Fit a sigmoid function to scale factor data.'''
    def sigmoid(x,a,b):
        return 1 / (1 + np.exp(-a * (x-b)) )

    popt, _ = curve_fit(
        sigmoid,
        xsf, ysf
    )

    return sigmoid(xsf, *popt)

def fit_scale_factors(input_dir, jeteta_config, year):
    '''For the given configuration, calculate and fit the data/MC scale factor.'''
    # Figure out the input txt files
    fnum = pjoin(input_dir, f'table_1m_recoil_SingleMuon_{year}_{jeteta_config}.txt')
    fden = pjoin(input_dir, f'table_1m_recoil_WJetsToLNu_HT_MLM_{year}_{jeteta_config}.txt')

    # Check the file paths
    check_files(fnum, fden)

    xnum, xedgnum, ynum, yerrnum = get_xy(fnum)
    xden, xedgden, yden, yerrden = get_xy(fden)


    # Calculate SF + error on the SF
    xsf = xnum
    ysf = ynum / yden
    ysferr = ratio_unc(ynum, yden, yerrnum, yerrden)

    # Only get the values for recoil > 250 GeV
    mask = xsf >= 250
    xsf = xsf[mask]
    ysf = ysf[mask]
    ysferr = ysferr[:,mask]

    fig, ax, rax = fig_ratio()
    ax.errorbar(xsf, ysf, yerr=ysferr, marker='o', ls='')

    ax.set_xlabel('Recoil (GeV)')
    ax.set_ylabel('Data/MC SF')

    # Get fitted function
    f = do_fit(xsf, ysf, ysferr)
    ax.plot(xsf, f)

    fig.savefig('test.pdf')

def fit_efficiencies(fdata, fmc, jeteta_config, year, outputrootfile, outdir):
    '''Fit the efficiency with a sigmoid function.'''
    x_data, x_edg_data, y_data, yerr_data = get_xy(fdata)
    x_mc, _, y_mc, yerr_mc = get_xy(fmc)

    fig, ax, rax = fig_ratio()
    # Plot original data and MC
    ax.errorbar(x_data, y_data, yerr=yerr_data, marker='o', ls='', label='Data')
    ax.errorbar(x_mc, y_mc, yerr=yerr_mc, marker='o', ls='', label='MC')

    # Get the sigmoid fit for data and MC
    f_data = do_fit(x_data, y_data, yerr_data)
    ax.plot(x_data, f_data, lw=2, label='Data fit')

    f_mc = do_fit(x_mc, y_mc, yerr_mc)
    ax.plot(x_mc, f_mc, lw=2, label='MC fit', ls='--')

    ax.set_ylabel('Trigger Efficiency')
    ax.legend()

    # Calculate scale factors
    sf_orig = y_data / y_mc
    sf_fit = f_data / f_mc

    sf_orig_err = yerr_data / y_mc

    opts = {
        'markersize' : 12.,
        'linestyle' : 'none',
        'marker' : '.',
        'color' : 'C1'
    }

    # Plot the original and fitted SF
    rax.errorbar(x_data, sf_orig, yerr=sf_orig_err, label='Data/MC', **opts)
    rax.plot(x_data, sf_fit, label='Data/MC Fit Ratio', color='k', lw=2)

    rax.set_xlabel('Recoil (GeV)')
    rax.set_ylabel('Data / MC SF')
    rax.grid(True)
    rax.set_ylim(0.95,1.05)
    rax.legend(prop={'size': 10.})

    loc1 = MultipleLocator(0.05)
    loc2 = MultipleLocator(0.01)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)

    ylim = rax.get_ylim()
    rax.plot([250., 250.], ylim, color='red', lw=2)
    rax.set_ylim(ylim)

    plt.text(0., 1., f'{get_pretty_legend_label(jeteta_config)} {year}',
        fontsize=16,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes
        )

    outpath = pjoin(outdir, f'eff_fit_{jeteta_config}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

    x_edges = np.array(list(x_edg_data[0]) + [x_edg_data[1][-1]])

    outputrootfile[f'sf_{jeteta_config}_{year}'] = (sf_fit, x_edges)

def main():
    # Input directory to read txt files from
    input_dir = './output/120pfht_mu_recoil/merged_2020-10-20_vbfhinv_03Sep20v7_trigger_study'
    outtag = input_dir.split('/')[-1]

    jeteta_configs = [
        'two_central_jets',
        'one_jet_forward_one_jet_central',
        'inclusive_nohfhf'
    ]

    # Save the SFs to an output ROOT file
    outdir = f'./output/120pfht_mu_recoil/{outtag}/sigmoid_fit'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outputrootpath = pjoin(outdir, 'fitted_sf.root')
    outputrootfile = uproot.recreate(outputrootpath)

    for year in [2017,2018]:
        for jeteta_config in jeteta_configs:
            # Figure out the input txt files
            f_data = pjoin(input_dir, f'table_1m_recoil_SingleMuon_{year}_{jeteta_config}.txt')
            f_mc = pjoin(input_dir, f'table_1m_recoil_WJetsToLNu_HT_MLM_{year}_{jeteta_config}.txt')
        
            # Check the file paths
            check_files(f_data, f_mc)

            fit_efficiencies(f_data, f_mc, 
                jeteta_config=jeteta_config, 
                year=year, 
                outdir=outdir,
                outputrootfile=outputrootfile)

if __name__ == '__main__':
    main()