#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import mplhep as hep
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt

pjoin = os.path.join

# Supress division warnings
np.seterr(divide='ignore', invalid='ignore')

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', help='The tag for the input files.')
    parser.add_argument('--years', help='The year to look at, the default is both 2017 and 2018.', type=int, nargs='*', default=[2017, 2018])
    parser.add_argument('--fit', help='cr_only (for CR-only fit) or sr_cr_fit (SR+CR fit, default)', default='sr_cr_fit')
    parser.add_argument('--compare', help='List of modules/functions to run, default is running both "shapes", "ratios".', nargs='*', default=['shapes', 'ratios'])
    args = parser.parse_args()
    return args

def get_input_files(tag, year, fit='sr_cr_fit'):
    '''Get the relevant input fit diagnostics files from both sides for pre-fit shape comparison.'''
    # Directory where the input root files are located
    inputdir = f'./inputs/fit_diagnostics/{tag}/{fit}'
    ic_file = pjoin(inputdir, f'IC_fitDiagnosticsMTR_{year}.root')
    bu_file = pjoin(inputdir, f'BU_fitDiagnosticsMTR_{year}.root')

    return ic_file, bu_file

# Region names in IC fit diagnostics files
ic_regions = {
    'vbf_2017_dielec' : 'MTR_2017_ZEE', 
    'vbf_2018_dielec' : 'MTR_2018_ZEE', 
    'vbf_2017_dimuon' : 'MTR_2017_ZMUMU', 
    'vbf_2018_dimuon' : 'MTR_2018_ZMUMU', 
    'vbf_2017_signal' : 'MTR_2017_SR', 
    'vbf_2018_signal' : 'MTR_2018_SR', 
    'vbf_2017_singleel' : 'MTR_2017_WENU', 
    'vbf_2018_singleel' : 'MTR_2018_WENU', 
    'vbf_2017_singlemu' : 'MTR_2017_WMUNU',
    'vbf_2018_singlemu' : 'MTR_2018_WMUNU'
}

# Process names in IC fit diagnostics files
ic_processes = {
    'total_background' : 'total_background',
    'ewk_wjets' : 'EWKW',
    'ewk_zll' : 'EWKZll',
    'qcd_wjets': 'WJETS',
    'qcd_zll' : 'DY',
    'ewk_zjets': 'EWKZNUNU',
    'qcd_zjets' : 'ZJETS',
    'top' : 'TOP',
    'qcd' : 'QCD',
    'diboson' : 'VV',
    'data' : 'data',
    'total_signal' : 'total_signal'
}

titles = {
    'total_background' : 'Total Background {}',
    'ewk_wjets' : r'EWK $W(\ell\nu)$ {}',
    'ewk_zll' : r'EWK $Z(\ell\ell)$ {}',
    'qcd_wjets': r'QCD $W(\ell\nu)$ {}',
    'qcd_zll' : 'DY {}',
    'ewk_zjets': r'EWK $Z(\nu\nu)$ {}',
    'qcd_zjets' : r'QCD $Z(\nu\nu)$ {}',
    'top' : 'Top {}',
    'qcd' : 'QCD {}',
    'diboson' : 'Diboson {}',
    'data' : 'Data {}',
    'total_signal' : 'Total Signal {}'
}

def mjj_bins():
    return [200., 400., 600., 900., 1200., 1500.,
            2000., 2750., 3500., 5000.]

def get_title_for_ratio(ratio_tag):
    year = int(ratio_tag.split('_')[-1])
    proc = ratio_tag.split('_')[0]

    mapping = {
        f'{proc}_wlnu_over_znunu_{year}' : r'{PROC} $W(\ell\nu) \ / \ Z(\nu\nu)$ {YEAR}'.format(PROC=proc.upper(), YEAR=year),
        f'{proc}_zmumu_over_znunu_{year}' : r'{PROC} $Z(\mu\mu) \ / \ Z(\nu\nu)$ {YEAR}'.format(PROC=proc.upper(), YEAR=year),
        f'{proc}_zee_over_znunu_{year}' : r'{PROC} $Z(ee) \ / \ Z(\nu\nu)$ {YEAR}'.format(PROC=proc.upper(), YEAR=year),
        f'{proc}_wmunu_over_wlnu_{year}' : r'{PROC} $W(\mu\nu) \ / \ W(\ell\nu)$ {YEAR}'.format(PROC=proc.upper(), YEAR=year),
        f'{proc}_wenu_over_wlnu_{year}' : r'{PROC} $W(e\nu) \ / \ W(\ell\nu)$ {YEAR}'.format(PROC=proc.upper(), YEAR=year),
    }

    return mapping[ratio_tag]

def get_histograms_for_ratios(f_ic, f_bu, ratio_tag):
    '''For each type of ratio, extract the regions and processes for numerator and denominator.'''
    year = int(ratio_tag.split('_')[-1])
    proc = ratio_tag.split('_')[0]
    ratios = {
        f'{proc}_wlnu_over_znunu_{year}' : {
            'region_num' : f'vbf_{year}_signal', 
            'region_den' : f'vbf_{year}_signal', 
            'histogram_num' : f'{proc}_wjets', 
            'histogram_den' : f'{proc}_zjets'
        },
        f'{proc}_zmumu_over_znunu_{year}' : {
            'region_num' : f'vbf_{year}_dimuon', 
            'region_den' : f'vbf_{year}_signal', 
            'histogram_num' : f'{proc}_zll', 
            'histogram_den' : f'{proc}_zjets'
        },
        f'{proc}_zee_over_znunu_{year}' : {
            'region_num' : f'vbf_{year}_dielec', 
            'region_den' : f'vbf_{year}_signal', 
            'histogram_num' : f'{proc}_zll', 
            'histogram_den' : f'{proc}_zjets'
        },
        f'{proc}_wmunu_over_wlnu_{year}' : {
            'region_num' : f'vbf_{year}_singlemu', 
            'region_den' : f'vbf_{year}_signal', 
            'histogram_num' : f'{proc}_wjets', 
            'histogram_den' : f'{proc}_wjets'
        },
        f'{proc}_wenu_over_wlnu_{year}' : {
            'region_num' : f'vbf_{year}_singleel', 
            'region_den' : f'vbf_{year}_signal', 
            'histogram_num' : f'{proc}_wjets', 
            'histogram_den' : f'{proc}_wjets'
        },
    }

    hist_info = ratios[ratio_tag]

    # First, histograms from BU files
    bu_hist_num = f_bu[ hist_info['region_num'] ][ hist_info['histogram_num'] ]
    bu_hist_den = f_bu[ hist_info['region_den'] ][ hist_info['histogram_den'] ]

    # Second, IC files
    ic_hist_num = f_ic[ ic_regions[hist_info['region_num']] ][ ic_processes[hist_info['histogram_num']] ]
    ic_hist_den = f_ic[ ic_regions[hist_info['region_den']] ][ ic_processes[hist_info['histogram_den']] ]

    return bu_hist_num, bu_hist_den, ic_hist_num, ic_hist_den

def compare_ratios(ic_file, bu_file, tag, year):
    '''Compare several ratios between pre-fit templates, as a function of mjj.'''
    f_bu = uproot.open(bu_file)['shapes_prefit']
    f_ic = uproot.open(ic_file)['shapes_prefit']

    # Output directory for the plots to be saved
    outdir = f'./output/prefit_shape_comparison/{tag}/ratios'
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    for proc in ['qcd', 'ewk']:
        ratio_tags = [
            f'{proc}_wlnu_over_znunu_{year}',
            f'{proc}_zmumu_over_znunu_{year}',
            f'{proc}_zee_over_znunu_{year}',
            f'{proc}_wmunu_over_wlnu_{year}',
            f'{proc}_wenu_over_wlnu_{year}'
        ]    

        for ratio_tag in ratio_tags:
            # Read the relevant histograms
            bu_hist_num, bu_hist_den, ic_hist_num, ic_hist_den = get_histograms_for_ratios(f_ic, f_bu, ratio_tag)

            # Sanity check!
            assert ( (bu_hist_num.edges == ic_hist_num.edges).all() )
            edges = bu_hist_num.edges

            # Calculate ratios for both sides
            ratio_bu = bu_hist_num.values / bu_hist_den.values
            ratio_ic = ic_hist_num.values / ic_hist_den.values

            # Plot comparison
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
            hep.histplot(ratio_ic, edges, ax=ax,  label='IC')
            hep.histplot(ratio_bu, edges, ax=ax,  label='BU')

            ax.legend()
            ax.set_title( get_title_for_ratio(ratio_tag) )

            dratio = ratio_bu / ratio_ic
            centers = ( (edges + np.roll(edges,-1))/2 )[:-1]

            rax.plot(centers, dratio, marker='o', ls='', color='black')

            rax.grid(True)
            rax.set_ylim(0.94,1.06)
            rax.set_ylabel('BU / IC')
            rax.set_xlabel(r'$M_{jj} \ (GeV)$')

            loc = matplotlib.ticker.MultipleLocator(base=0.02)
            rax.yaxis.set_major_locator(loc)

            xlim = rax.get_xlim()
            rax.plot(xlim, [1., 1.], color='red')
            rax.set_xlim(xlim)

            outpath = pjoin(outdir, f'{ratio_tag}.pdf')
            fig.savefig(outpath)
            print(f'MSG% File saved: {outpath}')
    
            plt.close(fig)
    
def compare_prefit_shapes(ic_file, bu_file, tag, year):
    '''Compare pre-fit shapes as a function of mjj.'''
    f_bu = uproot.open(bu_file)['shapes_prefit']
    f_ic = uproot.open(ic_file)['shapes_prefit']

    regions = [regionname.decode('utf-8').replace(';1', '') for regionname in f_bu.keys()]
    for region in regions:
        if 'photon' in region:
            continue

        region_year = int(re.findall('2017|2018', region)[0])
        # If this year is not specified, skip this region
        if region_year != year:
            continue

        print('*'*20)
        print(f'Region: {region}')
        print('*'*20)

        # Output directory to save the plots
        outdir = f'./output/prefit_shape_comparison/{tag}/{region}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for process, h_bu in f_bu[region].items():
            process = process.decode('utf-8').replace(';1', '')
            if not process in ic_processes.keys():
                continue
            print(f'Process: {process}')
            h_ic = f_ic[ic_regions[region] ][ic_processes[process] ]

            fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

            # Binning must be the same between the two
            edges = mjj_bins()
            if process != 'data':
                assert((h_ic.edges == edges).all())
                assert((h_bu.edges == edges).all())
                
                ic_vals = h_ic.values
                bu_vals = h_bu.values
                
                # Plot comparison of values
                hep.histplot(ic_vals, edges, ax=ax, label='IC')
                hep.histplot(bu_vals, edges, ax=ax, label='BU')

            # "data" is stored as a TGraph, handle it separately
            else:
                ic_vals = h_ic.yvalues
                bu_vals = h_bu.yvalues

                hep.histplot(ic_vals, edges, label='IC', ax=ax)
                hep.histplot(bu_vals, edges, label='BU', ax=ax)
    
            ax.legend()
            ax.set_yscale('log')
            if process not in ['diboson', 'top']:
                if 'dielec' in region or 'dimu' in region:
                    ax.set_ylim(1e-4, 1e3)
                else:
                    ax.set_ylim(1e-3, 1e4)
            else:
                ax.set_ylim(1e-6, 1e3)
            ax.set_title(titles[process].format(year))

            # Plot ratio
            ratio = bu_vals / ic_vals
            centers = ( (edges + np.roll(edges,-1))/2 )[:-1]
            rax.plot(centers, ratio, marker='o', ls='', color='black')
    
            rax.grid(True)
            if process in ['top', 'diboson', 'qcd']:
                rax.set_ylim(0.8,1.2)
            else:
                rax.set_ylim(0.5,1.5)
                # loc = matplotlib.ticker.MultipleLocator(base=0.02)
                # rax.yaxis.set_major_locator(loc)

            rax.set_ylabel('BU / IC')
            rax.set_xlabel(r'$M_{jj} \ (GeV)$')

            xlim = rax.get_xlim()
            rax.plot(xlim, [1., 1.], color='red')
            rax.set_xlim(xlim)

            outpath = pjoin(outdir, f'{region}_{process}_{year}.pdf')
            fig.savefig(outpath)
            print(f'MSG% File saved: {outpath}')
    
            plt.close(fig)

def main():
    args = parse_cli()
    years = args.years
    for year in years:
        ic_file, bu_file = get_input_files(args.tag, year, args.fit)
        if 'shapes' in args.compare:
            compare_prefit_shapes(ic_file, bu_file, args.tag, year)
        if 'ratios' in args.compare:
            compare_ratios(ic_file, bu_file, args.tag, year)

if __name__ == '__main__':
    main()

