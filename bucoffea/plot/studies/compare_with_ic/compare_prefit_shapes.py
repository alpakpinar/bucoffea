#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

# Supress division warnings
np.seterr(divide='ignore', invalid='ignore')

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', help='The tag for the input files.')
    parser.add_argument('--years', help='The year to look at, the default is both 2017 and 2018.', type=int, nargs='*', default=[2017, 2018])
    parser.add_argument('--fit', help='cr_only (for CR-only fit) or sr_cr_fit (SR+CR fit, default)', default='sr_cr_fit')
    args = parser.parse_args()
    return args

def get_input_files(tag, year=None, fit='cr_only'):
    '''Get the relevant input fit diagnostics files from both sides for pre-fit shape comparison.'''
    if year is None and tag == '05Oct20':
        raise RuntimeError('For 05Oct20 samples, please specify a year!')

    if tag != '05Oct20':
        bu_inputdir = f'./inputs/fit_diagnostics/{tag}'
        ic_inputdir = f'./inputs/fit_diagnostics'
        ic_file = pjoin(ic_inputdir, 'IC_fitDiagnosticsCRonlyFit.root')
        bu_file = pjoin(bu_inputdir, 'BU_fitDiagnosticsCRonlyFit.root')

    else:
        inputdir = f'./inputs/fit_diagnostics/{tag}/{fit}'
        ic_file = pjoin(inputdir, f'IC_fitDiagnosticsMTR_{year}_CRonlyFit.root')
        bu_file = pjoin(inputdir, 'BU_fitDiagnosticsCRonlyFit.root')

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
            if 'dielec' in region or 'dimu' in region:
                ax.set_ylim(1e-4, 1e3)
            else:
                ax.set_ylim(1e-3, 1e4)
            ax.set_title(titles[process].format(year))

            # Plot ratio
            ratio = bu_vals / ic_vals
            centers = ( (edges + np.roll(edges,-1))/2 )[:-1]
                
            rax.plot(centers, ratio, marker='o', ls='', color='black')
    
            rax.grid(True)
            rax.set_ylim(0.8,1.2)
            rax.set_ylabel('BU / IC')
            rax.set_xlabel(r'$M_{jj} \ (GeV)$')

            outpath = pjoin(outdir, f'{region}_{process}_{year}.pdf')
            fig.savefig(outpath)
            print(f'MSG% File saved: {outpath}')
    
            plt.close(fig)

def main():
    args = parse_cli()
    years = args.years
    for year in years:
        ic_file, bu_file = get_input_files(args.tag, year, args.fit)
        compare_prefit_shapes(ic_file, bu_file, args.tag, year)

if __name__ == '__main__':
    main()

