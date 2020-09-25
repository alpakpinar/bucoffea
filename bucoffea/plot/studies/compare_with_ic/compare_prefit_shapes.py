#!/usr/bin/env python

import os
import sys
import re
import uproot
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

# Supress division warnings
np.seterr(divide='ignore', invalid='ignore')

def get_input_files():
    '''Get the relevant input fit diagnostics files from both sides for pre-fit shape comparison.'''
    inputdir = './inputs/fit_diagnostics'
    ic_file = pjoin(inputdir, 'IC_fitDiagnosticsCRonlyFit.root')
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
    'ewkzll' : 'EWKZll',
    'qcd_wjets': 'WJETS',
    'qcdzll' : 'DY',
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
    'ewkzll' : r'EWK $Z(\ell\ell)$ {}',
    'qcd_wjets': r'QCD $W(\ell\nu)$ {}',
    'qcdzll' : 'DY {}',
    'ewk_zjets': r'EWK $Z(\nu\nu)$ {}',
    'qcd_zjets' : r'QCD $Z(\nu\nu)$ {}',
    'top' : 'Top {}',
    'qcd' : 'QCD {}',
    'diboson' : 'Diboson {}',
    'data' : 'Data {}',
    'total_signal' : 'Total Signal {}'
}

def compare_prefit_shapes(ic_file, bu_file):
    '''Compare pre-fit shapes as a function of mjj.'''
    f_bu = uproot.open(bu_file)['shapes_prefit']
    f_ic = uproot.open(ic_file)['shapes_prefit']

    regions = [regionname.decode('utf-8').replace(';1', '') for regionname in f_bu.keys()]
    for region in regions:
        if 'photon' in region:
            continue

        print('*'*20)
        print(f'Region: {region}')
        print('*'*20)

        year = re.findall('2017|2018', region)[0]

        # Output directory to save the plots
        outdir = f'./output/prefit_shape_comparison/{region}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for process, h_bu in f_bu[region].items():
            process = process.decode('utf-8').replace(';1', '')
            if not process in ic_processes.keys() or process == 'data':
                continue
            print(f'Process: {process}')
            h_ic = f_ic[ic_regions[region] ][ic_processes[process] ]

            # Binning must be the same between the two
            assert((h_ic.edges == h_bu.edges).all())
            edges = h_ic.edges
            
            ic_vals = h_ic.values
            bu_vals = h_bu.values
    
            # Plot comparison of values
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
            hep.histplot(ic_vals, edges, ax=ax, label='IC')
            hep.histplot(bu_vals, edges, ax=ax, label='BU')
    
            ax.legend()
            ax.set_yscale('log')
            ax.set_ylim(1e-2, 1e5)
            ax.set_title(titles[process].format(year))

            # Plot ratio
            ratio = bu_vals / ic_vals
            centers = ( (edges + np.roll(edges,-1))/2 )[:-1]
    
            rax.plot(centers, ratio, marker='o', ls='', color='black')
    
            rax.grid(True)
            rax.set_ylim(0.8,1.2)
            rax.set_ylabel('BU / IC')
            rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    
            outpath = pjoin(outdir, f'{process}_{year}_comp.pdf')
            fig.savefig(outpath)
            print(f'MSG% File saved: {outpath}')
    
            plt.close(fig)

def main():
    ic_file, bu_file = get_input_files()

    compare_prefit_shapes(ic_file, bu_file)

if __name__ == '__main__':
    main()

