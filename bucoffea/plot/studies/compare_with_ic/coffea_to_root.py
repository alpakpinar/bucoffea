#!/usr/bin/env python

import os
import sys
import re
import uproot
import warnings
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

def get_region_tag_for_bu(region):
    region_to_tag = {
        'SR' : 'sr_vbf',
        'Wenu' : 'cr_1e_vbf',
        'Wmunu' : 'cr_1m_vbf',
        'Zee' : 'cr_2e_vbf',
        'Zmumu' : 'cr_2m_vbf'
    }
    return region_to_tag[region]

def get_processes_and_regex(region, year):
    '''Get list of processes to look at and the dataset regex for the given region.'''
    region_to_procs = {
        'SR' : [
            {'name' : 'data_obs', 'regex' : f'MET_{year}'},
            {'name' : 'VBFHtoInv', 'regex' : re.compile(f'VBF_HToInv.*{year}')},
            {'name' : 'DY', 'regex' : re.compile(f'DYJetsToLL.*{year}')},
            {'name' : 'EWKW', 'regex' : re.compile(f'EWKW.*{year}')},
            {'name' : 'EWKZll', 'regex' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}')},
            {'name' : 'EWKZNUNU', 'regex' : re.compile(f'EWKZ2Jets.*ZToNuNu.*{year}')},
            {'name' : 'TOP', 'regex' : f'Top_FXFX_{year}'},
            {'name' : 'VV', 'regex' : f'Diboson_{year}'},
            {'name' : 'WJETS', 'regex' : re.compile(f'WJetsToLNu.*{year}')},
            {'name' : 'ZJETS', 'regex' : re.compile(f'ZJetsToNuNu.*{year}')}
        ],
        'Wenu' : [
            {'name' : 'data_obs', 'regex' : f'EGamma_{year}'},
            {'name' : 'EWKW', 'regex' : re.compile(f'EWKW.*{year}')},
            {'name' : 'WJETS', 'regex' : re.compile(f'WJetsToLNu.*{year}')}
        ],
        'Wmunu' : [
            {'name' : 'data_obs', 'regex' : f'MET_{year}'},
            {'name' : 'EWKW', 'regex' : re.compile(f'EWKW.*{year}')},
            {'name' : 'WJETS', 'regex' : re.compile(f'WJetsToLNu.*{year}')}
        ],
        'Zee' : [
            {'name' : 'data_obs', 'regex' : f'EGamma_{year}'},
            {'name' : 'EWKZll', 'regex' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}')},
            {'name' : 'ZJETS', 'regex' : re.compile(f'ZJetsToNuNu.*{year}')}
        ],
        'Zmumu' : [
            {'name' : 'data_obs', 'regex' : f'MET_{year}'},
            {'name' : 'EWKZll', 'regex' : re.compile(f'EWKZ2Jets.*ZToLL.*{year}')},
            {'name' : 'ZJETS', 'regex' : re.compile(f'ZJetsToNuNu.*{year}')}
        ],
    }

    return region_to_procs[region]

# Convert existing coffea files to root histograms and save them
def convert_to_root_for_sync(acc, tag):
    outfilename = f'./inputs/sync/{tag}/Histos_Nominal_ZJETS_2017_BU.root'
    outfile = uproot.recreate(outfilename)

    variables = ['ak4_eta0', 'ak4_eta1', 'mjj', 'mjj_nowgt', 'recoil', 'ak4_pt0', 'ak4_pt1']
    for var in variables:
        print(f'Variable: {var}')
        try:
            acc.load(var)
            h = acc[var]
        except KeyError:
            print(f'WARNING: Could not find variable {var}, skipping')
            continue

        h = merge_extensions(h, acc, reweight_pu=False)
        scale_xs_lumi(h)
        h = merge_datasets(h)

        h_znunu = h.integrate('dataset', f'ZJetsToNuNu_HT_2017').integrate('region', 'sr_vbf')
        th1 = hist.export1d(h_znunu)

        outfile[var] = th1

def save_prefit_shapes_to_root(acc, region, year=2017, jer=False):
    '''Take the prefit histograms stored in coffea files on BU side, save them into ROOT files for comparison with IC.'''
    acc.load('mjj')
    h = acc['mjj']

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    mjj_bins = hist.Bin('mjj', r'$M_{jj} \ (GeV)$',[200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000])
    h = h.rebin('mjj', mjj_bins)

    # Create output ROOT file
    outfilename = f'./inputs/prefit_shapes/BU_VBF_shapes_{region}.root'
    outfile = uproot.recreate(outfilename)

    histos = {}
    # Retrieve the histograms for the specified region
    region_tag = get_region_tag_for_bu(region)
    process_and_regex_list = get_processes_and_regex(region, year)
    
    for process_and_regex in process_and_regex_list:
        histname = process_and_regex['name']
        dataset_regex = process_and_regex['regex']
        histos[histname] = h.integrate('dataset', dataset_regex).integrate('region', region_tag)

    # Now, save the retrieved histograms into a root file
    for histname, histo in histos.items():
        _th1 = hist.export1d(histo)
        outfile[histname] = _th1

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    # Run this only for sync purposes 
    # tag = '23Sep20'
    # convert_to_root_for_sync(acc, tag)

    regions = ['SR', 'Wenu', 'Wmunu', 'Zee', 'Zmumu']
    for region in regions:
        save_prefit_shapes_to_root(acc, region=region)

if __name__ == '__main__':
    main()
