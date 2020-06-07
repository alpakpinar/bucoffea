#!/usr/bin/env python

from coffea.util import load
import sys
import argparse
from pprint import pprint

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The input coffea file.')
    # parser.add_argument('--analysis', help='monojet or vbf, default=vbf', default='vbf')
    args = parser.parse_args()
    return args

def check_histogram_names(acc, analysis='vbf'):
    '''Given the input accumulator, check whether specific histograms are there.'''
    histograms_to_search = {
        'monojet' : ('met', 'recoil', 'gen_vpt', 'gen_mjj', 'ak4_pt', 'ak4_eta'),
        'vbf' : ('mjj', 'met', 'recoil', 'gen_vpt', 'gen_mjj', 'ak4_pt', 'ak4_eta')
    }

    # Check that each histogram is there 
    for histo_name in histograms_to_search[analysis]:
        assert histo_name in acc.keys()

def check_regions(acc, analysis='vbf'):
    '''Given the input accumulator, check whether all the regions are present (SR + 5 CRs).'''
    region_suffix = '_j' if analysis == 'monojet' else '_vbf'
    regions_to_search = [f'sr{region_suffix}', f'cr_1m{region_suffix}', f'cr_1e{region_suffix}', f'cr_2m{region_suffix}', f'cr_2e{region_suffix}', f'cr_g{region_suffix}']

    # Check that each region is present in recoil distribution
    h = acc['recoil']
    for region in regions_to_search:
        assert region in h.identifiers('region')

def check_uncs(acc, analysis='vbf'):
    '''Given the input accumulator, check whether all the uncertainties on met/mjj are present.'''
    # Get the histogram where uncertainties are stored
    h = acc['mjj_unc']
    uncs_to_search = [
        'unc_zoverw_nlo_muf_down', 'unc_zoverw_nlo_muf_up',
        'unc_zoverw_nlo_mur_down', 'unc_zoverw_nlo_mur_up',
        'unc_zoverw_nlo_pdf_down', 'unc_zoverw_nlo_pdf_up',
        'unc_goverz_nlo_muf_down', 'unc_goverz_nlo_muf_up',
        'unc_goverz_nlo_mur_down', 'unc_goverz_nlo_mur_up',
        'unc_goverz_nlo_pdf_down', 'unc_goverz_nlo_pdf_up',
        'unc_w_ewkcorr_overz_common_up', 'unc_w_ewkcorr_overz_common_down'
    ]

    for unc in uncs_to_search:
        assert unc in h.identifiers('uncertainty')

def main():
    args = parse_cli()
    infile = args.infile
    # analysis = args.analysis

    acc = load(infile)

    # Do a number of checks to the coffea file
    check_histogram_names(acc)
    check_regions(acc)
    check_uncs(unc)

if __name__ == '__main__':
    main()