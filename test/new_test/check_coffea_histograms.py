#!/usr/bin/env python

from coffea.util import load
from coffea import hist
import sys
import os
import argparse
from pprint import pprint
from termcolor import colored

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The input coffea file.')
    parser.add_argument('--analysis', help='monojet or vbf, default=vbf', default='vbf')
    args = parser.parse_args()
    return args

def test_histogram_names(acc):
    '''Given the input accumulator, check whether specific histograms are there.'''
    histograms_to_search = ('mjj', 'met', 'recoil', 'gen_vpt', 'gen_mjj', 'ak4_pt', 'ak4_eta', 'detajj', 'dphijj')
    print(colored('='*20, 'red'))
    print(colored('CHECKING HISTOGRAMS', 'red'))
    print(colored('='*20, 'red'))

    # Check that each histogram is there 
    for histo_name in histograms_to_search:
        assert histo_name in acc.keys()
        print(f'Found {histo_name}')

    print(colored('-'*20, 'green'))
    print(colored('HISTOGRAM CHECK: DONE', 'green'))

def test_regions(acc, analysis='vbf'):
    '''Given the input accumulator, check whether all the regions are present (SR + 5 CRs).'''
    region_suffix = '_j' if analysis == 'monojet' else '_vbf'
    regions_to_search = [f'sr{region_suffix}', f'cr_1m{region_suffix}', f'cr_1e{region_suffix}', f'cr_2m{region_suffix}', f'cr_2e{region_suffix}', f'cr_g{region_suffix}']
    print(colored('='*20, 'red'))
    print(colored('CHECKING REGIONS', 'red'))
    print(colored('='*20, 'red'))

    # Check that each region is present in recoil distribution
    h = acc['recoil']
    region_names = []
    for region in h.identifiers('region'):
        region_names.append(region.name)
    
    for region in regions_to_search:
        assert region in region_names
        print(f'Found {region}')

    print(colored('-'*20, 'green'))
    print(colored('REGION CHECK: DONE', 'green'))

def test_binnings(acc):
    '''For several variables, check the binning of these (see whether they can be rebinned to the requested scheme later)'''
    recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]
    mjj_bins = list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]
    variables_to_check = ['recoil', 'mjj']

    print(colored('='*20, 'red'))
    print(colored(f'CHECKING BINNINGS: {", ".join(variables_to_check)}', 'red'))
    print(colored('='*20, 'red'))

    new_binnings = {
        'recoil' : hist.Bin('recoil', 'Recoil (GeV)', recoil_bins_2016),
        'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', mjj_bins)
    }

    for variable in variables_to_check:
        h = acc[variable]
        print(f'Checking binning: {variable}')
        # Do the rebinning and see if coffea complains
        try:
            h = h.rebin(variable, new_binnings[variable])
        except:
            raise AssertionError(f'The binning for {variable} is not consistent!')

    print(colored('-'*20, 'green'))
    print(colored('BINNING CHECK: DONE', 'green'))

def test_cuts(acc):
    '''For a given region (SR or CR), test whether all the cuts are applied.'''
    print(colored('='*20, 'red'))
    print(colored('CHECKING CUTFLOWS', 'red'))
    print(colored('='*20, 'red'))

    common_cuts = [
        'veto_ele',
        'veto_muo',
        'filt_met',
        'mindphijr',
        'recoil',
        'two_jets',
        'leadak4_pt_eta',
        'leadak4_id',
        'trailak4_pt_eta',
        'trailak4_id',
        'hemisphere',
        'mjj',
        'dphijj',
        'detajj',
        'veto_photon',
        'veto_tau',
        'veto_b',
    ]
    # The cuts that should be there
    cuts = {}
    cuts['sr_vbf'] = ['trig_met','metphihemextveto','hornveto'] + common_cuts + ['dpfcalo_sr']

    cr_2m_cuts = ['trig_met','two_muons', 'at_least_one_tight_mu', 'dimuon_mass', 'veto_ele', 'dimuon_charge'] + common_cuts[1:] + ['dpfcalo_cr']
    cr_2m_cuts.remove('veto_muo')
    cuts['cr_2m_vbf'] = cr_2m_cuts

    cr_1m_cuts = ['trig_met','one_muon', 'at_least_one_tight_mu',  'veto_ele'] + common_cuts[1:] + ['dpfcalo_cr']
    cr_1m_cuts.remove('veto_muo')
    cuts['cr_1m_vbf'] = cr_1m_cuts

    cr_2e_cuts = ['trig_ele','two_electrons', 'at_least_one_tight_el', 'dielectron_mass', 'veto_muo', 'dielectron_charge'] + common_cuts[2:] + ['dpfcalo_cr']
    # cr_2e_cuts.remove('veto_ele')
    cuts['cr_2e_vbf'] = cr_2e_cuts

    cr_1e_cuts = ['trig_ele','one_electron', 'at_least_one_tight_el', 'veto_muo','met_el'] + common_cuts[1:] + ['dpfcalo_cr','no_el_in_hem']
    # cr_1e_cuts.remove('veto_ele')
    cuts['cr_1e_vbf'] =  cr_1e_cuts

    cr_g_cuts = ['trig_photon', 'one_photon', 'at_least_one_tight_photon','photon_pt'] + common_cuts + ['dpfcalo_cr']
    cr_g_cuts.remove('veto_photon')
    cuts['cr_g_vbf'] = cr_g_cuts

    # Now check that these cuts exist in the respective regions
    regions = ['sr_vbf', 'cr_1m_vbf', 'cr_2m_vbf', 'cr_1e_vbf', 'cr_2e_vbf', 'cr_g_vbf']
    for region in regions:
        print('-'*20)
        print(f'Checking region: {region}')
        print('-'*20)
        # Get the cutflow
        cutflow = list(acc[f'cutflow_{region}'].values())[0]
        necessary_cuts = cuts[region]

        for cut in necessary_cuts:
            assert cut in cutflow.keys()
            print(f'Cut exists: {cut}' + u' \u2713') 

    print(colored('-'*20, 'green'))
    print(colored('CUT CHECK: DONE', 'green'))

# def test_uncs(acc, analysis='vbf'):
    # '''Given the input accumulator, check whether all the uncertainties on met/mjj are present.'''
    # Get the histogram where uncertainties are stored
    # h = acc['mjj_unc']
    # uncs_to_search = [
        # 'unc_zoverw_nlo_muf_down', 'unc_zoverw_nlo_muf_up',
        # 'unc_zoverw_nlo_mur_down', 'unc_zoverw_nlo_mur_up',
        # 'unc_zoverw_nlo_pdf_down', 'unc_zoverw_nlo_pdf_up',
        # 'unc_goverz_nlo_muf_down', 'unc_goverz_nlo_muf_up',
        # 'unc_goverz_nlo_mur_down', 'unc_goverz_nlo_mur_up',
        # 'unc_goverz_nlo_pdf_down', 'unc_goverz_nlo_pdf_up',
        # 'unc_w_ewkcorr_overz_common_up', 'unc_w_ewkcorr_overz_common_down'
    # ]
# 
    # for unc in uncs_to_search:
        # assert unc in h.identifiers('uncertainty')

def main():
    args = parse_cli()
    infile = args.infile
    analysis = args.analysis

    acc = load(infile)

    # Do a number of checks to the coffea file
    test_histogram_names(acc)
    test_regions(acc)
    test_binnings(acc)
    test_cuts(acc)
    # test_uncs(acc)

if __name__ == '__main__':
    main()