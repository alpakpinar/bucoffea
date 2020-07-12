#!/usr/bin/env python

from coffea.util import load
from coffea.hist.plot import plot1d
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from bucoffea.plot.util import scale_xs_lumi, merge_extensions, merge_datasets
import uproot
import re
import sys
import numpy as np
from coffea.hist.export import export1d
from coffea import hist
import ROOT as r
from pprint import pprint
import argparse

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Path containing merged coffea files.')
    parser.add_argument('--zwonly', help='Only run over Z and W.', action='store_true')
    parser.add_argument('--qcdonly', help='Only run over QCD samples.', action='store_true')
    parser.add_argument('--variable', help='The variable to derive the uncertainties as a function of, defaults to mjj.', default='mjj')
    args = parser.parse_args()
    return args

def from_coffea(inpath, outfile, run_z_w_only=False, run_qcd_only=False, variable='mjj'):

    acc = dir_archive(
                        inpath,
                        serialized=True,
                        compression=0,
                        memsize=1e3,
                        )

    # Merging, scaling, etc
    acc.load('sumw')
    acc.load('sumw_pileup')
    acc.load('nevents')
    # Axis definitions with new binnings
    new_axes = {
        'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', [200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000]),
        'ak4_eta0' : hist.Bin('jeteta', r'Leading Jet $\eta$', list(range(-5,6)))
    }

    if variable == 'mjj':
        distributions = ['mjj','mjj_unc', 'mjj_noewk']
    elif variable == 'ak4_eta0':
        distributions = ['ak4_eta0', 'ak4_eta0_unc']

    for distribution in distributions:
        acc.load(distribution)
        acc[distribution] = merge_extensions(
                                            acc[distribution],
                                            acc, 
                                            reweight_pu=not ('nopu' in distribution)
                                            )
        scale_xs_lumi(acc[distribution])
        acc[distribution] = merge_datasets(acc[distribution])
        if variable == 'mjj':
            acc[distribution] = acc[distribution].rebin(acc[distribution].axis('mjj'), new_axes['mjj'])
        elif variable == 'ak4_eta0':
            acc[distribution] = acc[distribution].rebin(acc[distribution].axis('jeteta'), new_axes['ak4_eta0'])

    pprint(acc[distribution].axis('dataset').identifiers())
    f = uproot.recreate(outfile)
    for year in [2017,2018]:
        # QCD V
        # h_z = acc['mjj'][re.compile(f'ZJetsToNuNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        h_z_mumu = acc[variable][re.compile(f'DYJetsToLL.*HT.*{year}')].integrate('region', 'cr_2m_vbf').integrate('dataset')
        h_z_ee = acc[variable][re.compile(f'DYJetsToLL.*HT.*{year}')].integrate('region', 'cr_2e_vbf').integrate('dataset')
        f[f'zmumu_qcd_{variable}_nominal_{year}'] = export1d(h_z_mumu)
        f[f'zee_qcd_{variable}_nominal_{year}'] = export1d(h_z_ee)

        h_w_munu = acc[variable][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'cr_1m_vbf').integrate('dataset')
        h_w_enu = acc[variable][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'cr_1e_vbf').integrate('dataset')
        f[f'wmunu_qcd_{variable}_nominal_{year}'] = export1d(h_w_munu)
        f[f'wenu_qcd_{variable}_nominal_{year}'] = export1d(h_w_enu)

        # print(f[f'wmunu_qcd_{variable}_nominal_{year}'] + f[f'wenu_qcd_{variable}_nominal_{year}'])

        if not run_z_w_only:
            h_ph = acc[variable][re.compile(f'GJets_DR-0p4.*HT.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
            f[f'gjets_qcd_{variable}_nominal_{year}'] = export1d(h_ph)

        # Scale + PDF variations for QCD Z(mumu)
        h_z_mumu_unc = acc[f'{variable}_unc'][re.compile(f'DYJetsToLL.*HT.*{year}')].integrate('region', 'cr_2m_vbf').integrate('dataset')
        # pprint(h_z_mumu_unc.axis('uncertainty').identifiers())
        for unc in map(str, h_z_mumu_unc.axis('uncertainty').identifiers()):
            if not re.match('.*zover(w|g).*', unc):
                continue
            # pprint(h_z_mumu_unc.identifiers('uncertainty'))
            h = h_z_mumu_unc.integrate(h_z_mumu_unc.axis('uncertainty'), unc)
            f[f'zmumu_qcd_{variable}_{unc}_{year}'] = export1d(h)
            
        # Scale + PDF variations for QCD Z(ee)
        h_z_ee_unc = acc[f'{variable}_unc'][re.compile(f'DYJetsToLL.*HT.*{year}')].integrate('region', 'cr_2e_vbf').integrate('dataset')
        for unc in map(str, h_z_ee_unc.axis('uncertainty').identifiers()):
            if not re.match('.*zover(w|g).*', unc):
                continue
            h = h_z_ee_unc.integrate(h_z_ee_unc.axis('uncertainty'), unc)
            f[f'zee_qcd_{variable}_{unc}_{year}'] = export1d(h)

        # Scale + PDF variations for QCD W(munu)
        h_w_munu_unc = acc[f'{variable}_unc'][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'cr_1m_vbf').integrate('dataset')
        for unc in map(str, h_w_munu_unc.axis('uncertainty').identifiers()):
            if not re.match('.*woverg.*', unc):
                continue
            # pprint(h_z_mumu_unc.identifiers('uncertainty'))
            h = h_w_munu_unc.integrate(h_w_munu_unc.axis('uncertainty'), unc)
            f[f'wmunu_qcd_{variable}_{unc}_{year}'] = export1d(h)
            
        # Scale + PDF variations for QCD W(enu)
        h_w_enu_unc = acc[f'{variable}_unc'][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'cr_1e_vbf').integrate('dataset')
        for unc in map(str, h_w_enu_unc.axis('uncertainty').identifiers()):
            if not re.match('.*woverg.*', unc):
                continue
            # pprint(h_z_mumu_unc.identifiers('uncertainty'))
            h = h_w_enu_unc.integrate(h_w_enu_unc.axis('uncertainty'), unc)
            f[f'wenu_qcd_{variable}_{unc}_{year}'] = export1d(h)
            
        # Scale + PDF variations for QCD photons
        # if not run_z_w_only:
            # h_ph_unc = acc[f'{variable}_unc'][re.compile(f'GJets_DR-0p4.*HT.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
            # for unc in map(str, h_ph_unc.axis('uncertainty').identifiers()):
                # if re.match('.*(zoverw|woverg|zoverg|ewkcorr).*', unc):
                    # continue
                # h = h_ph_unc.integrate(h_ph_unc.axis('uncertainty'), unc)
                # f[f'gjets_qcd_{variable}_{unc}_{year}'] = export1d(h)
    
        # EWK V
        if not run_qcd_only:
            h_z_mumu_ewk = acc[variable][re.compile(f'.*EWKZ.*ZToLL.*{year}')].integrate('region', 'cr_2m_vbf').integrate('dataset')
            h_z_ee_ewk = acc[variable][re.compile(f'.*EWKZ.*ZToLL.*{year}')].integrate('region', 'cr_2e_vbf').integrate('dataset')
            f[f'zmumu_ewk_{variable}_nominal_{year}'] = export1d(h_z_mumu_ewk)
            f[f'zee_ewk_{variable}_nominal_{year}'] = export1d(h_z_ee_ewk)
    
            h_w_munu_ewk = acc[variable][re.compile(f'.*EWKW.*{year}')].integrate('region', 'cr_1m_vbf').integrate('dataset')
            h_w_enu_ewk = acc[variable][re.compile(f'.*EWKW.*{year}')].integrate('region', 'cr_1e_vbf').integrate('dataset')
            f[f'wmunu_ewk_{variable}_nominal_{year}'] = export1d(h_w_munu_ewk)
            f[f'wenu_ewk_{variable}_nominal_{year}'] = export1d(h_w_enu_ewk)
    
            if not run_z_w_only:
                h_ph = acc[variable][re.compile(f'GJets_SM_5f_EWK.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
                f[f'gjets_ewk_{variable}_nominal_{year}'] = export1d(h_ph)
                print(h_ph.values())
    
            # Scale + PDF variations for EWK Z
            h_z_mumu_ewk_unc = acc[f'{variable}_unc'][re.compile(f'.*EWKZ.*ZToLL.*{year}')].integrate('region', 'cr_2m_vbf').integrate('dataset')
            for unc in map(str, h_z_mumu_ewk_unc.axis('uncertainty').identifiers()):
                if not re.match('.*zover(w|g).*', unc):
                    continue
                h = h_z_mumu_ewk_unc.integrate(h_z_mumu_ewk_unc.axis('uncertainty'), unc)
                f[f'zmumu_ewk_{variable}_{unc}_{year}'] = export1d(h)

            h_z_ee_ewk_unc = acc[f'{variable}_unc'][re.compile(f'.*EWKZ.*ZToLL.*{year}')].integrate('region', 'cr_2e_vbf').integrate('dataset')
            for unc in map(str, h_z_ee_ewk_unc.axis('uncertainty').identifiers()):
                if not re.match('.*zover(w|g).*', unc):
                    continue
                h = h_z_ee_ewk_unc.integrate(h_z_ee_ewk_unc.axis('uncertainty'), unc)
                f[f'zee_ewk_{variable}_{unc}_{year}'] = export1d(h)
    
            # Scale + PDF variations for EWK W
            h_w_munu_ewk_unc = acc[f'{variable}_unc'][re.compile(f'.*EWKW.*{year}')].integrate('region', 'cr_1m_vbf').integrate('dataset')
            for unc in map(str, h_w_munu_ewk_unc.axis('uncertainty').identifiers()):
                if not re.match('.*woverg.*', unc):
                    continue
                h = h_w_munu_ewk_unc.integrate(h_w_munu_ewk_unc.axis('uncertainty'), unc)
                f[f'wmunu_ewk_{variable}_{unc}_{year}'] = export1d(h)

            h_w_enu_ewk_unc = acc[f'{variable}_unc'][re.compile(f'.*EWKW.*{year}')].integrate('region', 'cr_1e_vbf').integrate('dataset')
            for unc in map(str, h_w_enu_ewk_unc.axis('uncertainty').identifiers()):
                if not re.match('.*woverg.*', unc):
                    continue
                h = h_w_enu_ewk_unc.integrate(h_w_enu_ewk_unc.axis('uncertainty'), unc)
                f[f'wenu_ewk_{variable}_{unc}_{year}'] = export1d(h)


def make_ratios(infile, run_z_w_only=False, run_qcd_only=False, variable='mjj'):
    f = r.TFile(infile)
    of = r.TFile(infile.replace('.root','_ratio.root'),'RECREATE')
    of.cd()

    sources = ['ewk', 'qcd'] if not run_qcd_only else ['qcd']

    for source in sources:
        for year in [2017,2018]:
            # Scale + PDF variations on Z(mumu) / W(munu) and Z(ee) / W(enu)
            denominator_wmunu = f.Get(f'wmunu_{source}_{variable}_nominal_{year}')
            denominator_wenu = f.Get(f'wenu_{source}_{variable}_nominal_{year}')
            denominator_gjets = f.Get(f'gjets_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                # Z(mumu) / W(munu)
                if name.startswith(f'zmumu_{source}') and ('zoverw' in name or 'nominal' in name):
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    print(name)
                    # Ratio: Z(mumu) / W(munu)
                    ratio_mu = f.Get(name).Clone(f'zmumu_over_wmunu_{source}_{variable}_{unc_tag}')
                    ratio_mu.Divide(denominator_wmunu)
                    ratio_mu.SetDirectory(of)
                    ratio_mu.Write()
                if name.startswith(f'zmumu_{source}') and ('zoverg' in name or 'nominal' in name): 
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    print(name)
                    # Ratio: Z(mumu) / GJets
                    ratio_gjets = f.Get(name).Clone(f'zmumu_over_gjets_{source}_{variable}_{unc_tag}')
                    ratio_gjets.Divide(denominator_gjets)
                    ratio_gjets.SetDirectory(of)
                    ratio_gjets.Write()
                if name.startswith(f'zee_{source}') and ('zoverw' in name or 'nominal' in name):
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    # Ratio: Z(ee) / W(enu)
                    ratio_e = f.Get(name).Clone(f'zee_over_wenu_{source}_{variable}_{unc_tag}')
                    ratio_e.Divide(denominator_wenu)
                    ratio_e.SetDirectory(of)
                    ratio_e.Write()
                if name.startswith(f'zee_{source}') and ('zoverg' in name or 'nominal' in name): 
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    # Ratio: Z(ee) / GJets
                    ratio_gjets = f.Get(name).Clone(f'zee_over_gjets_{source}_{variable}_{unc_tag}')
                    ratio_gjets.Divide(denominator_gjets)
                    ratio_gjets.SetDirectory(of)
                    ratio_gjets.Write()
                
                # Calculate W(munu) / GJets ratios
                if name.startswith(f'wmunu_{source}') and ('woverg' in name or 'nominal' in name):
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    print(name)
                    # Ratio: W(munu) / GJets
                    ratio_gjets = f.Get(name).Clone(f'wmunu_over_gjets_{source}_{variable}_{unc_tag}')
                    ratio_gjets.Divide(denominator_gjets)
                    ratio_gjets.SetDirectory(of)
                    ratio_gjets.Write()
                # Calculate W(enu) / GJets ratios
                if name.startswith(f'wenu_{source}') and ('woverg' in name or 'nominal' in name):
                    if not f"{year}" in name or 'ewkcorr' in name:
                        continue
                    if 'nominal' in name:
                        unc_tag = f'nominal_{year}'
                    else:
                        unc_tag = re.findall('unc_.*', name)[0]
                    print(name)
                    # Ratio: W(enu) / GJets
                    ratio_gjets = f.Get(name).Clone(f'wenu_over_gjets_{source}_{variable}_{unc_tag}')
                    ratio_gjets.Divide(denominator_gjets)
                    ratio_gjets.SetDirectory(of)
                    ratio_gjets.Write()
    
    of.Close()
    return str(of.GetName())

def make_uncertainties(infile, run_z_w_only=False, run_qcd_only=False, variable='mjj'):
    '''Calculate uncertainties on ratios and save into ROOT file. Combine uncs for Z and W CRs.'''
    f = r.TFile(infile)
    of = r.TFile(infile.replace('_ratio','_ratio_unc'),'RECREATE')
    of.cd()

    sources = ['ewk', 'qcd'] if not run_qcd_only else ['qcd']

    # Uncertainty in Z / W ratios (scale + PDF variations)
    for source in sources:
        for year in [2017,2018]:
            # Get the nominal ratios for each process ratio
            zee_over_wenu_nominal = f.Get(f'zee_over_wenu_{source}_{variable}_nominal_{year}')
            zmumu_over_wmunu_nominal = f.Get(f'zmumu_over_wmunu_{source}_{variable}_nominal_{year}')
            zmumu_over_gjets_nominal = f.Get(f'zmumu_over_gjets_{source}_{variable}_nominal_{year}')
            zee_over_gjets_nominal = f.Get(f'zee_over_gjets_{source}_{variable}_nominal_{year}')
            wmunu_over_gjets_nominal = f.Get(f'wmunu_over_gjets_{source}_{variable}_nominal_{year}')
            wenu_over_gjets_nominal = f.Get(f'wenu_over_gjets_{source}_{variable}_nominal_{year}')
            for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
                # Uncertainties for Z(mumu) / W(munu)
                if re.match(f'.*zmumu_over_wmunu_{source}_{variable}_unc_(.*)_{year}', name):
                    zmumu_over_wmunu_varied = f.Get(name)
                    variation = zmumu_over_wmunu_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(zmumu_over_wmunu_nominal)
                    variation.SetDirectory(of)
                    variation.Write()

                    zmumu_over_wmunu_varied.SetDirectory(of)
                    zmumu_over_wmunu_varied.Write()

                # Uncertainties for Z(ee) / W(enu)
                elif re.match(f'.*zee_over_wenu_{source}_{variable}_unc_(.*)_{year}', name):
                    zee_over_wenu_varied = f.Get(name)
                    variation = zee_over_wenu_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(zee_over_wenu_nominal)
                    variation.SetDirectory(of)
                    variation.Write()
                    
                    zee_over_wenu_varied.SetDirectory(of)
                    zee_over_wenu_varied.Write()
    
                # Uncertainties for Z(mumu) / GJets
                elif re.match(f'.*zmumu_over_gjets_{source}_{variable}_unc_(.*)_{year}', name):
                    zmumu_over_gjets_varied = f.Get(name)
                    variation = zmumu_over_gjets_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(zmumu_over_gjets_nominal)
                    variation.SetDirectory(of)
                    variation.Write()
                    
                    zmumu_over_gjets_varied.SetDirectory(of)
                    zmumu_over_gjets_varied.Write()
    
                # Uncertainties for Z(ee) / GJets
                elif re.match(f'.*zee_over_gjets_{source}_{variable}_unc_(.*)_{year}', name):
                    zee_over_gjets_varied = f.Get(name)
                    variation = zee_over_gjets_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(zee_over_gjets_nominal)
                    variation.SetDirectory(of)
                    variation.Write()
                    
                    zee_over_gjets_varied.SetDirectory(of)
                    zee_over_gjets_varied.Write()
    
                # Uncertainties for W(munu) / GJets
                elif re.match(f'.*wmunu_over_gjets_{source}_{variable}_unc_(.*)_{year}', name):
                    wmunu_over_gjets_varied = f.Get(name)
                    variation = wmunu_over_gjets_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(wmunu_over_gjets_nominal)
                    variation.SetDirectory(of)
                    variation.Write()

                    wmunu_over_gjets_varied.SetDirectory(of)
                    wmunu_over_gjets_varied.Write()
    
                # Uncertainties for W(enu) / GJets
                elif re.match(f'.*wenu_over_gjets_{source}_{variable}_unc_(.*)_{year}', name):
                    wenu_over_gjets_varied = f.Get(name)
                    variation = wenu_over_gjets_varied.Clone(f'uncertainty_{name}')

                    # Content: Varied ratio / Nominal ratio
                    variation.Divide(wenu_over_gjets_nominal)
                    variation.SetDirectory(of)
                    variation.Write()
                    
                    wenu_over_gjets_varied.SetDirectory(of)
                    wenu_over_gjets_varied.Write()

    of.Close()
    return str(of.GetName())

def combine_uncs(infile, variable):
    '''Combine the uncertainties for Z and W regions.'''
    f = r.TFile(infile)
    of = r.TFile(infile.replace('_ratio_unc', '_ratio_unc_combined'), 'RECREATE')

    sources = ['qcd', 'ewk']

    for source in sources:
        for year in [2017,2018]:
            # unc_sources = [
                # f'unc_zoverw_nlo_muf_down_{year}',
                # f'unc_zoverw_nlo_muf_up_{year}',
                # f'unc_zoverw_nlo_mur_down_{year}',
                # f'unc_zoverw_nlo_mur_up_{year}',
                # f'unc_zoverw_nlo_pdf_down_{year}',
                # f'unc_zoverw_nlo_pdf_up_{year}'
            # ]
            unc_sources = [
                f'nlo_muf_down_{year}',
                f'nlo_muf_up_{year}',
                f'nlo_mur_down_{year}',
                f'nlo_mur_up_{year}',
                f'nlo_pdf_down_{year}',
                f'nlo_pdf_up_{year}'
            ]
            # Read the uncertainties for Z(ee)/W(enu) and Z(mumu)/W(munu)
            # Calculate the combined uncertainties on Z(ll)/W(lnu)
            for unc_source in unc_sources:
                print(unc_source)
                unc_zmumu_over_wmunu = f.Get(f'uncertainty_zmumu_over_wmunu_{source}_{variable}_unc_zoverw_{unc_source}')
                unc_zee_over_wenu = f.Get(f'uncertainty_zee_over_wenu_{source}_{variable}_unc_zoverw_{unc_source}')
                unc_zmumu_over_gjets = f.Get(f'uncertainty_zmumu_over_gjets_{source}_{variable}_unc_zoverg_{unc_source}')
                unc_zee_over_gjets = f.Get(f'uncertainty_zee_over_gjets_{source}_{variable}_unc_zoverg_{unc_source}')
                unc_wmunu_over_gjets = f.Get(f'uncertainty_wmunu_over_gjets_{source}_{variable}_unc_woverg_{unc_source}')
                unc_wenu_over_gjets = f.Get(f'uncertainty_wenu_over_gjets_{source}_{variable}_unc_woverg_{unc_source}')

                # Combine in quadrature: Z / W
                combined_unc_zll_over_wlnu = f.Get(f'uncertainty_zmumu_over_wmunu_{source}_{variable}_unc_zoverw_{unc_source}').Clone(f'uncertainty_zll_over_wlnu_{source}_{variable}_unc_zoverw_{unc_source}')
                for idx in range(1, unc_zmumu_over_wmunu.GetNbinsX()+1):
                    zmumu_over_wmunu_unc = unc_zmumu_over_wmunu.GetBinContent(idx)
                    zee_over_wenu_unc = unc_zee_over_wenu.GetBinContent(idx)
                    if 'up' in unc_source:
                        combined_unc_zll_over_wlnu.SetBinContent(idx, 1+np.sqrt( (1-zmumu_over_wmunu_unc)**2 + (1-zee_over_wenu_unc)**2))
                    elif 'down' in unc_source:
                        combined_unc_zll_over_wlnu.SetBinContent(idx, 1-np.sqrt( (1-zmumu_over_wmunu_unc)**2 + (1-zee_over_wenu_unc)**2))

                # Combine in quadrature: Z / gamma
                combined_unc_zll_over_gjets = f.Get(f'uncertainty_zmumu_over_gjets_{source}_{variable}_unc_zoverg_{unc_source}').Clone(f'uncertainty_zll_over_gjets_{source}_{variable}_unc_zoverg_{unc_source}')
                for idx in range(1, unc_zmumu_over_gjets.GetNbinsX()+1):
                    zmumu_over_gjets_unc = unc_zmumu_over_gjets.GetBinContent(idx)
                    zee_over_gjets_unc = unc_zee_over_gjets.GetBinContent(idx)
                    if 'up' in unc_source:
                        combined_unc_zll_over_gjets.SetBinContent(idx, 1+np.sqrt( (1-zmumu_over_gjets_unc)**2 + (1-zee_over_gjets_unc)**2))
                    elif 'down' in unc_source:
                        combined_unc_zll_over_gjets.SetBinContent(idx, 1-np.sqrt( (1-zmumu_over_gjets_unc)**2 + (1-zee_over_gjets_unc)**2))
                            
                # Combine in quadrature: W / gamma
                combined_unc_wlnu_over_gjets = f.Get(f'uncertainty_wmunu_over_gjets_{source}_{variable}_unc_woverg_{unc_source}').Clone(f'uncertainty_wlv_over_gjets_{source}_{variable}_unc_woverg_{unc_source}')
                for idx in range(1, unc_zmumu_over_gjets.GetNbinsX()+1):
                    wmunu_over_gjets_unc = unc_wmunu_over_gjets.GetBinContent(idx)
                    wenu_over_gjets_unc = unc_wenu_over_gjets.GetBinContent(idx)
                    if 'up' in unc_source:
                        combined_unc_wlnu_over_gjets.SetBinContent(idx, 1+np.sqrt( (1-wmunu_over_gjets_unc)**2 + (1-wenu_over_gjets_unc)**2))
                    elif 'down' in unc_source:
                        combined_unc_wlnu_over_gjets.SetBinContent(idx, 1-np.sqrt( (1-wmunu_over_gjets_unc)**2 + (1-wenu_over_gjets_unc)**2))
                            
                # Save all histograms into the output ROOT file
                combined_unc_zll_over_wlnu.SetDirectory(of)
                combined_unc_zll_over_wlnu.Write()
                combined_unc_zll_over_gjets.SetDirectory(of)
                combined_unc_zll_over_gjets.Write()
                combined_unc_wlnu_over_gjets.SetDirectory(of)
                combined_unc_wlnu_over_gjets.Write()

import os
pjoin = os.path.join

def main():
    args = parse_cli()
    inpath = args.inpath 
    run_z_w_only = args.zwonly
    run_qcd_only = args.qcdonly
    variable = args.variable

    # Get the output tag for output directory naming
    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]
    
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if run_z_w_only and variable == 'mjj':
        outfile = pjoin(outdir, f'vbf_z_w_theory_unc.root')
    elif run_z_w_only and variable == 'ak4_eta0':
        outfile = pjoin(outdir, f'vbf_z_w_theory_unc_{variable}.root')
    elif not run_z_w_only and variable == 'mjj':
        outfile = pjoin(outdir, f'vbf_z_w_gjets_theory_unc.root')
    elif not run_z_w_only and variable == 'ak4_eta0':
        outfile = pjoin(outdir, f'vbf_z_w_gjets_theory_unc_{variable}.root')

    from_coffea(inpath, outfile, run_z_w_only=run_z_w_only, run_qcd_only=run_qcd_only, variable=variable)
    outfile = make_ratios(outfile, run_z_w_only=run_z_w_only, run_qcd_only=run_qcd_only, variable=variable)
    outfile = make_uncertainties(outfile, run_z_w_only=run_z_w_only, run_qcd_only=run_qcd_only, variable=variable)
    combine_uncs(outfile, variable=variable)

if __name__ == "__main__":
    main()
