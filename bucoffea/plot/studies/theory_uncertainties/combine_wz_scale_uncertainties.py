#!/usr/bin/env python

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
import os
pjoin = os.path.join

def from_coffea(inpath, outfile):

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
    mjj_ax = hist.Bin('mjj', r'$M_{jj}$ (GeV)', [200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000])
    for distribution in ['mjj','mjj_unc']:
        acc.load(distribution)
        acc[distribution] = merge_extensions(
                                            acc[distribution],
                                            acc, 
                                            reweight_pu=not ('nopu' in distribution)
                                            )
        scale_xs_lumi(acc[distribution])
        acc[distribution] = merge_datasets(acc[distribution])
        acc[distribution] = acc[distribution].rebin(acc[distribution].axis('mjj'), mjj_ax)

    pprint(acc[distribution].axis('dataset').identifiers())
    f = uproot.recreate(outfile)

    for year in [2017,2018]:
        # QCD V
        h_z = acc['mjj'][re.compile(f'ZJetsToNuNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        f[f'z_qcd_mjj_nominal_{year}'] = export1d(h_z)

        h_w = acc['mjj'][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        f[f'w_qcd_mjj_nominal_{year}'] = export1d(h_w)

        # h_ph = acc['mjj'][re.compile(f'GJets_DR-0p4.*HT.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        # f[f'gjets_qcd_mjj_nominal_{year}'] = export1d(h_ph)

        # Scale variations for QCD Z 
        h_z_unc = acc['mjj_unc'][re.compile(f'ZJ.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        for unc in map(str, h_z_unc.axis('uncertainty').identifiers()):
            if 'denom_varied' in unc:
                continue
            h = h_z_unc.integrate(h_z_unc.axis('uncertainty'), unc)
            f[f'z_qcd_mjj_{unc}_{year}'] = export1d(h)

        # Scale variations for QCD W 
        h_w_unc = acc['mjj_unc'][re.compile(f'WJ.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        for unc in map(str, h_w_unc.axis('uncertainty').identifiers()):
            if 'num_varied' in unc:
                continue
            h = h_w_unc.integrate(h_w_unc.axis('uncertainty'), unc)
            f[f'w_qcd_mjj_{unc}_{year}'] = export1d(h)

def make_ratios(infile):
    f = r.TFile(infile)
    of = r.TFile(infile.replace('.root','_ratio.root'),'RECREATE')
    of.cd()

    for year in [2017, 2018]:
        # Get nominal Z / W ratio
        nominal_z = f.Get(f'z_qcd_mjj_nominal_{year}') 
        nominal_w = f.Get(f'w_qcd_mjj_nominal_{year}')
        nominal_ratio = nominal_z.Clone(f'zoverw_qcd_mjj_nominal_{year}')
        nominal_ratio.Divide(nominal_w)
        nominal_ratio.SetDirectory(of)    
        nominal_ratio.Write()

        # Variations in Z / W ratio
        for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
            name_for_clone = '_'.join(name.split('_')[2:])
            # Variations of Z
            if name.startswith('z_qcd_mjj_unc') and f'{year}' in name:
                ratio = f.Get(name).Clone(f'zoverw_qcd_{name_for_clone}')
                ratio.Divide(nominal_w)
                ratio.SetDirectory(of)
                ratio.Write()
            # Variations of W
            elif name.startswith('w_qcd_mjj_unc') and f'{year}' in name:
                ratio = nominal_z.Clone(f'zoverw_qcd_{name_for_clone}')
                ratio.Divide(f.Get(name))
                ratio.SetDirectory(of)
                ratio.Write()
    of.Close()
    return str(of.GetName())

def make_uncertainties(infile):
    f = r.TFile(infile)
    of = r.TFile(infile.replace('_ratio','_ratio_unc'),'RECREATE')
    of.cd()

    for year in [2017, 2018]:
        # Get double ratios first: (Z/W) varied / (Z/W) nominal
        nominal_ratio = f.Get(f'zoverw_qcd_mjj_nominal_{year}')
        # Get varied ratios, seperately for Z and W
        for name in map(lambda x:x.GetName(), f.GetListOfKeys()):
            if 'nominal' in name or f'{year}' not in name:
                continue
            ratio = f.Get(name).Clone(f'unc_{name}')
            ratio.Divide(nominal_ratio)
            ratio.SetDirectory(of)
            ratio.Write()
    
    of.Close()
    return str(of.GetName())

def make_combined_uncertanties(infile):
    f = r.TFile(infile)
    of = r.TFile(infile.replace('_ratio_unc','_ratio_unc_combined'),'RECREATE')
    of.cd()

    # Combine uncertainties on Z/W ratio:
    # Consider opposite scale variations of Z and W
    # --> Z mu_r up & W mu_r down etc.

    var_pairs = [
        ('num_varied_nlo_muf_up', 'denom_varied_nlo_muf_down'),
        ('num_varied_nlo_muf_down', 'denom_varied_nlo_muf_up'),
        ('num_varied_nlo_mur_up', 'denom_varied_nlo_mur_down'),
        ('num_varied_nlo_mur_down', 'denom_varied_nlo_mur_up')
    ]

    for year in [2017, 2018]:
        for var_pair in var_pairs:
            z_variated_name = f'unc_zoverw_qcd_mjj_unc_zoverw_{var_pair[0]}_{year}'
            w_variated_name = f'unc_zoverw_qcd_mjj_unc_zoverw_{var_pair[1]}_{year}'
            
            z_variated = f.Get(z_variated_name).Clone()
            w_variated = f.Get(w_variated_name).Clone()
            combined_zoverw = f.Get(z_variated_name).Clone(f'combined_unc_zoverw_{var_pair[0]}_{var_pair[1]}_{year}')

            # Combine the two uncertainties and save it into the new root file
            for i in range(z_variated.GetNbinsX()):
                z_bin = z_variated.GetBinContent(i+1)
                w_bin = w_variated.GetBinContent(i+1)

                combined = np.sqrt((1-z_bin)**2 + (1-w_bin)**2)
                combined = 1 + np.sign(1-z_bin) * combined
                combined_zoverw.SetBinContent(i+1, combined)

            combined_zoverw.SetDirectory(of)
            combined_zoverw.Write()

def main():
    inpath = sys.argv[1] 

    # Get the output tag for output directory naming
    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]
    
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = pjoin(outdir, f'vbf_z_w_ind_theory_unc.root')
    from_coffea(inpath, outfile)
    outfile = make_ratios(outfile)
    outfile = make_uncertainties(outfile)
    make_combined_uncertanties(outfile)

if __name__ == "__main__":
    main()
