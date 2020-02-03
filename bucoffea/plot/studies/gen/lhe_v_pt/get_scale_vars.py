#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from bucoffea.helpers.paths import bucoffea_path
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt

def get_old_kfac(tag):
    '''Given the dataset tag, get the nominal 2D VBF k-factors.'''
    f = uproot.open(bucoffea_path('data/sf/theory/2017_gen_v_pt_qcd_sf.root'))
    return f[f'2d_{tag}_vbf'].values

def get_scale_variations(acc, regex, tag, scale_var, outputrootfile, old_kfac):
    '''Calculate the new k-factors with a scale weight variation.
       Dumps the ratio: 
       --- New k-factors with variation / Old (nominal) k-factors
       into the given output ROOT file.'''

    print(f'Working on: {tag}, {scale_var}')

    # Define rebinning
    if tag in ['wjet', 'dy']:
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)',[0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640, 760, 880,1200])
        mjj_ax = hist.Bin('mjj','M(jj) (GeV)',list(range(0,2500,500)))
    elif tag in ['gjets']:
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)',[0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640])
        mjj_ax = hist.Bin('mjj','M(jj) (GeV)',[0,200,500,1000,1500])

    # Set the correct pt type
    pt_tag = 'combined' if tag != 'gjets' else 'stat1'
    acc.load(f'gen_vpt_vbf_{pt_tag}')
    h = acc[f'gen_vpt_vbf_{pt_tag}']

    h = h.rebin('vpt', vpt_ax)
    h = h.rebin('mjj', mjj_ax)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    h = h[re.compile(regex)]

    lo = h[re.compile('.*HT.*')].integrate('dataset')
    nlo = h[re.compile('.*(LHE|amcat).*')].integrate('dataset')
    
    xaxis = lo.axes()[0]
    yaxis = lo.axes()[1]

    # Print choose the relevant scale variation (relevant to NLO only)
    # For LO, choose the nominal (i.e. no variation)
    lo = lo.integrate('var', 'nominal')
    nlo_var = nlo.integrate('var', scale_var)
    
    sumw_lo, sumw2_lo = lo.values(overflow='over', sumw2=True)[()]
    sumw_nlo_var, sumw2_nlo_var = nlo_var.values(overflow='over', sumw2=True)[()]

    # Scale factor with variation
    sf_var = sumw_nlo_var / sumw_lo
    
    # Calculate the scale factor without variation
    nlo_nom = nlo.integrate('var', 'nominal')
    sumw_nlo_nom, sumw2_nlo_nom = nlo_nom.values(overflow='over', sumw2=True)[()]

    sf_nom = sumw_nlo_nom / sumw_lo 

    # Calculate the ratio of the new k-factors
    # with variation and old nominal k-factors
    ratio = sf_var / sf_nom 

    tup = (ratio, xaxis.edges(overflow='over'), yaxis.edges(overflow='over'))
 
    # Save to the ROOT file
    outputrootfile[f'2d_{tag}_vbf_{scale_var}'] = tup
    
def main():
    inpath = sys.argv[1]

    acc = dir_archive(
                       inpath,
                       serialized=True,
                       compression=0,
                       memsize=1e3
                     )

    acc.load('sumw')
    acc.load('sumw2')
    
    outputrootfile = uproot.recreate('vbf_sf_scale_var.root')

    scale_var_dict = {
        'gjets' : [
            'scale_1', #mu_R = 0.5 & mu_F = 1.0
            'scale_3', #mu_R = 1.0 & mu_F = 0.5
            'scale_4', #mu_R = 1.0 & mu_F = 1.0 (nominal)
            'scale_5', #mu_R = 1.0 & mu_F = 2.0
            'scale_7', #mu_R = 2.0 & mu_F = 1.0
                ],
        'wjet/dy' : [
            'scale_1', #mu_R = 0.5 & mu_F = 1.0
            'scale_3', #mu_R = 1.0 & mu_F = 0.5
            'scale_4', #mu_R = 1.0 & mu_F = 2.0
            'scale_6', #mu_R = 2.0 & mu_F = 1.0    
                ]
                
    }

    tag_regex = {
        'wjet'  : 'WN?JetsToLNu.*',
        'dy'    : 'DYN?JetsToLL.*',
        'gjets' : 'G\d?Jet.*' 
    }
    
    for tag,regex in tag_regex.items():
        # Get the nominal VBF k-factors
        old_kfac = get_old_kfac(tag)

        scale_var_list = scale_var_dict['gjets'] if tag == 'gjets' else scale_var_dict['wjet/dy']

        for scale_var in scale_var_list:
            get_scale_variations(acc=acc,
                                 regex=regex,
                                 tag=tag,
                                 scale_var=scale_var,
                                 outputrootfile=outputrootfile,
                                 old_kfac=old_kfac
                                 )

if __name__ == '__main__':
    main()
