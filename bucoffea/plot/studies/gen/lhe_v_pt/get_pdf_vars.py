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

pjoin = os.path.join

def hessian_unc(nom, var):
    '''Calculate PDF uncertainty for a Hessian set.'''
    unc=np.zeros_like(nom) 
    for variation in var.values():
        unc += (nom - variation)**2
    return np.sqrt(unc)

def mc_unc(nom, var):
    '''Calculate PDF uncertainty for a MC set.'''
    # Calculate the average of all variations
    var_avg = np.zeros_like(nom)
    for variation in var.values():
        var_avg += variation
    var_avg /= len(var)
    # Calculate the MC uncertainty
    unc = np.zeros_like(nom)
    for variation in var.values():
        unc += (nom-var_avg)**2
    return np.sqrt(unc/(len(var)-1))

def calculate_pdf_unc(nom, var, tag):
    '''Given the nominal and varied weight content,
       calculate the PDF uncertainty.'''
    # Use PDF uncertainties for Hessian sets 
    # if samples is a DY or W sample
    if tag in ['wjet', 'dy']:
        unc = hessian_unc(nom, var)
    elif tag == 'gjets':
        unc = mc_unc(nom, var) 
    # Return percent uncertainty
    return unc/nom 
    
def get_pdf_uncertainty(acc, regex, tag):
    '''Given the input accumulator, calculate the
       PDF uncertainty from all PDF variations.'''
    # Define rebinning
    vpt_ax_fine = list(range(0,400,40)) + list(range(400,1200,80))
    if tag in ['wjet', 'dy']:
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_fine)
        mjj_ax = hist.Bin('mjj','M(jj) (GeV)',[0,200]+list(range(500,2500,500)))
    elif tag in ['gjets']:
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_fine)
        mjj_ax = hist.Bin('mjj','M(jj) (GeV)',[0,200,500,1000,1500,2000])

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

    # Integrate out mjj to get 1D variations 
    # as a function of V-pt
    mjj_slice = slice(200,2000)
    h = h.integrate('mjj', mjj_slice, overflow='over')

    # Get NLO distribution
    nlo = h[re.compile('.*(LHE|amcat).*')].integrate('dataset')

    # Nominal: NLO with no PDF variation
    nlo_nom = nlo.integrate('var', 'nominal').values(overflow='over')[()]

    # NLO with PDF variations
    # Use a dict to collect NLO contents with all PDF variations
    nlo_var = {}

    for var in nlo.identifiers('var'):
        var_name = var.name
        if 'pdf' not in var_name: 
            continue
        # Patch for problem in DY samples:
        # extra PDF variations were added
        # Just take the usual 33 PDF variations for now.
        if tag != 'gjets' and var_name in ['pdf_100', 'pdf_101', 'pdf_102']: continue
        if tag != 'gjets' and var_name == 'pdf_33': break
        nlo_var[var_name] = nlo.integrate('var', var_name).values(overflow='over')[()]
    print(nlo_var.keys())

    unc = calculate_pdf_unc(nlo_nom, nlo_var, tag)
    print(unc)

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

    get_pdf_uncertainty(acc, regex='WNJetsToLNu.*', tag='wjet')
    get_pdf_uncertainty(acc, regex='DYNJetsToLL.*', tag='dy')
    get_pdf_uncertainty(acc, regex='G1Jet.*', tag='gjets')

if __name__ == '__main__':
    main()

