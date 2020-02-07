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

def plot_variations(nom, var, tag):
    '''For the nominal weights and each set of
       variational weights, plot the ratio nom/var
       as a histogram.'''
    var_over_nom = []
    for variation in var.values():
        for idx in range(len(variation)):
            ratio = variation[idx]/nom[idx]
            var_over_nom.append(ratio)
    
    # Plot the results
    fig, ax = plt.subplots(1,1)
    tag_to_title = {
        'dy'    : r'PDF variations: $Z\rightarrow \ell \ell$',
        'wjet'  : r'PDF variations: $W\rightarrow \ell \nu$',
        'gjets' : r'PDF variations: $\gamma$ + jets'
    }
    if tag == 'd':
        bins = np.linspace(0.2, 1.8)
    else:
        bins = np.linspace(0.9, 1.1, 20)
    ax.hist(var_over_nom, bins=bins)
    ax.set_xlabel('Var / Nom')
    ax.set_ylabel('Counts')
    title = tag_to_title[tag]
    ax.set_title(title)

    # Save figure
    outdir = './output/kfac_variations/pdf'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'{tag}_var_nom_weights.pdf')
    fig.savefig(outpath)
    print(f'Histogram saved in: {outpath}')

def hessian_unc(nom, var):
    '''Calculate PDF uncertainty for a Hessian set.'''
    unc=np.zeros_like(nom) 
    for idx,variation in enumerate(var.values()):
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
        unc += (variation-var_avg)**2
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

def get_pdf_uncertainty(acc, regex, tag, outputrootfile):
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

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    h = h[re.compile(regex)]

    # Integrate out mjj to get 1D variations 
    # as a function of V-pt
    mjj_slice = slice(200,7500)
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
        true_pdf_list = ['pdf_0', 'pdf_1', 'pdf_10', 'pdf_11', 'pdf_12', 'pdf_13', 'pdf_14', 'pdf_15', 'pdf_16', 'pdf_17', 'pdf_18', 'pdf_19', 'pdf_2', 'pdf_20', 'pdf_21', 'pdf_22', 'pdf_23', 'pdf_24', 'pdf_25', 'pdf_26', 'pdf_27', 'pdf_28', 'pdf_29', 'pdf_3', 'pdf_30', 'pdf_4', 'pdf_5', 'pdf_6', 'pdf_7', 'pdf_8', 'pdf_9']
        if tag != 'gjets' and var_name not in true_pdf_list: continue
        nlo_var[var_name] = nlo.integrate('var', var_name).values(overflow='over')[()]

    unc = calculate_pdf_unc(nlo_nom, nlo_var, tag)
    print(unc)

    plot_variations(nlo_nom, nlo_var, tag)

    # Plot the uncertainty as a function of V-pt
    fig, ax = plt.subplots(1,1)
    vpt_edges = vpt_ax.edges(overflow='over')
    vpt_centers = ((vpt_edges + np.roll(vpt_edges, -1))/2)[:-1]
    ax.plot(vpt_centers, unc, 'o')
    ax.set_xlabel(r'$p_T(V) \ (GeV)$')
    ax.set_ylabel(r'$\sigma_{pdf}$ / Nominal Counts')
    tag_to_title = {
        'dy'    : r'$Z\rightarrow \ell \ell$',
        'wjet'  : r'$W\rightarrow \ell \nu$',
        'gjets' : r'$\gamma$ + jets'
    }
    title = tag_to_title[tag]
    ax.set_title(title)
    ax.grid(True)

    if tag == 'gjets':
        ax.set_ylim(0,0.05)
    else:
        ax.set_ylim(0,0.4)

    # Save the figure
    outdir = './output/kfac_variations/pdf'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{tag}_pdf_unc.pdf')
    fig.savefig(outpath)

    # Save the uncertainties as a ROOT histogram
    outputrootfile[f'{tag}_pdf_unc'] = (unc, vpt_ax.edges(overflow='over'))

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

    # Create the output ROOT file to save the 
    # PDF uncertainties as a function of v-pt
    outputrootfile = uproot.recreate('vbf_pdf_var.root')

    get_pdf_uncertainty(acc, regex='WNJetsToLNu.*', tag='wjet', outputrootfile=outputrootfile)
    get_pdf_uncertainty(acc, regex='DYNJetsToLL.*', tag='dy', outputrootfile=outputrootfile)
    get_pdf_uncertainty(acc, regex='G1Jet.*', tag='gjets', outputrootfile=outputrootfile)

if __name__ == '__main__':
    main()

