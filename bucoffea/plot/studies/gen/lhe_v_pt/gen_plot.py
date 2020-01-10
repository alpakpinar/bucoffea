#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from bucoffea.plot.util import (merge_datasets, 
                                merge_extensions, 
                                scale_xs_lumi)
                            
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
import uproot

AX_LABELS = {
    'mjj' : r'$M_{jj}$ (GeV)',
    'vpt' : r'$p_T(V)$ (GeV)'
}

def plot_gen_spectrum(acc, tag='stat1', variable='vpt'):
    '''Plot 1D gen pt or mjj distribution for LO and NLO
    GJets samples on the same canvas.
    ===============
    PARAMETERS
    acc : Input accumulator.
    tag : Type of pt (stat1 or dress). 
          Default is stat1.
    variable : The variable to plot. 
               Should be specified as "mjj" or "vpt".
               Defualt is "vpt".
    ===============
    '''
    if variable not in ['mjj', 'vpt']:
        raise ValueError(f'{variable}: Not a valid argument for variable. Should be specified as "mjj" or "vpt"')

    # Specify the variable to integrate over
    if variable == 'vpt':
        integrate = 'mjj'
    else:
        integrate = 'vpt'

    dist = f'gen_vpt_vbf_{tag}'
    acc.load(dist)
    histogram = acc[dist]

    # Merge datasets/extensions, 
    # scale the histogram according to x-sec
    histogram = merge_extensions(histogram, acc, reweight_pu=False)
    scale_xs_lumi(histogram)
    histogram = merge_datasets(histogram)

    # LO and NLO GJets samples
    dataset = re.compile('G\d?Jet.*_(HT|Pt).*')

    histogram = histogram.integrate('jpt').integrate(f'{integrate}') 

    # Plot the histogram and save the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    hist.plot1d(histogram[dataset], ax=ax, overlay='dataset', binwnorm=True) 

    #ax.set_yscale('log')
    ax.set_ylabel('Events / Bin Width')
    ax.set_xlabel(AX_LABELS[variable])
    ax.set_title(r'LO and NLO GJets Comparison')

    outpath = f'./output/gen_{variable}.pdf'
    fig.savefig(outpath)
    print(f'Saved histogram in {outpath}')

def plot_2d_gen_spectrum(acc, tag='stat1', sample_order='lo'):
    '''Plot 2D gen-vpt/mjj spectrum for LO and NLO GJets samples.
    Saves the 2D histogram into a ROOT file.
    ==============
    PARAMETERS
    acc : The coffea accumulator containing all the histograms.
    tag : Type of pt (stat1 or dress). 
          Default is stat1.
    sample_order : Order of the sample.
                   Should be specified as 'lo' or 'nlo'.
                   Default is 'lo'.
    ==============
    '''
    if sample_order == 'lo':
        dataset_name = 'GJets_HT_MLM_2016'
    elif sample_order == 'nlo':
        dataset_name = 'G1Jet_Pt-amcatnlo_2016'
    else:
        raise ValueError(f'{sample_order}: Invalid argument for sample_order. Should be specified as "lo" or "nlo"')

    dist = f'gen_vpt_vbf_{tag}'
    acc.load(dist)
    histogram = acc[dist]

    # Merge datasets/extensions, 
    # scale the histogram according to x-sec
    histogram = merge_extensions(histogram, acc, reweight_pu=False)
    scale_xs_lumi(histogram)
    histogram = merge_datasets(histogram)
    
    histogram = histogram.integrate('dataset', dataset_name).integrate('jpt')
    
    # Plot the 2D histogram and save the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    hist.plot2d(histogram, xaxis='mjj', ax=ax, binwnorm=True)

    ax.set_xlabel(AX_LABELS['mjj'])
    ax.set_ylabel(AX_LABELS['vpt'])
    ax.set_title(f'{dataset_name}')

    outpath = f'./output/gen_vpt_mjj_{sample_order}.pdf'
    fig.savefig(outpath)
    print(f'Saved histogram in {outpath}')

    # Crate a ROOT file and save 
    # the 2D histogram
    try:
        f = uproot.open('./output/2d_gen_spectrum.root')
    except:
        f = uproot.recreate('./output/2d_gen_spectrum.root')
    xaxis = histogram.axes()[0]
    yaxis = histogram.axes()[1]
    tup = (histogram, xaxis.edges(overflow='over'), yaxis.edges(overflow='over'))
    f[f'{sample_order}_gen_vpt_mjj'] = tup

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

    plot_gen_spectrum(acc, variable='vpt')
    plot_gen_spectrum(acc, variable='mjj')
    plot_2d_gen_spectrum(acc, sample_order='lo')
    plot_2d_gen_spectrum(acc, sample_order='nlo')

if __name__ == '__main__':
    main()

