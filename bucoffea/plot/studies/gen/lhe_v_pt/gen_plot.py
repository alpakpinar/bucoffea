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

REBIN_AX = {
    'dy/wjet': {
        'vpt' : hist.Bin('vpt', AX_LABELS['vpt'], [0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640, 760, 880,1200]),
        'mjj' : hist.Bin('mjj', AX_LABELS['mjj'], list(range(0,2500,500)))
    },
    'gjets' : {
        'vpt' : hist.Bin('vpt', AX_LABELS['vpt'], [0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640]),
        'mjj' : hist.Bin('mjj', AX_LABELS['mjj'], [0,200,500,1000,1500])
    }
}

##################
# TO BE UPDATED
##################
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

def plot_2d_gen_spectrum(acc, tag, regex, outputrootfile, pt_type='stat1', sample_order='lo'):
    '''Plot 2D gen-vpt/mjj spectrum for LO and NLO GJets samples.
    Saves the 2D histogram into a ROOT file.
    ==============
    PARAMETERS
    acc : The coffea accumulator containing all the histograms.
    tag : Dataset tag. (w/dy/gjets)
    regex : Regular expression for dataset name matching.
    outputrootfile : Name of output ROOT file where the histogram will be saved. 
    pt_type : Type of pt (stat1 or dress). 
              Default is stat1.
    sample_order : Order of the sample.
                   Should be specified as 'lo' or 'nlo'.
                   Default is 'lo'.
    ==============
    '''
    dist = f'gen_vpt_vbf_{pt_type}'
    acc.load(dist)
    histogram = acc[dist]

    if tag == 'gjets':
        _REBIN_AX = REBIN_AX['gjets']
    else:
        _REBIN_AX = REBIN_AX['dy/wjet']

    # Merge datasets/extensions, 
    # scale the histogram according to x-sec
    histogram = merge_extensions(histogram, acc, reweight_pu=False)
    scale_xs_lumi(histogram)
    histogram = merge_datasets(histogram)
    histogram = histogram[re.compile(regex)]

    histogram = histogram.integrate('jpt')
    histogram = histogram.rebin(histogram.axis('vpt'), _REBIN_AX['vpt'])
    histogram = histogram.rebin(histogram.axis('mjj'), _REBIN_AX['mjj'])

    if sample_order == 'lo':
        histogram = histogram[re.compile('.*HT.*')].integrate('dataset') 
    elif sample_order == 'nlo':
        histogram = histogram[re.compile('.*(LHE|amcat).*')].integrate('dataset') 
    else:
        raise ValueError(f'{sample_order}: Invalid argument for sample_order. Should be specified as "lo" or "nlo"')

    # Plot the 2D histogram and save the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    hist.plot2d(histogram, xaxis='mjj', ax=ax, binwnorm=True)

    ax.set_xlabel(AX_LABELS['mjj'])
    ax.set_ylabel(AX_LABELS['vpt'])
    ax.set_title(f'{tag}_{sample_order}')

    outpath = f'./output/{tag}_gen_vpt_mjj_{sample_order}.pdf'
    fig.savefig(outpath)
    print(f'Saved histogram in {outpath}')

    # Save to output ROOT file
    xaxis = histogram.axes()[0]
    yaxis = histogram.axes()[1]
    w = histogram.values(overflow='over')[()]
    tup = (w, xaxis.edges(overflow='over'), yaxis.edges(overflow='over'))
    outputrootfile[f'{tag}_{sample_order}_gen_vpt_mjj'] = tup

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

    filename = './output/2d_gen_vpt_mjj.root'
    outputrootfile = uproot.recreate(filename)

#    plot_gen_spectrum(acc, variable='vpt')
#    plot_gen_spectrum(acc, variable='mjj')

    tag_regex_pt = {
        'wjet' : ('WN?JetsToLNu.*', 'dress'),
        'dy' : ('DYN?JetsToLL.*', 'dress'),
        'gjets' : ('G\d?Jet.*', 'stat1')
    }

    for tag, regex_pt in tag_regex_pt.items():
        
        plot_2d_gen_spectrum(acc, tag=tag, regex=regex_pt[0], pt_type=regex_pt[1], sample_order='lo', outputrootfile=outputrootfile)
        plot_2d_gen_spectrum(acc, tag=tag, regex=regex_pt[0], pt_type=regex_pt[1], sample_order='nlo', outputrootfile=outputrootfile)

if __name__ == '__main__':
    main()

