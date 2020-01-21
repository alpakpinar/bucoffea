#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt

pjoin = os.path.join

recoil_bins_2016 = [ 250,  280,  310,  340,  370,  400,  430,  470,  510, 550,  590,  640,  690,  740,  790,  840,  900,  960, 1020, 1090, 1160, 1250, 1400]

REBIN = {
    'recoil' : hist.Bin('recoil', r'Recoil (GeV)', recoil_bins_2016),
    'ak4_pt' : hist.Bin('jetpt',r'All AK4 jet $p_{T}$ (GeV)',list(range(100,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt0' : hist.Bin('jetpt',r'Leading AK4 jet $p_{T}$ (GeV)',list(range(80,600,20)) + list(range(600,1000,20)) ),
    'ak4_pt1' : hist.Bin('jetpt',r'Trailing AK4 jet $p_{T}$ (GeV)',list(range(40,600,20)) + list(range(600,1000,20)) ),
    'photon_pt0' : hist.Bin('pt',r'Photon $p_{T}$ (GeV)',list(range(200,600,20)) + list(range(600,1000,20)) ),
    'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
}

def get_tag(inpath):
    '''From the submission directory name, get the tag
       for saving the output files.'''
    if inpath.endswith('/'):
        inpath = inpath[:-1]

    tag = inpath.split('/')[-1]
    
    return tag


def plot(acc, distribution, tag, use_dr_gjets_sample=False):
    '''Plot the distribution for events with
       2000 < mjj < 2750 GeV, from the given accumulator.'''
    acc.load(distribution)
    h = acc[distribution]

    # Define data and MC samples for 2017
    data = 'EGamma_2017'
    if use_dr_gjets_sample:
        mc = re.compile('(GJets_(DR|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*2017')

    else:
        mc = re.compile('(GJets_(HT|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*2017')

    # Merge extensions/datasets, scale w.r.t
    # x-section and lumi
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Choose the region (photon CR with mjj cut) and dataset
    h = h.integrate('region', 'cr_g_vbf_mjjcut')

    # Rebin the histogram if necessary 
    if distribution in REBIN.keys():
        h = h.rebin(REBIN[distribution].name, REBIN[distribution])

    fig, ax = plt.subplots(1,1,figsize=(7,5))

    data_err_options = {
        'linestyle' : 'none',
        'marker' : '.',
        'markersize' : 10.,
        'color' : 'k',
        'elinewidth' : 1
    }    

    # Plot data and MC
    hist.plot1d(h[data], ax=ax, overlay='dataset', error_opts=data_err_options)
    hist.plot1d(h[mc], ax=ax, overlay='dataset', stack=True, clear=False)
    ax.set_title(r'$2000 < m_{jj} < 2750\ GeV$')
    
    # Create output directory if it doesn't exist
    # and save the histogram as a .pdf file
    outpath = f'./output/{tag}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    filepath = pjoin(outpath, f'{distribution}_mjj2000_2750.pdf')
    fig.savefig(filepath)
    print(f'Histogram saved in {filepath}')        

def main():
    inpath = sys.argv[1]
    
    tag = get_tag(inpath)

    acc = dir_archive(
                      inpath,
                      serialized=True,
                      compression=0,
                      memsize=1e3
                    )

    acc.load('sumw')
    acc.load('sumw2')

    distributions = ['ak4_pt', 'ak4_pt0', 'ak4_pt1', 'ak4_eta', 'ak4_eta0', 'ak4_eta1', 'ak4_phi', 'recoil', 'detajj', 'dphijj', 'photon_pt0', 'photon_eta0']

    for distribution in distributions:
        plot(acc, distribution=distribution, tag=tag, use_dr_gjets_sample=True)

if __name__ == '__main__':
    main()
    
    
