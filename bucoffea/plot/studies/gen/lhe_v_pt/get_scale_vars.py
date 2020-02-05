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

def get_old_kfac(tag):
    '''Given the dataset tag, get the nominal 2D VBF k-factors.'''
    f = uproot.open(bucoffea_path('data/sf/theory/2017_gen_v_pt_qcd_sf.root'))
    return f[f'2d_{tag}_vbf'].values

def get_scale_variations(acc, regex, tag, scale_var, scale_var_type, outputrootfile):
    '''Calculate the new k-factors with a scale weight variation.
       Dumps the ratio: 
       --- New k-factors with variation / Old (nominal) k-factors
       into the given output ROOT file.'''

    print(f'Working on: {tag}, {scale_var}')

    # Define rebinning
    if tag in ['wjet', 'dy']:
        vpt_ax_coarse = [0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640, 760, 880,1200]
        vpt_ax_fine = list(range(0,400,40)) + list(range(400,1200,80))
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_fine)
        mjj_ax = hist.Bin('mjj','M(jj) (GeV)', [0,200] + list(range(500,2500,500)))
    elif tag in ['gjets']:
        vpt_ax_coarse = [0, 40, 80, 120, 160, 200, 240, 280, 320, 400, 520, 640]
        vpt_ax_fine = list(range(0,400,40)) + list(range(400,1200,80)) 
        vpt_ax = hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_fine)
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
    nlo_nom = nlo.integrate('var', 'nominal')

    # Get 1D LO and NLO weights to calculate the variation
    if tag in ['wjet', 'dy']:
        mjj_slice = slice(200,2000)
    elif tag in ['gjets']:
        mjj_slice = slice(200,1500)
    lo_1d = lo.integrate('mjj', mjj_slice, overflow='over')
    nlo_var_1d = nlo_var.integrate('mjj', mjj_slice, overflow='over')
    nlo_nom_1d = nlo_nom.integrate('mjj', mjj_slice, overflow='over')
    
    sumw_lo_1d = lo_1d.values(overflow='over')[()]
    sumw_nlo_var_1d = nlo_var_1d.values(overflow='over')[()]
    sumw_nlo_nom_1d = nlo_nom_1d.values(overflow='over')[()]

    # Calculate 1D scale factors, nominal and varied
    # as a function of V-pt
    sf_nom_1d = sumw_nlo_nom_1d / sumw_lo_1d
    sf_var_1d = sumw_nlo_var_1d / sumw_lo_1d

    # Calculate 1D variation ratio, as a function of V-pt
    var_ratio = sf_var_1d / sf_nom_1d
    
    # Calculate nominal 2D scale factor 
    sumw_lo = lo.values(overflow='over')[()]
    sumw_nlo_nom = nlo_nom.values(overflow='over')[()]

    sf_nom = sumw_nlo_nom / sumw_lo 

    tup = (var_ratio, yaxis.edges(overflow='over'))
 
    # Save to the ROOT file
    outputrootfile[f'{tag}_vbf_{scale_var_type}'] = tup

    return tup

def plot_ratio_vpt(tup, var, tag):
    '''Given the tuple contatining the SF ratio (variational/nominal) and 
       bin edges, plot ratios in each mjj bin as a function of v-pt.'''
    ratio, x_edges = tup
    vpt_centers = ((x_edges + np.roll(x_edges,0))/2)[:-1]
    fig, ax = plt.subplots(1,1)

    # Figure out the variation and the relevant title
    var_title = {
        'gjets' : {
            'scale_1' : r'$\gamma$ + jets: $\mu_R = 0.5$, $\mu_F = 1.0$',
            'scale_3' : r'$\gamma$ + jets: $\mu_R = 1.0$, $\mu_F = 0.5$',
            'scale_5' : r'$\gamma$ + jets: $\mu_R = 1.0$, $\mu_F = 2.0$',
            'scale_7' : r'$\gamma$ + jets: $\mu_R = 2.0$, $\mu_F = 1.0$'
        },
        'wjet' : {
            'scale_1' : r'$W\rightarrow l \nu$: $\mu_R = 0.5$, $\mu_F = 1.0$',
            'scale_3' : r'$W\rightarrow l \nu$: $\mu_R = 1.0$, $\mu_F = 0.5$',
            'scale_4' : r'$W\rightarrow l \nu$: $\mu_R = 1.0$, $\mu_F = 2.0$',
            'scale_6' : r'$W\rightarrow l \nu$: $\mu_R = 2.0$, $\mu_F = 1.0$'
        },
        'dy' : {
            'scale_1' : r'$Z\rightarrow ll$: $\mu_R = 0.5$, $\mu_F = 1.0$',
            'scale_3' : r'$Z\rightarrow ll$: $\mu_R = 1.0$, $\mu_F = 0.5$',
            'scale_4' : r'$Z\rightarrow ll$: $\mu_R = 1.0$, $\mu_F = 2.0$',
            'scale_6' : r'$Z\rightarrow ll$: $\mu_R = 2.0$, $\mu_F = 1.0$'
        },
    }

    fig_title = var_title[tag][var] 

    ax.set_xlim(-50.0, vpt_centers[-1]+50.0)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel(r'$p_T(V) \ (GeV)$')
    ax.set_ylabel('Varied / Nominal SF')
    ax.set_title(fig_title)
    ax.plot([-50.0, vpt_centers[-1]+50.0], [1, 1], 'r')
    ax.plot(vpt_centers, ratio, 'o')
    ax.grid(True)

    # Save the figure
    outpath = './output/kfac_variations'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = pjoin(outpath, f'{tag}_kfac_ratio_{var}.png')
    fig.savefig(outfile)

def plot_ratio_vpt_combined(tup_combined, tag):
    '''Given the tup_combinedle contatining the SF ratio (variational/nominal) and 
       bin edges for all variations for a given process, 
       plot ratios in each mjj bin as a function of v-pt.'''
    ratio_combined, x_edges = tup_combined[:,0], tup_combined[0,1]
    vpt_centers = ((x_edges + np.roll(x_edges,0))/2)[:-1]
    fig, ax = plt.subplots(1,1)

    # Figure out the variation and the relevant title, and label for the legend
    var_title_label = {
        'gjets' : {
            'scale_1' : (r'$\gamma$ + jets', r'$\mu_R = 0.5$, $\mu_F = 1.0$'),
            'scale_3' : (r'$\gamma$ + jets', r'$\mu_R = 1.0$, $\mu_F = 0.5$'),
            'scale_5' : (r'$\gamma$ + jets', r'$\mu_R = 1.0$, $\mu_F = 2.0$'),
            'scale_7' : (r'$\gamma$ + jets', r'$\mu_R = 2.0$, $\mu_F = 1.0$')
        },
        'wjet' : {
            'scale_1' : (r'$W\rightarrow \ell \nu$', r'$\mu_R = 0.5$, $\mu_F = 1.0$'),
            'scale_3' : (r'$W\rightarrow \ell \nu$', r'$\mu_R = 1.0$, $\mu_F = 0.5$'),
            'scale_4' : (r'$W\rightarrow \ell \nu$', r'$\mu_R = 1.0$, $\mu_F = 2.0$'),
            'scale_6' : (r'$W\rightarrow \ell \nu$', r'$\mu_R = 2.0$, $\mu_F = 1.0$')
        },
        'dy' : {
            'scale_1' : (r'$Z\rightarrow \ell \ell$', r'$\mu_R = 0.5$, $\mu_F = 1.0$'),
            'scale_3' : (r'$Z\rightarrow \ell \ell$', r'$\mu_R = 1.0$, $\mu_F = 0.5$'),
            'scale_4' : (r'$Z\rightarrow \ell \ell$', r'$\mu_R = 1.0$, $\mu_F = 2.0$'),
            'scale_6' : (r'$Z\rightarrow \ell \ell$', r'$\mu_R = 2.0$, $\mu_F = 1.0$')
        },
    }

    variations = var_title_label[tag]
    for idx, var_info in enumerate(variations.values()):
        ax.plot([-50.0, vpt_centers[-1]+50.0], [1, 1], 'r')
        ax.plot(vpt_centers, ratio_combined[idx], 'o', label=var_info[1])

    ax.set_xlim(-50.0, vpt_centers[-1]+50.0)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel(r'$p_T(V) \ (GeV)$')
    ax.set_ylabel('Varied / Nominal SF')
    fig_title = variations['scale_1'][0]
    ax.set_title(fig_title)
    ax.grid(True)
    plt.legend()

    # Save the figure
    outpath = './output/kfac_variations'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = pjoin(outpath, f'{tag}_kfac_ratio_combined.png')
    fig.savefig(outfile)

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
            ('scale_1', 'mu_r_down'),
            ('scale_3', 'mu_f_down'),
            ('scale_5', 'mu_f_up'),
            ('scale_7', 'mu_r_up')
                ],
        'wjet/dy' : [
            ('scale_1', 'mu_r_down'),
            ('scale_3', 'mu_f_down'),
            ('scale_4', 'mu_f_up'),
            ('scale_6', 'mu_r_up')
                ]
                
    }

    tag_regex = {
        'wjet'  : 'WN?JetsToLNu.*',
        'dy'    : 'DYN?JetsToLL.*',
        'gjets' : 'G\d?Jet.*' 
    }
    
    for tag,regex in tag_regex.items():
        scale_var_list = scale_var_dict['gjets'] if tag == 'gjets' else scale_var_dict['wjet/dy']

        tup_combined = []

        for scale_var, scale_var_type in scale_var_list:
            tup = get_scale_variations( acc=acc,
                                        regex=regex,
                                        tag=tag,
                                        scale_var=scale_var,
                                        scale_var_type=scale_var_type,
                                        outputrootfile=outputrootfile
                                        )

            tup_combined.append(tup)

            plot_ratio_vpt(tup, var=scale_var, tag=tag)
        
        tup_combined = np.array(tup_combined)
        plot_ratio_vpt_combined(tup_combined, tag=tag)

if __name__ == '__main__':
    main()
