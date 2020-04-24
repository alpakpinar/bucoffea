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
from pprint import pprint

pjoin = os.path.join

vpt_ax_coarse = [200, 240, 280, 320, 400, 520, 640, 760, 880, 1080]

BINNING = {
    'vpt' : hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_coarse),
    'mjj' : hist.Bin('mjj','M(jj) (GeV)', [200] + list(range(500,2500,500)))
}

def get_2d_scale_variations(acc, regex, tag, scale_var):
    '''Get scale variations as a function of mjj and gen v-pt.'''
    print(f'Working on: {tag}, {scale_var}')

    vpt_ax = BINNING['vpt']
    mjj_ax = BINNING['mjj']

    # Get the correct pt type from coffea input
    pt_tag = 'combined' if tag != 'gjets' else 'stat1'
    acc.load(f'gen_vpt_vbf_{pt_tag}')
    h = acc[f'gen_vpt_vbf_{pt_tag}']

    # Rebin
    h = h.rebin('vpt', vpt_ax)
    h = h.rebin('mjj', mjj_ax)

    # Merging extensions/datasets, scaling w.r.t xs and lumi 
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    h = h[re.compile(regex)]

    # Get LO and NLO inputs, to calculate the scale factors later
    lo = h[re.compile('.*HT.*')].integrate('dataset')
    nlo = h[re.compile('.*(LHE|amcat).*')].integrate('dataset')
    
    # Choose the relevant scale variation (relevant to NLO only)
    # For LO, choose the nominal (i.e. no variation)
    lo = lo.integrate('var', 'nominal')
    nlo_var = nlo.integrate('var', scale_var)
    nlo_nom = nlo.integrate('var', 'nominal')

    sumw_lo_2d = lo.values()[()]
    sumw_nlo_var_2d = nlo_var.values()[()]
    sumw_nlo_nom_2d = nlo_nom.values()[()]

    # Calculate 2D scale factors, nominal and varied
    # as a function of V-pt and mjj
    sf_nom_2d = sumw_nlo_nom_2d / sumw_lo_2d
    sf_var_2d = sumw_nlo_var_2d / sumw_lo_2d

    # Calculate 2D variation ratio, as a function of V-pt and mjj
    var_ratio = sf_var_2d / sf_nom_2d

    tup1 = (var_ratio, h.axis('vpt'), h.axis('mjj') )
    tup2 = (sumw_nlo_var_2d, sumw_nlo_nom_2d)

    # Return a tuple containing the SF ratio, V-pt and mjj axes
    return tup1, tup2

def plot_individual_scale_vars(tup, var, tag, outtag):
    '''Given the tuple from get_2d_scale_variations, plot the 2D scale variation'''
    ratio, vpt_axis, mjj_axis = tup
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

    im = ax.pcolormesh(vpt_axis.edges(), mjj_axis.edges(), ratio)
    vpt_centers = vpt_axis.centers()
    mjj_centers = mjj_axis.centers()
    ax.set_title(fig_title)
    ax.set_xlabel(r'$p_T(V) \ (GeV)$')
    ax.set_ylabel(r'$M_{jj} \ (GeV)$')
    for ix in range(len(vpt_centers)):
        for iy in range(len(mjj_centers)):
            # textcol = 'white' if ratio[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
            ax.text(
                    vpt_centers[ix],
                    mjj_centers[iy],
                    f'{ratio.T[ix, iy]:.3f}',
                    ha='center',
                    va='center',
                    # color=textcol,
                    fontsize=6
                    )

    cb = fig.colorbar(im)
    cb.set_label('Varied SF / Nominal SF')
    im.set_clim([0.9, 1.1])

    # Save the figure
    outpath = f'./output/theory_variations/{outtag}/scale/individual/2d'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = pjoin(outpath, f'{tag}_kfac_ratio_{var}.pdf')
    fig.savefig(outfile)
    print(f'Figure saved: {outfile}')

def get_2d_ratios(sumw_var, tag, var1, var2):
    '''Get ratio for two physics processes, for a given scale variation.'''
    # Figure out the processes
    if tag == 'zoverw':
        tag1 = 'dy'
        tag2 = 'wjet'
    elif tag == 'goverz':
        tag1 = 'gjets'
        tag2 = 'dy'

    # Get varied and nominal NLO weights for
    # the given scale variation
    sumw1_var, sumw1_nom = sumw_var[tag1][var1]    
    sumw2_var, sumw2_nom = sumw_var[tag2][var2]
    
    # Below, we calculate three ratios: 
    # Ratio with num varied and denom kept nominal
    # Ratio with num kept nominal and denom varied
    # Ratio with both num and denom kept nominal
    ratio_var1 = sumw1_var / sumw2_nom
    ratio_var2 = sumw1_nom / sumw2_var
    ratio_nom = sumw1_nom / sumw2_nom
    
    # Return the varied and nominal ratios
    return (ratio_var1, ratio_var2), ratio_nom 

def plot_ratio_variation(sumw_var, tag, vpt_axis, mjj_axis, outtag, outputrootfile):
    # Combination of individual variations works as follows:
    # The opposite scale variations are combined
    # As an example, when combining zvar_over_w and z_over_wvar, we take the following variations and combine them:
    # zvar_over_w --> mu_r_down AND z_over_wvar --> mu_r_up
    # And the final naming of the histogram in the ROOT file is chosen according to the FIRST ONE (mu_r_down in this case). 
    var_pairs = [
        ('mu_r_down', 'mu_r_up'),
        ('mu_r_up', 'mu_r_down'),
        ('mu_f_down', 'mu_f_up'),
        ('mu_f_up', 'mu_f_down')
    ]

    # Labels for variations
    var_to_label = {
        'mu_r_down' : r'$\mu_R$ down',
        'mu_r_up' : r'$\mu_R$ up',
        'mu_f_down' : r'$\mu_F$ down',
        'mu_f_up' : r'$\mu_F$ up',
    }

    # Calculate combined variation on ratio for each variation pair
    for var1, var2 in var_pairs:
        ratios_var, ratio_nom = get_2d_ratios(sumw_var, tag, var1, var2)
        ratio_with_num_varied, ratio_with_denom_varied = ratios_var

        # Calculate the ratio of ratios for the two cases!
        dratio_with_num_varied = ratio_with_num_varied / ratio_nom
        dratio_with_denom_varied = ratio_with_denom_varied / ratio_nom

        # Combine the two double ratios
        combined_dratio = np.hypot(
            1 - dratio_with_num_varied, 
            1 - dratio_with_denom_varied
            )
        
        combined_dratio = 1 + np.sign(1 - dratio_with_num_varied) * combined_dratio

        # Plot the result as a 2D histogram
        fig, ax = plt.subplots()
        mjj_edges = mjj_axis.edges()
        vpt_edges = vpt_axis.edges()
        im = ax.pcolormesh(mjj_axis.edges(), vpt_axis.edges(), combined_dratio.T)
        
        vpt_centers = vpt_axis.centers()
        mjj_centers = mjj_axis.centers()

        for ix in range(len(mjj_centers)):
            for iy in range(len(vpt_centers)):
                # textcol = 'white' if ratio[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
                ax.text(
                        mjj_centers[ix],
                        vpt_centers[iy],
                        f'{combined_dratio.T[iy, ix]:.3f}',
                        ha='center',
                        va='center',
                        # color=textcol,
                        fontsize=6
                        )


        ax.set_ylabel(r'$p_{T}(V) \ (GeV)$')
        ax.set_xlabel(r'$M_{jj} \ (GeV)$')

        if tag == 'zoverw':
            fig_title = r'$Z(\ell \ell)$ ' + var_to_label[var1] + r' / $W(\ell \nu)$ ' + var_to_label[var2] 
        elif tag == 'goverz':
            fig_title = r'$\gamma$ + jets ' + var_to_label[var1] + r' / $Z(\ell \ell)$ ' + var_to_label[var2] 

        ax.set_title(fig_title)

        cb = fig.colorbar(im)
        cb.set_label('Combined Scale Unc')

        # Save figure
        outdir = f'./output/theory_variations/{outtag}/scale/ratioplots/combinedunc/2d'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'{tag}_{var1}_{var2}.pdf')
        fig.savefig(outpath)

        print(f'Figure saved: {outpath}')

        # Save 2D variations into ROOT file
        var_to_roothistname = {
            'mu_r_down' : 'renScaleDown',
            'mu_r_up' : 'renScaleUp',
            'mu_f_down' : 'facScaleDown',
            'mu_f_up' : 'facScaleUp',
        }

        outputrootfile[f'{tag}_{var_to_roothistname[var1]}'] = (combined_dratio, mjj_edges, vpt_edges)

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

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

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
        'wjet'  : r'WN?JetsToLNu.*',
        'dy'    : r'DYN?JetsToLL.*',
        'gjets' : r'G\d?Jet.*' 
    }

    sumw_var = {}
    for tag,regex in tag_regex.items():
        scale_var_list = scale_var_dict['gjets'] if tag == 'gjets' else scale_var_dict['wjet/dy']
        
        sumw_var[tag] = {}

        for scale_var, scale_var_type in scale_var_list:
            tup, sumw_var[tag][scale_var_type] = get_2d_scale_variations( acc=acc,
                                                                        regex=regex,
                                                                        tag=tag,
                                                                        scale_var=scale_var
                                                                        )        

            plot_individual_scale_vars(tup, var=scale_var, tag=tag, outtag=outtag)

    # After filling out sumw_var, now calculate the variations on ratios
    # Two ratios: Z/W and photons/Z
    ratio_tags = ['zoverw', 'goverz']

    # Create the output ROOT file to save the 
    # 2D scale uncertainties on ratios as a function of v-pt and mjj
    outputrootpath = f'./output/theory_variations/{outtag}/rootfiles'
    if not os.path.exists(outputrootpath):
        os.makedirs(outputrootpath)
    
    outputrootfile_z_over_w = uproot.recreate( pjoin(outputrootpath, 'zoverw_scale_unc_2d.root') )
    outputrootfile_g_over_z = uproot.recreate( pjoin(outputrootpath, 'goverz_scale_unc_2d.root') )

    outputrootfiles = {
        'zoverw' : outputrootfile_z_over_w,
        'goverz' : outputrootfile_g_over_z
    }

    vpt_axis = BINNING['vpt']
    mjj_axis = BINNING['mjj']

    for ratio_tag in ratio_tags:
        plot_ratio_variation(sumw_var, tag=ratio_tag, vpt_axis=vpt_axis, mjj_axis=mjj_axis, outtag=outtag, outputrootfile=outputrootfiles[ratio_tag])

if __name__ == '__main__':
    main()