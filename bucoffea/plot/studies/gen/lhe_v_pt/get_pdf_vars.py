#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import argparse

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from coffea import hist
from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

vpt_ax_fine = list(range(200,400,40)) + list(range(400,1200,80))

BINNING = {
    'vpt' :  hist.Bin('vpt','V $p_{T}$ (GeV)', vpt_ax_fine),
    'mjj' :  hist.Bin('mjj','M(jj) (GeV)',[200]+list(range(500,2500,500)))
}

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='Input path containing merged coffea files.')
    parser.add_argument('--derive_2d', help='Derive and save 2D PDF uncertainties.', action='store_true')
    args = parser.parse_args()

    return args

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
    outdir = './output/theory_variations/pdf'
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
    # Return uncertainty and percent uncertainty
    return unc, unc/nom 

def get_pdf_uncertainty(acc, regex, tag, nominal='pdf_0', integrate_mjj=True):
    '''Given the input accumulator, calculate the
       PDF uncertainty from all PDF variations.'''
    # Define rebinning
    vpt_ax = BINNING['vpt']
    mjj_ax = BINNING['mjj']
    # Old binning for GJets
    # mjj_ax = hist.Bin('mjj','M(jj) (GeV)',[0,200,500,1000,1500,2000])

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
    # as a function of V-pt (if requested)
    if integrate_mjj:
        mjj_slice = slice(200,7500)
        h = h.integrate('mjj', mjj_slice, overflow='over')
    else:
        h = h.rebin('mjj', mjj_ax)
        
    # Get NLO distribution
    nlo = h[re.compile('.*(LHE|amcat).*')].integrate('dataset')

    # Nominal NLO weights, as specified in arguments
    # By defualt, use first PDF variation as nominal
    nlo_nom = nlo.integrate('var', nominal).values(overflow='over')[()]

    # NLO with PDF variations
    # Use a dict to collect NLO contents with all PDF variations
    nlo_var = {}

    for var in nlo.identifiers('var'):
        var_name = var.name
        if 'pdf' not in var_name: 
            continue
        nlo_var[var_name] = nlo.integrate('var', var_name).values(overflow='over')[()]

    unc, percent_unc = calculate_pdf_unc(nlo_nom, nlo_var, tag)
    print(percent_unc)

    # plot_variations(nlo_nom, nlo_var, tag)

    # Plot the % uncertainty as a function of V-pt 
    fig, ax = plt.subplots(1,1)
    if integrate_mjj:
        vpt_edges = vpt_ax.edges(overflow='over')
        vpt_centers = vpt_ax.centers(overflow='over')
        ax.plot(vpt_centers, percent_unc, 'o')
        ax.set_xlabel(r'$p_T(V) \ (GeV)$')
        ax.set_ylabel(r'$\sigma_{pdf}$ / Nominal Counts')
        ax.plot([200, 200], [0, 0.07], 'r')
        ax.set_ylim(0, 0.07)
        ax.grid(True)

    # Make a 2D plot if mjj is not integrated out
    else:
        vpt_edges = vpt_ax.edges(overflow='over')
        vpt_centers = vpt_ax.centers(overflow='over')
        mjj_edges = mjj_ax.edges(overflow='over')
        mjj_centers = mjj_ax.centers(overflow='over')
        
        im = ax.pcolormesh(mjj_edges, vpt_edges, percent_unc.T)
        ax.set_xlabel(r'$M_{jj} \ (GeV)$')
        ax.set_ylabel(r'$p_T(V) \ (GeV)$')
        for ix in range(len(mjj_centers)):
            for iy in range(len(vpt_centers)):
                # textcol = 'white' if ratio[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
                ax.text(
                        mjj_centers[ix],
                        vpt_centers[iy],
                        f'{percent_unc.T[iy, ix]:.3f}',
                        ha='center',
                        va='center',
                        # color=textcol,
                        fontsize=6
                        )

        cb = fig.colorbar(im)
        cb.set_label('PDF unc')
        im.set_clim(0, 0.1)

    tag_to_title = {
        'dy'    : r'$Z\rightarrow \ell \ell$',
        'wjet'  : r'$W\rightarrow \ell \nu$',
        'gjets' : r'$\gamma$ + jets'
    }
    title = tag_to_title[tag]
    ax.set_title(title)
           
    # Save the figure
    if integrate_mjj:
        outdir = './output/theory_variations/pdf'
    else:
        outdir = './output/theory_variations/pdf/2d'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{tag}_pdf_unc.pdf')
    fig.savefig(outpath)
    
    # Return nominal weights and uncertainty
    return nlo_nom, unc

def plot_ratio(noms, uncs, tag, vpt_axis, mjj_axis, outputrootfile, mjj_integrated=True):
    '''Plot the ratio of two processes for nominal case and two variations:
       Variation 1: (nom1+unc1)/(nom2+unc2)
       Variation 2: (nom1-unc1)/(nom2-unc2)
       Nominal: nom1/nom2
       '''
    nom1, nom2 = noms
    unc1, unc2 = uncs

    # Labels for y-axis
    tag_to_label = {
        'z_over_w' : r'$Z \rightarrow \ell \ell$ / $W \rightarrow \ell \nu$',
        'g_over_z' : r'$\gamma$ + jets / $Z \rightarrow \ell \ell$',
    }

    vpt_centers = vpt_axis.centers(overflow='over')
    vpt_edges = vpt_axis.edges(overflow='over')

    ratio_nom = nom1 / nom2
    ratio_up = (nom1+unc1) / (nom2+unc2)
    ratio_down = (nom1-unc1) / (nom2-unc2)

    # Plot the ratios
    fig, ax = plt.subplots(1,1)
    if mjj_integrated:
        ax.plot(vpt_centers, ratio_nom, 'o', label='Nominal')    
        ax.plot(vpt_centers, ratio_up, 'o', label='PDF up')    
        ax.plot(vpt_centers, ratio_down, 'o', label='PDF down')    
        ax.set_xlabel(r'$p_{T} (V)\ (GeV)$')
        ax.set_ylabel(tag_to_label[tag])
        ax.grid(True)
        ax.legend()
    
        ax.set_ylim(0,10)
        ax.plot([200,200], [0,10], 'r')

        # Save output
        outdir = './output/theory_variations/pdf/ratioplots'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        filepath = pjoin(outdir, f'{tag}_pdfunc_ratio.pdf')
        fig.savefig(filepath)

    else:
        pass

    ###########################
    # Plot the ratio of ratios!
    # Ratio(var) / Ratio(nom)
    ###########################
    dratio_up = ratio_up/ratio_nom
    dratio_down = ratio_down/ratio_nom
    
    # Plot the ratio of ratios (1D or 2D)
    plt.close('all')
    if mjj_integrated:
        fig, ax = plt.subplots(1,1)
        ax.plot(vpt_centers, dratio_up, marker='o', label='PDF up / Nominal')
        ax.plot(vpt_centers, dratio_down, marker='o', label='PDF down / Nominal')
    
        ax.set_xlabel(r'$p_T (V)\ (GeV)$')
        ax.set_ylabel(f'{tag_to_label[tag]} (Var / Nom)')
        ax.legend()
        ax.grid(True)

        if tag == 'z_over_w':
            ax.set_ylim(0.99,1.01)
        elif tag == 'g_over_z':
            ax.set_ylim(0.95,1.05)
        
        ax.plot([200, 200], [ax.get_ylim()[0], ax.get_ylim()[1]], 'r')

        # Save figure    
        outdir = './output/theory_variations/pdf/ratioplots'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        filepath = pjoin(outdir, f'{tag}_pdfunc_doubleratio.pdf')
        fig.savefig(filepath)

        # Save into root file
        outputrootfile[f'{tag}_pdfUp'] = (dratio_up, vpt_edges)
        outputrootfile[f'{tag}_pdfDown'] = (dratio_down, vpt_edges)

    else:
        dratios = [('up', dratio_up), ('down', dratio_down)]
        for variation_tag, dratio in dratios:
            fig, ax = plt.subplots(1,1)
            mjj_edges = mjj_axis.edges(overflow='over')
            im = ax.pcolormesh(mjj_edges, vpt_edges, dratio.T)
            ax.set_xlabel(r'$M_{jj} \ (GeV)$')
            ax.set_ylabel(r'$p_T(V) \ (GeV)$')

            cb = fig.colorbar(im)
            cb.set_label('PDF unc on ratio')
            if tag == 'z_over_w':
                im.set_clim(0.99, 1.01)
            elif tag == 'g_over_z':
                if variation_tag == 'up':
                    im.set_clim(0.97, 0.98)
                elif variation_tag == 'down':
                    im.set_clim(1.02, 1.03)
            
            # Set figure title
            if tag == 'z_over_w':
                fig_title = r'$Z(\ell \ell) \ / \ W(\ell \nu)$: PDF {}'.format(variation_tag)
            elif tag == 'g_over_z':
                fig_title = r'$\gamma + jets \ / \ Z(\ell \ell)$: PDF {}'.format(variation_tag)
            
            ax.set_title(fig_title)

            mjj_centers = mjj_axis.centers(overflow='over')

            for ix in range(len(mjj_centers)):
                for iy in range(len(vpt_centers)):
                    # textcol = 'white' if ratio[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
                    ax.text(
                            mjj_centers[ix],
                            vpt_centers[iy],
                            f'{dratio.T[iy, ix]:.4f}',
                            ha='center',
                            va='center',
                            # color=textcol,
                            fontsize=6
                            )

            # Save figure    
            outdir = './output/theory_variations/pdf/ratioplots/2d'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
    
            filepath = pjoin(outdir, f'{tag}_pdfunc_doubleratio_{variation_tag}.pdf')
            fig.savefig(filepath)

        # Save into root file
        outputrootfile[f'{tag}_pdfUp'] = (dratio_up, mjj_edges, vpt_edges)
        outputrootfile[f'{tag}_pdfDown'] = (dratio_down, mjj_edges, vpt_edges)

def main():
    args = parse_cli()
    inpath = args.inpath

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
    outputrootpath = './output/theory_variations/rootfiles'
    if not os.path.exists(outputrootpath):
        os.makedirs(outputrootpath)
    
    rootfile_suffix = '_2d' if args.derive_2d else ''

    outputrootfile_z_over_w = uproot.recreate( pjoin(outputrootpath, f'zoverw_pdf_unc{rootfile_suffix}.root') )
    outputrootfile_g_over_z = uproot.recreate( pjoin(outputrootpath, f'goverz_pdf_unc{rootfile_suffix}.root') )

    w_nom, w_unc = get_pdf_uncertainty(acc, regex='WNJetsToLNu.*', tag='wjet', integrate_mjj=False)
    dy_nom, dy_unc = get_pdf_uncertainty(acc, regex='DYNJetsToLL.*', tag='dy', integrate_mjj=False)
    gjets_nom, gjets_unc = get_pdf_uncertainty(acc, regex='G1Jet.*', tag='gjets', integrate_mjj=False)

    data_for_ratio = {
        'z_over_w' : {'noms' : (dy_nom, w_nom), 'uncs' : (dy_unc, w_unc), 'rootfile' : outputrootfile_z_over_w},
        'g_over_z' : {'noms' : (gjets_nom, dy_nom), 'uncs' : (gjets_unc, dy_unc), 'rootfile' : outputrootfile_g_over_z},
    }
    
    for tag, entry in data_for_ratio.items():
        noms = entry['noms']
        uncs = entry['uncs']
        plot_ratio(noms=noms, 
                   uncs=uncs,
                   tag=tag,
                   vpt_axis=BINNING['vpt'],
                   mjj_axis=BINNING['mjj'],
                   outputrootfile=entry['rootfile'],
                   mjj_integrated=False
                  )

if __name__ == '__main__':
    main()

