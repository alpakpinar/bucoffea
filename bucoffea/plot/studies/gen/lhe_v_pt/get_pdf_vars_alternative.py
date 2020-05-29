#!/usr/bin/env python

# ============================
# Calculate PDF uncertainty in an alternative way:
# Directly feed in Z/W ratio to hessian uncertainity equation.
# ============================

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
    args = parser.parse_args()
    return args

def hessian_unc(nom, var):
    '''Calculate PDF uncertainty for a Hessian set.'''
    unc=np.zeros_like(nom) 
    for variation in var.values():
        unc += (nom - variation)**2
    return np.sqrt(unc)

def plot_unc(xaxis, yaxis, unc_array, outtag):
    '''Plot given 2D uncertainty array.'''
    fig, ax = plt.subplots()
    im = ax.pcolormesh(xaxis.edges(overflow='over'), yaxis.edges(overflow='over'), unc_array.T)

    x_centers = xaxis.centers(overflow='over')
    y_centers = yaxis.centers(overflow='over')

    # Print out the uncertainty values on the plot
    for ix in range(len(x_centers)):
            for iy in range(len(y_centers)):
                # textcol = 'white' if sf.T[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
                ax.text(
                        x_centers[ix],
                        y_centers[iy],
                        f'  {unc_array.T[iy, ix]:.3f}',
                        ha='center',
                        va='center',
                        # color=textcol,
                        fontsize=6
                        )

    ax.set_ylabel(r'$p_T(V) \ (GeV)$')
    ax.set_xlabel(r'$M_{jj} \ (GeV)$')
    ax.set_title(r'$Z(\ell \ell) \ / \ W(\ell \nu)$')
    cb = fig.colorbar(im)
    cb.set_label('PDF uncertainty')

    # Save figure
    outdir = f'./output/theory_variations/{outtag}/pdf/alternative'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'z_over_w_2duncs.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')


def get_pdf_uncertainty(acc, regex, tag, outtag, nominal='pdf_0'):
    '''Calculate the PDF uncertainty on the ratio directly, regex should contain two datasets to take the ratio.'''
    vpt_ax = BINNING['vpt']
    mjj_ax = BINNING['mjj']
    acc.load('gen_vpt_vbf_combined')
    h = acc['gen_vpt_vbf_combined']

    h = h.rebin('vpt', vpt_ax)
    h = h.rebin('mjj', mjj_ax)

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)
    h = h[re.compile(regex)]

    # Get Z and W NLO distributions
    nlo_z = h[re.compile('DY.*(LHE|amcat).*')].integrate('dataset')
    nlo_w = h[re.compile('W.*(LHE|amcat).*')].integrate('dataset')

    # Get nominal values for each 
    nlo_z_nom = nlo_z.integrate('var', nominal).values(overflow='over')[()]
    nlo_w_nom = nlo_w.integrate('var', nominal).values(overflow='over')[()]

    # NLO with PDF variations
    nlo_z_var = {}
    nlo_w_var = {}

    for var in nlo_z.identifiers('var'):
        var_name = var.name
        if 'pdf' not in var_name:
            continue
        nlo_z_var[var_name] = nlo_z.integrate('var', var_name).values(overflow='over')[()]
        nlo_w_var[var_name] = nlo_w.integrate('var', var_name).values(overflow='over')[()]
    
    # Calculate the uncertainty directly on the Z/W ratio
    z_over_w_nom = nlo_z_nom / nlo_w_nom
    z_over_w_var = {}
    for var in nlo_z_var.keys():
        z_over_w_var[var] = nlo_z_var[var] / nlo_w_var[var]

    unc = hessian_unc(z_over_w_nom, z_over_w_var)
    print(unc)

    # Plot the 2D uncertainty on the ratio
    plot_unc(h.axis('mjj'), h.axis('vpt'), unc, outtag)

    # Return the nominal ratio and uncertainty
    return z_over_w_nom, unc

def plot_ratio(nom, unc, vpt_axis, mjj_axis, outputrootfile, tag, outtag):
    '''Plot the double ratio for PDF uncertainty.'''
    vpt_edges = vpt_axis.edges(overflow='over')
    vpt_centers = vpt_axis.centers(overflow='over')
    mjj_edges = mjj_axis.edges(overflow='over')
    mjj_centers = mjj_axis.centers(overflow='over')

    # Calculate the double ratios
    ratio_up = (nom + unc) / nom
    ratio_down = (nom - unc) / nom

    # Plot the ratios
    ratios = [('Up', ratio_up), ('Down', ratio_down)]
    for variation_tag, ratio in ratios:
        fig, ax = plt.subplots(1,1)
        im = ax.pcolormesh(mjj_edges, vpt_edges, ratio.T)
        ax.set_xlabel(r'$M_{jj} \ (GeV)$')
        ax.set_ylabel(r'$p_T(V) \ (GeV)$')

        cb = fig.colorbar(im)
        cb.set_label('PDF unc on ratio')
        if variation_tag == 'Up':
            im.set_clim(1.,1.03)
        if variation_tag == 'Down':
            im.set_clim(0.97,1.)

        fig_title = r'$Z(\ell \ell) \ / \ W(\ell \nu)$: PDF {}'.format(variation_tag)
        ax.set_title(fig_title)

        for ix in range(len(mjj_centers)):
            for iy in range(len(vpt_centers)):
                # textcol = 'white' if ratio[iy, ix] < 0.5*(clims[0]+clims[1]) else 'black'
                ax.text(
                        mjj_centers[ix],
                        vpt_centers[iy],
                        f'{ratio.T[iy, ix]:.4f}',
                        ha='center',
                        va='center',
                        # color=textcol,
                        fontsize=6
                        )

        # Save the figure
        outdir = f'./output/theory_variations/{outtag}/pdf/alternative/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        filepath = pjoin(outdir, f'{tag}_pdfunc_doubleratio_{variation_tag}.pdf')
        fig.savefig(filepath)

        print(f'File saved: {filepath}')

    # Save up and down variations to ROOT file
    outputrootfile[f'{tag}_pdfUp'] = (ratio_up, mjj_edges, vpt_edges)
    outputrootfile[f'{tag}_pdfDown'] = (ratio_down, mjj_edges, vpt_edges)
    print(f'PDF up/down variations are saved to ROOT file: {outputrootfile}')
    print('DONE')

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

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    # Get the mjj, V-pt axes and the calculated 2D PDF uncertainty
    z_over_w_nom, pdf_unc = get_pdf_uncertainty(acc, regex='DYNJetsToLL.*|WNJetsToLNu.*', tag='zoverw', outtag=outtag)

    # Output ROOT file, where the PDF variations are to be saved
    outputrootpath = f'./output/theory_variations/{outtag}/pdf/alternative'
    if not os.path.exists(outputrootpath):
        os.makedirs(outputrootpath)
    outputrootfile = uproot.recreate( pjoin(outputrootpath, 'z_over_w_pdf_unc.root') )

    plot_ratio(nom=z_over_w_nom, unc=pdf_unc, vpt_axis=BINNING['vpt'], mjj_axis=BINNING['mjj'], outputrootfile=outputrootfile, tag='z_over_w', outtag=outtag)

if __name__ == '__main__':
    main()
