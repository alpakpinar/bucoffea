#!/usr/bin/env python

import os
import sys
import uproot
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint

pjoin = os.path.join

def make_plot(h_before, h_after, histo, outdir):
    assert (h_before.edges == h_after.edges).all()
    edges = h_before.edges
    
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(h_before.values, edges, ax=ax, label='Before Tau')
    hep.histplot(h_after.values, edges, ax=ax, label='After Tau')

    ax.legend()
    histo_pretty_name = histo.decode('utf-8').replace(';1', '')
    ax.set_title(histo_pretty_name)

    # Plot ratio
    centers = ( ( edges+np.roll(edges,-1) ) /2 )[:-1]
    r = h_after.values / h_before.values
    rax.plot(centers, r, ls='', marker='o', color='k')

    rax.grid(True)
    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('After tau / Before')
    rax.set_ylim(0.95,1.05)

    outpath = pjoin(outdir, f'{histo_pretty_name}.pdf')
    fig.savefig(outpath)

    plt.close(fig)

def compare_files(f_before_tau, f_after_tau, directory):
    '''Compare the model files before and after the gen tau update.'''
    w_before_tau = uproot.open(f_before_tau)[directory]
    w_after_tau = uproot.open(f_after_tau)[directory]

    # Output directory to save plots
    outdir = f'./output/{directory}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    assert w_before_tau.keys() == w_after_tau.keys()
    histos = w_before_tau.keys()
    for histo in histos:
        print(f'Working on: {histo}')
        h_before = w_before_tau[histo]
        h_after = w_after_tau[histo]

        make_plot(h_before, h_after, histo, outdir)

def main():
    # Paths to files before and after the gen tau update
    inputdir = './inputs'
    f_before_tau = pjoin(inputdir, 'combined_model_vbf_before_tau.root')
    f_after_tau = pjoin(inputdir, 'combined_model_vbf_after_tau.root')

    directories = [
        'Z_constraints_qcd_withphoton_category_vbf_2017',
        'Z_constraints_qcd_withphoton_category_vbf_2018',
        'W_constraints_qcd_category_vbf_2017',
        'W_constraints_qcd_category_vbf_2018',
        'Z_constraints_ewk_withphoton_category_vbf_2017',
        'Z_constraints_ewk_withphoton_category_vbf_2018',
        'W_constraints_ewk_category_vbf_2017',
        'W_constraints_ewk_category_vbf_2018'
        ]

    for directory in directories:
        compare_files(f_before_tau, f_after_tau, directory)

if __name__ == '__main__':
    main()