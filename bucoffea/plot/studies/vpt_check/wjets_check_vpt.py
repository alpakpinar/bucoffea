#!/usr/bin/env python
import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from bucoffea.plot.util import fig_ratio
from bucoffea.plot.style import markers, matplotlib_rc

pjoin = os.path.join

matplotlib_rc()

vpt_edges = [200, 240, 280, 320, 400, 520, 640, 760, 880, 1200]

def get_contributions_to_mjj(f, region='cr_1m_vbf'):
    # Read off the gen boson pt values
    gen_v_pt = f[region]['gen_v_pt'].array()
    print(gen_v_pt)

def plot_gen_v_pt(f, year, region='cr_1m_vbf', mask=None):
    '''Plot gen boson pt distribution'''
    gen_v_pt = f[region]['gen_v_pt'].array()

    # If this mask is specified, look for events with mjj > 2 TeV
    if mask == 'mjj_cut':
        maskarr = f[region]['mjj'].array() > 2000
        gen_v_pt = gen_v_pt[maskarr]

    # Make a histogram and plot it
    h, edges = np.histogram(gen_v_pt, bins=vpt_edges)
    fig, ax, rax = fig_ratio()
    hep.histplot(h, edges, ax=ax)

    ax.set_ylabel('Events')
    ax.set_yscale('log')
    if mask is None:
        title = r'QCD W: {} ($M_{{jj}}$ inclusive)'
        ax.set_ylim(1e2,1e6)
    elif mask == 'mjj_cut':
        title = r'QCD W: {} ($M_{{jj}} > 2 \ TeV$)'
        ax.set_ylim(1e0,1e4)
    ax.set_title(title.format(year))

    # Get the relative distribution of boson pt
    rh = h / np.sum(h)
    opts = markers('data')
    opts.pop('emarker')
    hep.histplot(rh, edges, ax=rax, histtype='errorbar', **opts)

    rax.set_xlabel(r'$p_T(V) \ (GeV)$')
    rax.grid(True)
    rax.set_ylim(0,0.3)
    rax.set_ylabel('Frac. Contribution')

    loc1 = MultipleLocator(0.1)
    loc2 = MultipleLocator(0.05)
    rax.yaxis.set_major_locator(loc1)
    rax.yaxis.set_minor_locator(loc2)

    plt.text(0., 1., r'1-$\mu$ CR',
                fontsize=14,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes
                )

    outdir = './output'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if mask is None:
        outname = f'gen_v_pt_dist_{year}.pdf'
    elif mask == 'mjj_cut':
        outname = f'gen_v_pt_dist_large_mjj_{year}.pdf'
        
    outpath = pjoin(outdir, outname)
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    for year in [2017, 2018]:
        # Input file containing the trees with mjj and gen boson pt
        infile = f'./input/tree_WJetsToLNu_{year}_merged.root'
        f = uproot.open(infile)
    
        plot_gen_v_pt(f, year=year)
        plot_gen_v_pt(f, year=year, mask='mjj_cut')

if __name__ == '__main__':
    main()