#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

regions = {
    'muons' : ['cr_2m_vbf_nodphijjcut', 'cr_1m_vbf_nodphijjcut'],
    'electrons' : ['cr_2e_vbf_nodphijjcut', 'cr_1e_vbf_nodphijjcut'],
}

def calculate_percentage_with_high_dphijj(h):
    '''Calculate rough percentage of events with dphijj > 1.5'''
    dphi_ax_centers = h.axis('dphi').centers()
    high_dphijj_mask = dphi_ax_centers > 1.5

    events_with_high_dphijj = np.sum(h.values()[()][high_dphijj_mask])
    total_events = np.sum(h.values()[()])

    return events_with_high_dphijj / total_events

def preprocess(h, acc, two_dim=False):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Use coarser binnings for dphijj
    new_dphi_ax = hist.Bin("dphi", r"$\Delta\phi_{jj}$", 25, 0, 3.5)
    h = h.rebin('dphi', new_dphi_ax)

    # Rebin mjj too if we're plotting 2D histogram
    if two_dim:
        new_mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', new_mjj_ax)

    return h

def compare_dphijj_dists(acc, outtag, year, channel='muons'):
    '''Compare (inclusive) dphijj distributions between QCD Z and QCD W.'''
    acc.load('dphijj')
    h = preprocess(acc['dphijj'], acc)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    regions_to_look = regions[channel]

    h_z = h.integrate('region', regions_to_look[0]).integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))
    h_w = h.integrate('region', regions_to_look[1]).integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))

    # Plot dphijj distributions (normalized)
    fig, ax = plt.subplots()
    hist.plot1d(h_z, ax=ax, density=True)
    hist.plot1d(h_w, ax=ax, density=True, clear=False)

    ax.set_ylabel('Normalized Counts')

    # Vertical line at dphijj < 1.5 threshold
    ax.axvline(1.5, ymin=0, ymax=1, color='k')

    # Calculate the percentage of events with high dphijj
    percent_events_with_high_dphijj_z = calculate_percentage_with_high_dphijj(h_z)
    percent_events_with_high_dphijj_w = calculate_percentage_with_high_dphijj(h_w)

    ax.text(0., 1., f'Z: {percent_events_with_high_dphijj_z * 100:.2f}%, W: {percent_events_with_high_dphijj_w * 100:.2f}%',
        fontsize=14,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    legend_labels = {
        'muons' : [r'$Z(\mu\mu)$', r'$W(\mu\nu)$'],
        'electrons' : [r'$Z(ee)$', r'$W(e\nu)$'],
    }

    ax.legend(labels=legend_labels[channel])

    ax.text(1., 1., year,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    outpath = pjoin(outdir, f'dphijj_comp_{channel}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def plot_2d_dphijj_vs_mjj(acc, outtag, year, channel='muons'):
    '''Plot dphijj vs mjj histogram for the Z / W ratio.'''
    acc.load('dphijj_vs_mjj')
    h = preprocess(acc['dphijj_vs_mjj'], acc, two_dim=True)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    regions_to_look = regions[channel]

    h_z = h.integrate('region', regions_to_look[0]).integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))
    h_w = h.integrate('region', regions_to_look[1]).integrate('dataset', re.compile(f'WJetsToLNu.*{year}'))

    # Plot dphijj distributions (normalized)
    fig, ax = plt.subplots()
    mjj_edges = h_z.axis('mjj').edges()
    dphi_edges = h_z.axis('dphi').edges()

    # Calculate the Z / W ratio (both normalized)
    sumw_z_norm = h_z.values()[()] / np.sum(h_z.values()[()] )
    sumw_w_norm = h_w.values()[()] / np.sum(h_w.values()[()] )

    r = sumw_z_norm / sumw_w_norm

    opts = {'cmap' : 'viridis'}
    ax.pcolormesh(mjj_edges, dphi_edges, r.T, **opts)

    ax.set_xlabel(r'$M_{jj} \ (GeV)$', fontsize=14)
    ax.set_ylabel(r'$\Delta\phi_{jj}$', fontsize=14)

    titles = {
        'muons' : r'$Z(\mu\mu)$ / $W(\mu\nu)$',
        'electrons' : r'$Z(ee)$ / $W(e\nu)$',
    }

    ax.set_title(titles[channel], fontsize=14)

    ax.text(1., 1., year,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
    )

    # Save figure
    outpath = pjoin(outdir, f'dphijj_vs_mjj_z_over_w_{channel}_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for year in [2017, 2018]:
        for channel in ['electrons', 'muons']:
            compare_dphijj_dists(acc, outtag, year, channel=channel)

            # Try to plot 2D dpphijj vs mjj histogram
            # If we can't find it, just skip it
            try:
                plot_2d_dphijj_vs_mjj(acc, outtag, year, channel=channel)
            except KeyError:
                print(x'Could not find the 2D dphijj/mjj histogram, skipping')

if __name__ == '__main__':
    main()