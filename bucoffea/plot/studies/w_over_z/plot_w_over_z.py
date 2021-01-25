#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
import mplhep as hep

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

pjoin = os.path.join

def get_regions(channel):
    if channel == 'muons':
        return ['dimuon', 'singlemu']
    elif channel == 'electrons':
        return ['dielec', 'singleel']

    raise RuntimeError(f'Cannot recognize channel: {channel}')

def get_ylabel(channel):
    mapping = {
        'muons'     : r'$Z(\mu\mu) \ / \ W(\mu\nu)$',
        'electrons' : r'$Z(ee) \ / \ W(e\nu)$',
    }
    return mapping[channel]

def plot_z_over_w(infile, outtag, channel='muons', plot='total_bkg'):
    '''Given the fit diagnostics file, plot the pre-fit Z/W ratio as a function of mjj.'''
    regions = get_regions(channel)
    prefit_dir = infile['shapes_prefit']
    
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        # Directories for Z and W control regions
        z_indir = prefit_dir[f'vbf_{year}_{regions[0]}']
        w_indir = prefit_dir[f'vbf_{year}_{regions[1]}']

        if plot == 'total_bkg':
            zll_hist = z_indir['total_background']
            wlv_hist = w_indir['total_background']
            plotlabel = 'Total Bkg Ratio (MC)'
        elif plot == 'leading_bkg':
            zll_hist = z_indir['qcd_zll']
            wlv_hist = w_indir['qcd_wjets']
            plotlabel = 'QCD Z / W'
        elif plot == 'subleading_ewk_bkg':
            zll_hist = z_indir['ewk_zll']
            wlv_hist = w_indir['ewk_wjets']
            plotlabel = 'EWK Z / W'
        else:
            raise ValueError(f'Invalid value for plot variable: {plot}')

        # Compute the ratio and the uncertainty on the ratio
        edges = zll_hist.edges
        ratio = zll_hist.values / wlv_hist.values
        ratio_err = np.sqrt(zll_hist.variances) / wlv_hist.values

        fig, ax = plt.subplots()
        hep.histplot(ratio, edges, yerr=ratio_err, ax=ax)

        ax.text(0., 1., plotlabel, 
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year, 
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.set_xlabel(r'$M_{jj} \ (GeV)$', fontsize=14)
        ax.set_ylabel(get_ylabel(channel), fontsize=14)
        ax.set_xlim(200, 5000)

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        ax.yaxis.set_ticks_position('both')

        # Horizontal line at the theoretically expected ratios
        if plot == 'leading_bkg':
            ax.axhline(0.1, xmin=0, xmax=1, color='black', ls='--')
        elif plot == 'subleading_ewk_bkg':
            ax.axhline(0.08, xmin=0, xmax=1, color='black', ls='--')

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_{channel}_{plot}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    # Input path to fit diagnostics files
    indir = sys.argv[1]
    inpath = pjoin(indir, 'fitDiagnostics_vbf_combined.root')
    infile = uproot.open(inpath)

    outtag = re.findall('merged_.*', indir)[0].replace('/', '')

    # Z(ee)/W(ev) and Z(mm)/W(mv) ratios
    for channel in ['electrons', 'muons']:
        for plot in ['total_bkg', 'leading_bkg', 'subleading_ewk_bkg']:
            plot_z_over_w(infile, outtag, channel=channel, plot=plot)

if __name__ == '__main__':
    main()