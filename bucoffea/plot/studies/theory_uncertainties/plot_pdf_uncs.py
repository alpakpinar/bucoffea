#!/usr/bin/env python

# =============================
# Script to plot PDF up/down uncertainties as a function of mjj.
# =============================

from coffea.hist.plot import plot1d
from matplotlib import pyplot as plt
import uproot
import re
import sys
import numpy as np
import os
from pprint import pprint

pjoin = os.path.join

histnames = {
    'PDF up' : '',
    'PDF down' : 'uncertainty_ratio_z_qcd_mjj_unc_zoverw_nlo_pdf_down_{year}'
}

def plot_pdf_uncs(infile, indir):
    '''Plot PDF uncertainties as a function of mjj.'''
    f = uproot.open(infile)
    for year in [2017,2018]:
        fig, ax = plt.subplots()

        pdf_up_histName = f'uncertainty_ratio_z_qcd_mjj_unc_zoverw_nlo_pdf_up_{year}'
        pdf_up = f[pdf_up_histName].values
        pdf_down_histName = f'uncertainty_ratio_z_qcd_mjj_unc_zoverw_nlo_pdf_down_{year}'
        pdf_down = f[pdf_down_histName].values

        mjj_edges = f[pdf_up_histName].edges
        mjj_centers = ((mjj_edges + np.roll(mjj_edges, -1))/2)[:-1]

        ax.plot(mjj_centers, pdf_up, label='PDF up', marker='o')
        ax.plot(mjj_centers, pdf_down, label='PDF down', marker='o')

        ax.set_xlabel(r'$M_{jj} \ (GeV)$')
        ax.set_ylabel('PDF uncertainty')
        ax.grid(True)
        ax.set_title(f'Uncertainty on Z / W: {year}')

        # Save figure
        outdir = indir
        outpath = pjoin(outdir, f'pdf_uncs_{year}.pdf')
        fig.savefig(outpath)

        print(f'File saved: {outpath}')

def main():
    infile = sys.argv[1]
    indir = os.path.dirname(infile)
    plot_pdf_uncs(infile, indir)

if __name__ == '__main__':
    main()

