#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np
from matplotlib import pyplot as plt

pjoin = os.path.join

def get_combined_sf(fin, outpath):
    '''From the individual endcap SFs, get the combined SF for the positive and negative halves of endcap.'''
    # Output root file to save the results
    outputrootdir = pjoin(outpath, 'root')
    if not os.path.exists(outputrootdir):
        os.makedirs(outputrootdir)
    
    outrootpath = pjoin(outputrootdir, 'jet_sf_endcaps_final.root')
    fout = uproot.recreate(outrootpath)
    print(f'ROOT file created: {fout}')
    for year in [2017, 2018]:
        sf_combined = fin[f'jetsf_ak40_in_endcap_{year}'].values
        sf_pos = fin[f'jetsf_ak40_in_pos_endcap_{year}'].values
        sf_neg = fin[f'jetsf_ak40_in_neg_endcap_{year}'].values

        edges = fin[f'jetsf_ak40_in_endcap_{year}'].edges
        centers = 0.5 * np.sum(fin[f'jetsf_ak40_in_endcap_{year}'].bins, axis=1)

        # Stat errors
        sf_combined_statUp = np.abs(fin[f'jetsf_ak40_in_endcap_{year}_statUp'].values - sf_combined)
        sf_combined_statDown = np.abs(fin[f'jetsf_ak40_in_endcap_{year}_statDown'].values - sf_combined)
        
        sf_pos_statUp = np.abs(fin[f'jetsf_ak40_in_pos_endcap_{year}_statUp'].values - sf_pos)
        sf_pos_statDown = np.abs(fin[f'jetsf_ak40_in_pos_endcap_{year}_statDown'].values - sf_pos)
        
        sf_neg_statUp = np.abs(fin[f'jetsf_ak40_in_neg_endcap_{year}_statUp'].values - sf_neg)
        sf_neg_statDown = np.abs(fin[f'jetsf_ak40_in_neg_endcap_{year}_statDown'].values - sf_neg)

        # For the first three bins (p_T < 160 GeV), get individual SF
        # For the last three bins (p_T > 160 GeV), get the combined SF
        new_sf_pos = np.concatenate([sf_pos[:3], sf_combined[3:]])
        new_sf_neg = np.concatenate([sf_neg[:3], sf_combined[3:]])

        # Same for the stat errors
        new_sf_pos_statUp = np.concatenate([
            sf_pos_statUp[:3],
            sf_combined_statUp[3:]
        ])
        new_sf_pos_statDown = np.concatenate([
            sf_pos_statDown[:3],
            sf_combined_statDown[3:]
        ])
        
        new_sf_pos_statErr = np.vstack([new_sf_pos_statUp, new_sf_pos_statDown])

        new_sf_neg_statUp = np.concatenate([
            sf_neg_statUp[:3],
            sf_combined_statUp[3:]
        ])
        new_sf_neg_statDown = np.concatenate([
            sf_neg_statDown[:3],
            sf_combined_statDown[3:]
        ])

        new_sf_neg_statErr = np.vstack([new_sf_neg_statUp, new_sf_neg_statDown])

        # Plot the SF!
        fig, ax = plt.subplots()
        ax.errorbar(centers, y=new_sf_pos, yerr=new_sf_pos_statErr, marker='o', label=r'$2.5 < \eta < 3.0$')
        ax.errorbar(centers, y=new_sf_neg, yerr=new_sf_neg_statErr, marker='o', label=r'$-3.0 < \eta < -2.5$')

        ax.legend()
        ax.set_xlabel(r'Jet $p_T \ (GeV)$')
        ax.set_ylabel('Data / MC SF')
        ax.set_ylim(0.8,1.2)
        ax.set_title(f'Endcap SF: {year}', fontsize=14)

        # Save figure
        outpdffile = pjoin(outpath, f'endcap_sf_{year}.pdf')
        fig.savefig(outpdffile)
        plt.close(fig)
        print(f'File saved: {outpdffile}')

        # Save into output ROOT file
        fout[f'sf_pos_endcap_{year}'] = (new_sf_pos, edges)
        fout[f'sf_pos_endcap_{year}_statUp'] = (new_sf_pos + new_sf_pos_statUp, edges)
        fout[f'sf_pos_endcap_{year}_statDown'] = (new_sf_pos + new_sf_pos_statDown, edges)

        fout[f'sf_neg_endcap_{year}'] = (new_sf_neg, edges)
        fout[f'sf_neg_endcap_{year}_statUp'] = (new_sf_neg + new_sf_neg_statUp, edges)
        fout[f'sf_neg_endcap_{year}_statDown'] = (new_sf_neg + new_sf_neg_statDown, edges)

def main():
    # Path to the input ROOT file containing individual SFs and their uncertainties
    inpath = sys.argv[1]
    fin = uproot.open(inpath)

    outpath = os.path.dirname(os.path.dirname(inpath))

    get_combined_sf(fin, outpath)

if __name__ == '__main__':
    main()