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

        # For the first three bins (p_T < 160 GeV), get individual SF
        # For the last three bins (p_T > 160 GeV), get the combined SF
        new_sf_pos = np.concatenate([sf_pos[:3], sf_combined[3:]])
        new_sf_neg = np.concatenate([sf_neg[:3], sf_combined[3:]])
        
        # Get all errors
        err_dict = {
            'pos_eta' : {},
            'neg_eta' : {}
        }
        for sys in ['stat', 'jesTotal', 'jer', 'pileup', 'prefire']:
            sf_combined_up = np.abs(fin[f'jetsf_ak40_in_endcap_{year}_{sys}Up'].values - sf_combined)
            sf_combined_down = np.abs(fin[f'jetsf_ak40_in_endcap_{year}_{sys}Down'].values - sf_combined)
            
            sf_pos_up = np.abs(fin[f'jetsf_ak40_in_pos_endcap_{year}_{sys}Up'].values - sf_pos)
            sf_pos_down = np.abs(fin[f'jetsf_ak40_in_pos_endcap_{year}_{sys}Down'].values - sf_pos)
            
            sf_neg_up = np.abs(fin[f'jetsf_ak40_in_neg_endcap_{year}_{sys}Up'].values - sf_neg)
            sf_neg_down = np.abs(fin[f'jetsf_ak40_in_neg_endcap_{year}_{sys}Down'].values - sf_neg)

            # Same for the stat errors
            new_sf_pos_up = np.concatenate([
                sf_pos_up[:3],
                sf_combined_up[3:]
            ])
            new_sf_pos_down = np.concatenate([
                sf_pos_down[:3],
                sf_combined_down[3:]
            ])
            
            # Store the up/down errors in the error dictionary
            err_dict['pos_eta'][sys] = np.vstack([new_sf_pos_up, new_sf_pos_down])

            new_sf_neg_up = np.concatenate([
                sf_neg_up[:3],
                sf_combined_up[3:]
            ])
            new_sf_neg_down = np.concatenate([
                sf_neg_down[:3],
                sf_combined_down[3:]
            ])
    
            err_dict['neg_eta'][sys] = np.vstack([new_sf_neg_up, new_sf_neg_down])

        # Plot the SF (with the stat uncs for now)!
        fig, ax = plt.subplots()
        ax.errorbar(centers, y=new_sf_pos, yerr=err_dict['pos_eta']['stat'], marker='o', label=r'$2.5 < \eta < 3.0$')
        ax.errorbar(centers, y=new_sf_neg, yerr=err_dict['neg_eta']['stat'], marker='o', label=r'$-3.0 < \eta < -2.5$')

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

        # Save the variations as well
        for sys, stack in err_dict['pos_eta'].items():
            sf_up, sf_down = stack
            fout[f'sf_pos_endcap_{year}_{sys}Up'] = (new_sf_pos + sf_up, edges)
            fout[f'sf_pos_endcap_{year}_{sys}Down'] = (new_sf_pos + sf_down, edges)

        fout[f'sf_neg_endcap_{year}'] = (new_sf_neg, edges)
        for sys, stack in err_dict['neg_eta'].items():
            sf_up, sf_down = stack
            fout[f'sf_neg_endcap_{year}_{sys}Up'] = (new_sf_neg + sf_up, edges)
            fout[f'sf_neg_endcap_{year}_{sys}Down'] = (new_sf_neg + sf_down, edges)

def main():
    # Path to the input ROOT file containing individual SFs and their uncertainties
    inpath = sys.argv[1]
    fin = uproot.open(inpath)

    outpath = os.path.dirname(os.path.dirname(inpath))

    get_combined_sf(fin, outpath)

if __name__ == '__main__':
    main()