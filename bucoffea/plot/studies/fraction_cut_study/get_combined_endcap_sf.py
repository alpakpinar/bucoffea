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

    return fout

def save_to_root(fin, outpath):
    '''
    Prepare final ROOT files with the SF for positive and negative endcaps, to be applied in the analysis.
    Uncertainties:
    -  Stat uncertainties (shape)
    -  Flat JES/JER 1% for 2017
    '''
    outputrootdir = pjoin(outpath, 'root')
    if not os.path.exists(outputrootdir):
        os.makedirs(outputrootdir)
    
    outrootpath = pjoin(outputrootdir, 'jet_id_sf_for_endcaps.root')
    fout = uproot.recreate(outrootpath)
    print(f'ROOT file created: {fout}')

    for year in [2017, 2018]:
        for endcap in ['pos', 'neg']:
            sf_nom = fin[f'sf_{endcap}_endcap_{year}'].values

            # Stat up/down variations
            sf_statUp = fin[f'sf_{endcap}_endcap_{year}_statUp'].values
            sf_statDown = fin[f'sf_{endcap}_endcap_{year}_statDown'].values

            stat_err_up = sf_statUp / sf_nom - 1
            stat_err_down = sf_statDown / sf_nom - 1

            # For 2017, add flat 1% for JES and 1% for JER
            if year == 2017:
                jes = 0.01
                jer = 0.01
            else:
                jes = 0.
                jer = 0.

            total_err_up = np.sqrt(stat_err_up**2 + jes**2 + jer**2)
            total_err_down = np.sqrt(stat_err_down**2 + jes**2 + jer**2)

            # Calculate the final up and down variations on the SF
            sf_totalUp = sf_nom * (1 + total_err_up)
            sf_totalDown = sf_nom * (1 - total_err_down)

            edges = fin[f'sf_{endcap}_endcap_{year}'].edges

            # Save SF + final uncertainties to output ROOT file
            fout[f'sf_{endcap}_endcap_{year}'] = (sf_nom, edges)
            fout[f'sf_{endcap}_endcap_{year}_up'] = (sf_totalUp, edges)
            fout[f'sf_{endcap}_endcap_{year}_down'] = (sf_totalDown, edges)

    return fout

def plot_final_uncs(fin, outpath):
    '''Plot the final SF + uncertainties, all in the same plot.'''
    legend_labels = {
        'pos': r'$2.5 < \eta < 3.0$, {}',
        'neg': r'$-3.0 < \eta < -2.5$, {}',
    }

    fig, ax = plt.subplots()

    for year in [2017, 2018]:
        for endcap in ['pos', 'neg']:
            sf_nom = fin[f'sf_{endcap}_endcap_{year}'].values
            sf_up = fin[f'sf_{endcap}_endcap_{year}_up'].values
            sf_down = fin[f'sf_{endcap}_endcap_{year}_down'].values

            centers = 0.5 * np.sum(fin[f'sf_{endcap}_endcap_{year}'].bins, axis=1)

            sf_err = np.vstack([
                np.abs(sf_up - sf_nom),
                np.abs(sf_down - sf_nom),
            ])

            # Plot the SF with uncertainties
            legend_label = legend_labels[endcap].format(year)
            ax.errorbar(centers, sf_nom, yerr=sf_err, marker='o', label=legend_label)
    
    ax.legend(title='Region, year')
    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel('Data / MC SF')
    ax.set_ylim(0.8,1.2)
    ax.set_title('Jet ID SF For VBF Analysis')
    ax.grid(True)

    # Save figure
    outfile = pjoin(outpath, 'final_sf.pdf')
    fig.savefig(outfile)
    plt.close(fig)

    print(f'File saved: {outfile}')

def main():
    # Path to the input ROOT file containing individual SFs and their uncertainties
    inpath = sys.argv[1]
    fin = uproot.open(inpath)

    outpath = os.path.dirname(os.path.dirname(inpath))

    fout = get_combined_sf(fin, outpath)

    fout = save_to_root(fout, outpath)

    plot_final_uncs(fout, outpath)

if __name__ == '__main__':
    main()