#!/usr/bin/env python

import os
import sys
import re
import argparse
import uproot
import warnings
import numpy as np

from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint
from compare_eff import ratio_unc

pjoin = os.path.join

warnings.filterwarnings('ignore')

def do_coarse_rebinning(h):
    '''Make coarser binning for 2D SF histogram.'''
    coarse_binnings = {
        'jetpt' : hist.Bin('jetpt', r'Jet $p_T \ (GeV)$', [40,80,120,160,200,250])
    }

    h = h.rebin('jetpt', coarse_binnings['jetpt'])
    return h

def preprocess(h, acc, year):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h_data = h.integrate('dataset', f'SingleMuon_{year}')[re.compile('.*EmEF.*')]
    h_mc = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))[re.compile('.*EmEF.*')]

    return h_data, h_mc

def calc_eff(h_data, h_mc, regiontag='ak40_in_endcap'):
    '''
    Calculate the efficiencies in data and MC and their stat uncertainties.
    Calculate the values for endcap jets, on positive or negative side or both.
    '''
    h_data_withCut = h_data.integrate('region', f'cr_2m_withEmEF_{regiontag}')
    h_data_withoutCut = h_data.integrate('region', f'cr_2m_noEmEF_{regiontag}')

    h_mc_withCut = h_mc.integrate('region', f'cr_2m_withEmEF_{regiontag}')
    h_mc_withoutCut = h_mc.integrate('region', f'cr_2m_noEmEF_{regiontag}')

    data_num = h_data_withCut.values(overflow='over')[()]
    data_den = h_data_withoutCut.values(overflow='over')[()]

    mc_num = h_mc_withCut.values(overflow='over')[()]
    mc_den = h_mc_withoutCut.values(overflow='over')[()]

    data_eff = data_num / data_den
    mc_eff = mc_num / mc_den

    # Calculate the uncertainties in data eff and MC eff
    clopper_pearson_interval = hist.clopper_pearson_interval
    data_eff_unc = clopper_pearson_interval(data_num, data_den)
    mc_eff_unc = clopper_pearson_interval(mc_num, mc_den)

    return data_eff, mc_eff, data_eff_unc, mc_eff_unc
    
def plot_sf_for_endcap(acc, outtag, rootfile, year=2017, regiontag='ak40_in_endcap'):
    '''Plot 1D SF as a function of jet pt for endcap jets.'''
    variable = 'ak4_pt0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning(h_data)
    h_mc = do_coarse_rebinning(h_mc)

    # Calculate efficiencies for the given endcap region:
    # Specified by regiontag as one of the following:
    # "ak40_in_endcap" : Positive and negative etas combined
    # "ak40_in_pos_endcap" : Positive endcap side
    # "ak40_in_neg_endcap" : Negative endcap side
    data_eff, mc_eff, data_eff_unc, mc_eff_unc = calc_eff(h_data, h_mc, regiontag)

    # Calculate SF
    sf = data_eff / mc_eff

    # Calculate the error on SF
    sf_err = ratio_unc(
        data_eff,
        mc_eff,
        data_eff_unc,
        mc_eff_unc
    )

    # Guard against NaN values
    sf[np.isnan(sf) | np.isinf(sf)] = 1.

    pt_ax = h_data.axis('jetpt')
    xcenters = pt_ax.centers(overflow='over')
    xedges = pt_ax.edges(overflow='over')

    # Plot the 1D SF for the two regions (for now, do not plot errorbars until we figure it out)
    fig, ax = plt.subplots()
    ax.errorbar(xcenters, sf, yerr=sf_err, marker='o', label=None)

    ax.set_xlabel(r'Jet $p_T \ (GeV)$')
    ax.set_ylabel('Data/MC SF')
    ax.set_ylim(0.8,1.2)

    ax.set_title(f'Jet SFs in Endcap: {year}')

    regiontag_to_text = {
        'ak40_in_endcap' : r'$2.5 < |\eta| < 3.0$',
        'ak40_in_pos_endcap' : r'$2.5 < \eta < 3.0$',
        'ak40_in_neg_endcap' : r'$-3.0 < \eta < -2.5$',
    }

    ax.text(1., 1., regiontag_to_text[regiontag],
        fontsize=12,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes
        )

    # Save figure
    outdir = f'./output/{outtag}/endcap_sf'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, f'sf_{regiontag}_{year}.pdf')

    fig.savefig(outpath)
    print(f'File saved: {outpath}')

    # Save the SF to ROOT file, with the stat uncertainties
    root_hist_tag = f'jetsf_{regiontag}_{year}'
    rootfile[root_hist_tag] = (sf, xedges)
    rootfile[f'{root_hist_tag}_statUp'] = (sf+sf_err[0], xedges)
    rootfile[f'{root_hist_tag}_statDown'] = (sf-sf_err[1], xedges)

def calculate_sf_with_variations(h_mc, data_eff, data_eff_err, variation, sf):
    '''Given the MC histogram containing the variations and the variation type, compute the variations of the SF.'''
    # Calculate the up and down variations of the SF
    for var in [f'{variation}Up', f'{variation}Down']:
        h_mc_withCut = h_mc.integrate('region', f'cr_2m_withEmEF_{var}')
        h_mc_withoutCut = h_mc.integrate('region', f'cr_2m_noEmEF_{var}')

        mc_num = h_mc_withCut.values(overflow='over')[()]
        mc_den = h_mc_withoutCut.values(overflow='over')[()]
    
        mc_eff = mc_num / mc_den
        mc_eff_err = np.abs(hist.clopper_pearson_interval(mc_num, mc_den) - mc_eff)

        sf = data_eff / mc_eff
        sf_err = ratio_unc(
            data_eff,
            mc_eff,
            data_eff_err,
            mc_eff_err
        )

        sf[var] = {
            'sf' : sf,
            'err' : sf_err
        }

    return sf

def plot_sf_with_variations(acc, outtag, rootfile, variation, year=2017, regiontag='ak40_in_endcap'):
    '''
    Plot data/MC SF and save to a ROOT file with the given variation. Variations can be one of the systematic sources:
    1. Jet energy scale
    2. Jet energy resolution
    3. Pileup
    4. Prefire
    '''
    variable = 'ak4_pt0'
    acc.load(variable)
    h = acc[variable]

    h_data, h_mc = preprocess(h, acc, year)

    # Use coarser binnings
    h_data = do_coarse_rebinning(h_data)
    h_mc = do_coarse_rebinning(h_mc)

    xcenters = h_data.axis('jetpt').centers(overflow='over')
    xedges = h_data.axis('jetpt').edges(overflow='over')

    # For data, go ahead and calculate the efficiency in the nominal case
    h_data_withCut = h_data.integrate('region', 'cr_2m_withEmEF')
    h_data_withoutCut = h_data.integrate('region', 'cr_2m_noEmEF')    

    data_num = h_data_withCut.values(overflow='over')[()]
    data_den = h_data_withoutCut.values(overflow='over')[()]
    
    data_eff = data_num / data_den
    data_eff_err = np.abs(hist.clopper_pearson_interval(data_num, data_den) - data_eff)

    # For MC calculate the nominal efficiency and the variations
    h_mc_withCut = h_mc.integrate('region', 'cr_2m_withEmEF')
    h_mc_withoutCut = h_mc.integrate('region', 'cr_2m_noEmEF')
    mc_num = h_mc_withCut.values(overflow='over')[()]
    mc_den = h_mc_withoutCut.values(overflow='over')[()]

    mc_eff_nom = mc_num / mc_den
    mc_eff_nom_err = np.abs(hist.clopper_pearson_interval(mc_num, mc_den) - mc_eff)

    # SF dictionary will contain:
    # Data/MC SF for each case: Nominal + variations
    # Stat error on data/MC SF for each case: Nominal + variations
    sf = {}
    sf_nom = data_eff / mc_eff_nom
    sf_nom_err = ratio_unc(
            data_eff, 
            mc_eff_nom,
            data_eff_err,
            mc_eff_nom_err
            )

    # Nominal SF and its error
    sf['nom'] : {
        'sf' : sf_nom, 
        'err' : sf_nom_err
    }

    # Calculate the variations of the SF with the given uncertainty
    sf = calculate_sf_with_variations(h_mc, data_eff, data_eff_err, variation, sf)

    # Plot the variations in SF
    fig, ax, rax = fig_ratio()
    for var, data in sf.items(): 
        ax.errorbar(xcenters, y=data['sf'], yerr=data['err'], label=var, marker='o')

    ax.legend()

    # Plot the ratio of the variations to the nominal
    for idx, (var, data) in enumerate(sf.items()):
        if var == 'nom':
            continue
        rsf = data['sf'] / sf_nom
        rsf_err = data['err'] / sf_nom
        rax.errorbar(xcenters, y=rsf, yerr=rsf_err, marker='o', color=f'C{idx}', ls='')

    rax.set_xlabel(r'Jet $p_T \ (GeV)$')
    rax.set_ylabel('Ratio to nominal')
    rax.set_ylim(0.9,1.1)
    rax.grid(True)

    # Save figure
    outdir = f'./output/{outtag}/endcap_sf'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{variation}_sf_variations_{year}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

    # Save the variations to the output ROOT file
    root_hist_tag = f'jetsf_{regiontag}_{year}'
    rootfile[f'{root_hist_tag}_{variation}Up'] = (sf[f'{variation}Up']['sf'], xedges)
    rootfile[f'{root_hist_tag}_{variation}Down'] = (sf[f'{variation}Down']['sf'], xedges)

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    regiontags = [
        'ak40_in_endcap',
        'ak40_in_pos_endcap',
        'ak40_in_neg_endcap',
    ]

    outputrootpath = f'./output/{outtag}/endcap_sf/root'
    if not os.path.exists(outputrootpath):
        os.makedirs(outputrootpath)
    
    rootfile = uproot.recreate( pjoin(outputrootpath, 'jet_sf_endcaps.root') )
    print(f'ROOT file created: {rootfile}')

    for year in [2017, 2018]:
        for regiontag in regiontags:
            # Calculate 1D SF as a function of jet pt for the endcap jets only
            plot_sf_for_endcap(acc, outtag, year=year, regiontag=regiontag, rootfile=rootfile)

if __name__ == '__main__':
    main()
