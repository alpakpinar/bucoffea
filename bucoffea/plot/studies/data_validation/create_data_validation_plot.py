#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import mplhep as hep
import uproot
from coffea import hist
from coffea.hist.plot import poisson_interval
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

xlabels = {
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
}

ylabels = {
    'zll_over_wlv'     : r'$Z(\ell\ell) \ / \ W(\ell\nu)$',
    'zll_over_gjets'   : r'$Z(\ell\ell) \ / \ \gamma + jets$',
    'zee_over_gjets'   : r'$Z(ee) \ / \ \gamma + jets$',
    'wmunu_over_gjets' : r'$W(\mu\nu) \ / \ \gamma + jets$',
    'wenu_over_gjets'  : r'$W(e\nu) \ / \ \gamma + jets$',
    'wlnu_over_gjets'  : r'$W(\ell\nu) \ / \ \gamma + jets$'
}

binning = {
    # 'ak4_eta0' : hist.Bin('jeteta', r'Leading Jet $\eta$', [-5, -4, -3] + list(np.arange(-2.8,3,0.4)) + [3, 4, 5]),
    # 'ak4_eta1' : hist.Bin('jeteta', r'Trailing Jet $\eta$', [-5, -4, -3] + list(np.arange(-2.8,3,0.4)) + [3, 4, 5]),
    'ak4_eta0' : hist.Bin('jeteta', r'Leading Jet $\eta$', np.arange(-5,6)),
    'ak4_eta1' : hist.Bin('jeteta', r'Trailing Jet $\eta$', np.arange(-5,6)),
}

# Regions to be combined for Z and W
to_be_combined = {
    'zll' : ('cr_2m_vbf', 'cr_2e_vbf'),
    'wlv' : ('cr_1m_vbf', 'cr_1e_vbf')
}

# Flat uncertainty values
flat_uncs = {
    'zll_over_wlv' : {
        'jes_jer' : 0.02, # 2% for JES/JER uncertainties
    }
}

def create_data_validation_plot(acc, tag, outtag, region1, region2, year, variable='ak4_eta0'):
    '''Create data validation plot, given the input accumulator and two regions being considered.'''
    acc.load(variable)
    h = acc[variable]

    # Preprocess the histogram
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin
    if variable in ['ak4_eta0', 'ak4_eta1']:
        h = h.rebin('jeteta', binning[variable])

    # Regular expressions for data in each region
    data = {
        'sr_vbf' : None,
        'cr_1m_vbf' : f'MET_{year}',
        'cr_2m_vbf' : f'MET_{year}',
        'cr_1e_vbf' : f'EGamma_{year}',
        'cr_2e_vbf' : f'EGamma_{year}',
        'cr_g_vbf' : f'EGamma_{year}',
    }

    # Regular expressions for MC in each region
    mc_lo = {
        'sr_vbf' : re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_g_vbf' : re.compile(f'(GJets_(DR-0p4|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*{year}'),
    }

    # Get data and MC
    if region1 in to_be_combined.keys():
        regions_to_be_combined = to_be_combined[region1]
        data_region1_sumw_1, data_region1_sumw2_1 = h.integrate('region', regions_to_be_combined[0]).integrate(
                                'dataset', data[regions_to_be_combined[0]]
                                ).values(sumw2=True)[()]
        data_region1_sumw_2, data_region1_sumw2_2 = h.integrate('region', regions_to_be_combined[1]).integrate(
                                'dataset', data[regions_to_be_combined[1]]
                                ).values(sumw2=True)[()]
        
        data_region1_sumw = data_region1_sumw_1 + data_region1_sumw_2
        data_region1_sumw2 = data_region1_sumw2_1 + data_region1_sumw2_2

        mc_region1_sumw_1, mc_region1_sumw2_1 = h.integrate('region', regions_to_be_combined[0]).integrate(
            'dataset', mc_lo[regions_to_be_combined[0]]
            ).values(sumw2=True)[()]
        mc_region1_sumw_2, mc_region1_sumw2_2 = h.integrate('region', regions_to_be_combined[1]).integrate(
            'dataset', mc_lo[regions_to_be_combined[1]]
            ).values(sumw2=True)[()]

        mc_region1_sumw = mc_region1_sumw_1 + mc_region1_sumw_2
        mc_region1_sumw2 = mc_region1_sumw2_1 + mc_region1_sumw2_2

    else:
        data_region1_sumw, data_region1_sumw2 = h.integrate('region', region1).integrate('dataset', data[region1]).values(sumw2=True)[()]
        mc_region1_sumw, mc_region1_sumw2 = h.integrate('region', region1).integrate('dataset', mc_lo[region1]).values(sumw2=True)[()]
    
    if region2 in to_be_combined.keys():
        regions_to_be_combined = to_be_combined[region2]
        data_region2_sumw_1, data_region2_sumw2_1 = h.integrate('region', regions_to_be_combined[0]).integrate(
                                'dataset', data[regions_to_be_combined[0]]
                                ).values(sumw2=True)[()]
        data_region2_sumw_2, data_region2_sumw2_2 = h.integrate('region', regions_to_be_combined[1]).integrate(
                                'dataset', data[regions_to_be_combined[1]]
                                ).values(sumw2=True)[()]

        data_region2_sumw = data_region2_sumw_1 + data_region2_sumw_2
        data_region2_sumw2 = data_region2_sumw2_1 + data_region2_sumw2_2

        mc_region2_sumw_1, mc_region2_sumw2_1 = h.integrate('region', regions_to_be_combined[0]).integrate(
                                'dataset', mc_lo[regions_to_be_combined[0]]
                                ).values(sumw2=True)[()]
        mc_region2_sumw_2, mc_region2_sumw2_2 = h.integrate('region', regions_to_be_combined[1]).integrate(
                                'dataset', mc_lo[regions_to_be_combined[1]]
                                ).values(sumw2=True)[()]

        mc_region2_sumw = mc_region2_sumw_1 + mc_region2_sumw_2
        mc_region2_sumw2 = mc_region2_sumw2_1 + mc_region2_sumw2_2
    else:
        data_region2_sumw, data_region2_sumw2 = h.integrate('region', region2).integrate('dataset', data[region2]).values(sumw2=True)[()]
        mc_region2_sumw, mc_region2_sumw2 = h.integrate('region', region2).integrate('dataset', mc_lo[region2]).values(sumw2=True)[()]

    # Calculate the ratio of data in the two regions 
    data_ratio = data_region1_sumw / data_region2_sumw
    mc_ratio = mc_region1_sumw / mc_region2_sumw
    # Guard against NaN values
    data_ratio[np.isinf(data_ratio) | np.isnan(data_ratio)] = 1.
    mc_ratio[np.isinf(mc_ratio) | np.isnan(mc_ratio)] = 1.

    # Calculate the error on the ratio
    # Following in the footsteps of: 
    # https://github.com/CoffeaTeam/coffea/blob/199f7c8aeb4a40bc12240df4fee9dc6bd6c964c2/coffea/hist/plot.py#L364
    data_ratio_err = np.abs(poisson_interval(data_ratio, data_region1_sumw2 / data_region2_sumw**2 ) - data_ratio)

    # Calculate the error in MC: Theory uncs (shape) + JES/JER + lepton uncs (flat)
    # File containing the theory variations
    theory_unc_file = './theory_uncs/vbf_z_w_gjets_theory_unc_ak4_eta0_ratio_unc_combined.root'
    f_theory = uproot.open(theory_unc_file)
    theory_uncs_up = {}
    theory_uncs_down = {}
    if region2 == 'wlv':
        theory_unc_names_up = [unc.decode('utf-8') for unc in f_theory.keys() if 'zll_over_wlnu' in unc.decode('utf-8') and 'up' in unc.decode('utf-8') and f'{year}' in unc.decode('utf-8')]
        theory_unc_names_down = [unc.decode('utf-8') for unc in f_theory.keys() if 'zll_over_wlnu' in unc.decode('utf-8') and 'down' in unc.decode('utf-8') and f'{year}' in unc.decode('utf-8')]

        for unc in theory_unc_names_up:
            theory_uncs_up[unc] = f_theory[unc].values - 1
        for unc in theory_unc_names_down:
            theory_uncs_down[unc] = f_theory[unc].values - 1

        # Combine the theory uncertainties
        combined_theory_uncs_up = sum(theory_uncs_up.values())
        combined_theory_uncs_down = sum(theory_uncs_down.values())
        
        print(combined_theory_uncs_up)
        print(combined_theory_uncs_down)

        # Add in the flat uncertainties
        jes_jer_unc = 0.02
        mc_uncs_up = 1 + np.hypot(combined_theory_uncs_up, jes_jer_unc)
        mc_uncs_down = 1 - np.hypot(combined_theory_uncs_down, jes_jer_unc)

        print(mc_uncs_up)
        print(mc_uncs_down)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }
    if variable in ['ak4_eta0', 'ak4_eta1']:
        bins = h.axis('jeteta').edges()

    # Plot ratio in data
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(data_ratio, bins, ax=ax, yerr=data_ratio_err, histtype='errorbar', label=f'{ylabels[tag]} Data', **data_err_opts)

    mc_err_opts = {
        'linestyle':'-',
        'color':'crimson'
    }

    # Plot ratio in MC
    hep.histplot(mc_ratio, bins, ax=ax, histtype='step', label=f'{ylabels[tag]} MC', **mc_err_opts)

    # Add in the uncertainty band on MC ratio, containing theory + systematic uncertainties
    hep.histplot(mc_ratio*mc_uncs_up, bins, ax=ax, histtype='step', label='MC up', color='gray')
    hep.histplot(mc_ratio*mc_uncs_down, bins, ax=ax, histtype='step', label='MC down', color='gray')
    # ax.fill_between(bins[:-1], mc_ratio*mc_uncs_up, mc_ratio*mc_uncs_down, where='post', color='gray', alpha=0.5)

    # Aesthetics for the top panel
    ax.set_ylabel(ylabels[tag])
    if tag in ['zll_over_wlv']:
        ax.set_ylim(0,0.3) 
    ax.legend()

    # Calculate the double ratio: Data ratio / MC ratio
    dratio = data_ratio / mc_ratio
    dratio[np.isinf(dratio) | np.isnan(dratio)] = 1.

    # In the ratio pad, plot the % uncertainty in data ratio
    data_ratio_err_percent = data_ratio_err / data_ratio

    # Plot the ratio
    hep.histplot(dratio, bins, yerr=data_ratio_err_percent, ax=rax, histtype='errorbar', **data_err_opts)
    rax.set_xlabel(xlabels[variable])
    rax.set_ylabel('Ratio in Data / Ratio in MC')
    rax.grid(True)
    rax.set_ylim(0.5,1.5)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.], 'r--')
    rax.set_xlim(xlim)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    outpath = pjoin(outdir, f'{tag}_data_validation_{variable}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    create_data_validation_plot(acc, tag='zll_over_wlv', outtag=outtag, region1='zll', 
                region2='wlv', year=2017, variable='ak4_eta0'
                )

if __name__ == '__main__':
    main()


